

import argparse
import logging
import os
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler,SequentialSampler, \
    TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import glob

from models import DCBertForSequenceClassification, TheseusBertForSequenceClassification
from models import ConstantReplacementScheduler, LinearReplacementScheduler
from transformers import BertModel, BertForSequenceClassification, BertTokenizer, BertConfig,\
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from transformers import TheseusMegatronGPT2ModelForGeneration, MegatronGPT2Tokenizer, MegatronGPT2Config
from transformers import WEIGHTS_NAME
from transformers import AdamW, get_linear_schedule_with_warmup

from data_utils import compute_metrics, load_and_cache_examples, \
    processors, output_modes
from config.distiller_config import DistilConfig
from utils import init_gpu_params

import arguments

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BERT = 'bert'
ROBERTA = 'roberta'
DC_BERT = 'dc-bert'
THESEUS_BERT = 'theseus-bert'
TINY_BERT = 'tiny-bert'
RNN = 'rnn'

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'dc-bert': (BertConfig, DCBertForSequenceClassification, BertTokenizer),
    'theseus-bert': (BertConfig, TheseusBertForSequenceClassification, BertTokenizer),
    'tiny-bert': (BertConfig, TheseusBertForSequenceClassification, BertTokenizer),
    'theseus-gpt2': (MegatronGPT2Config, TheseusMegatronGPT2ModelForGeneration, MegatronGPT2Tokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir=args.output_dir, comment= '_'.join([args.task_name, args.model_type, str(args.learning_rate), str(args.num_train_epochs)]) )

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank in[-1, 0 ] else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    if args.bert_without_grad:
        for n, p in model.named_parameters():
            if 'bert' in n:
                p.requires_grad = False

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # warm up steps
    num_warmup_steps = args.warmup_steps
    if num_warmup_steps == 0:
        num_warmup_steps = int(args.warmup_percentage * t_total)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    if args.model_type == 'theseus-bert':
        # Replace rate scheduler
        if args.theseus_scheduler['scheduler_type'] == 'none':
            replacing_rate_scheduler = ConstantReplacementScheduler(bert_encoder=model.bert.encoder,
                                                                    replacing_rate= args.theseus_scheduler['replacing_rate'],
                                                                    replacing_steps= args.theseus_scheduler['steps_for_replacing'])
        elif args.theseus_scheduler['scheduler_type']  == 'linear':
            replacing_rate_scheduler = LinearReplacementScheduler(bert_encoder=model.bert.encoder,
                                                                  base_replacing_rate= args.theseus_scheduler['replacing_rate'],
                                                                  k= args.theseus_scheduler['scheduler_linear_k'])

    if args.model_type == 'theseus-gpt2':
        # Replace rate scheduler
        if args.theseus_scheduler['scheduler_type'] == 'none':
            replacing_rate_scheduler = ConstantReplacementScheduler(bert_encoder=model.transformer,
                                                                    replacing_rate= args.theseus_scheduler['replacing_rate'],
                                                                    replacing_steps= args.theseus_scheduler['steps_for_replacing'])
        elif args.theseus_scheduler['scheduler_type']  == 'linear':
            replacing_rate_scheduler = LinearReplacementScheduler(bert_encoder=model.transformer,
                                                                  base_replacing_rate= args.theseus_scheduler['replacing_rate'],
                                                                  k= args.theseus_scheduler['scheduler_linear_k'])

    if args.fp16:
        #setup for fp16
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        if args.model_type in ['theseus-bert', 'theseus-gpt2']:
            logger.warning("[BERT-of-Theseus] We haven't tested our model under multi-gpu. Please be aware!")
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank not in [-1,0]:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank not in [-1,0] else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if 'distill' not in  args.model_type :
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet',
                                                                           'dc-bert'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            if args.model_type == 'dc-bert':
                inputs.update({
                    'input_ids_a': batch[4],
                    'attention_mask_a': batch[5],
                    'token_type_ids_a': batch[6] if args.model_type in ['bert', 'xlnet', 'dc-bert'] else None,
                    'input_ids_b': batch[7],
                    'attention_mask_b': batch[8],
                    'token_type_ids_b': batch[9] if args.model_type in ['bert', 'xlnet', 'dc-bert'] else None,
                    # XLM and RoBERTa don't use segment_ids
                })
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                if args.model_type in ['theseus-bert', 'theseus-gpt2']:
                    replacing_rate_scheduler.step()  # Update replace rate scheduler
                model.zero_grad()
                global_step += 1

                if (
                        args.local_rank in [-1,0] or torch.distributed.get_rank() == 0) and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank in [-1,0] and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            # print(key, value, global_step)
                            tb_writer.add_scalar('eval_{}'.format(key), value[1] if type(value) is list else value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info('average loss: {}, global step: {}'.format(
                        (tr_loss - logging_loss) / args.logging_steps, global_step))
                    logging_loss = tr_loss

                if (
                        args.local_rank in [-1,0] or torch.distributed.get_rank() == 0) and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = [args.task_name.lower()]
    eval_outputs_dirs = [args.output_dir]

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank in [-1,0] else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if 'distill' not in args.model_type :
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet', 'dc-bert'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                if args.model_type == 'dc-bert':
                    inputs.update({
                        'input_ids_a': batch[4],
                        'attention_mask_a': batch[5],
                        'token_type_ids_a': batch[6] if args.model_type in ['bert', 'xlnet', 'dc-bert'] else None,
                        'input_ids_b': batch[7],
                        'attention_mask_b': batch[8],
                        'token_type_ids_b': batch[9] if args.model_type in ['bert', 'xlnet', 'dc-bert'] else None,
                        # XLM and RoBERTa don't use segment_ids
                    })
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                logits = torch.nn.Softmax()(logits)

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            # if nb_eval_steps > 10:
            #     break
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            # if nb_eval_steps > 10:
            #     break


        eval_loss = eval_loss / nb_eval_steps
        pred_logits = preds
        logger.info('pred logits: {}'.format(len(pred_logits)))
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        elif args.output_mode == "generation":
            # TODO 目前只支持greedy search
            preds = torch.argmax(logits, dim=-1)
            batch_size, seq_length = preds.size()
            preds = preds.tolist()
            print_logits = []
            for batch_idx in range(batch_size):
                # for p in preds[batch_idx]:
                print_logits.append([logits[batch_idx, i, p].item() for i, p in enumerate(preds[batch_idx])])
        # TODO 没有实现metrics
        # result = compute_metrics(eval_task, preds, out_label_ids)
        # output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        # logger.info('eval loss: {}'.format(eval_loss))
        # with open(output_eval_file, "w") as writer:
        #     logger.info("***** Eval results {} *****".format(prefix))
        #     for key in sorted(result.keys()):
        #         logger.info("  %s = %s", key, str(result[key]))
        #         writer.write("%s = %s\n" % (key, str(result[key])))
        # result['precision'] = result['precision'].tolist()
        # result['recall'] = result['recall'].tolist()
        # result['f1'] = result['f1'].tolist()
        # result['true_sum'] = result['true_sum'].tolist()
        # result['logits'] = pred_logits.tolist()
        # result['preds'] = preds.tolist()
        # result['labels'] = out_label_ids.tolist()
        # logger.info('save result json to {}'.format(os.path.join(eval_output_dir, "eval_results.json")))
        # with open(os.path.join(eval_output_dir, "eval_results.json"), 'w', encoding='utf-8') as fout:
        #     import json
        #     json.dump(result, fout)
        result = {}
        result['preds'] = preds
        results.update(result)

        # with open(os.path.join(eval_output_dir, "eval_results_4eva.txt"), 'w', encoding='utf-8') as fout:
        #     for score, label in zip(pred_logits, out_label_ids):
        #         fout.write('\t'.join([str(s) for s in score]) + '\t' + str(label))
        #         fout.write('\n')
    return results

def main():
    arguments.parse()
    args = arguments.args

    # arguments check
    device, n_gpu = arguments.args_check(args)
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank not in [-1,0]), args.fp16)

    distiller_config = DistilConfig.from_json_file('config/distill.example.1.json')
    arguments.setup_args_from_output(args, distiller_config)

    args.output_dir = os.path.join(args.output_dir, args.model_type)

    # Set seed
    set_seed(args)

    # init gpu
    init_gpu_params(args)

    # init output dir
    arguments.path_check(args)

    #prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.output_mode = output_modes[args.task_name]

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = distiller_config.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_json_file(distiller_config.teacher_model['config_file'])
    if args.model_type == "dc-bert":
        if distiller_config.dc_bert_model_config['last_layer_model_type']:
            setattr(config, 'last_layer_model_type', distiller_config.dc_bert_model_config['last_layer_model_type'])
        else:
            logger.info("User does not set the last layer type for dc-bert, so we use default: transformer")
    if args.model_type == 'theseus-bert':
        logger.info('set theseus_scheduler for theseus-bert')
        #buchong

    tokenizer = tokenizer_class.from_pretrained(distiller_config.teacher_model['model_name_or_path'],
                                                do_lower_case=args.do_lower_case)

    if distiller_config.model_type != 'theseus-gpt2':
        model = model_class.from_pretrained(distiller_config.teacher_model['model_name_or_path'],
                                            from_tf=bool('.ckpt' in distiller_config.teacher_model['model_name_or_path']),
                                            config=config)
    elif distiller_config.model_type == 'theseus-gpt2':
        model = model_class.from_pretrained(distiller_config.teacher_model['model_name_or_path'],
                                            from_tf=bool('.ckpt' in distiller_config.teacher_model['model_name_or_path']),
                                            config=config,
                                            tokenizer=tokenizer)


    if 'distill' in distiller_config.model_type.lower():
        raise ValueError("This script is not used for distill type. Use run_distill instead")

    if args.model_type == 'dc-bert':
        model.init_top_layer_from_bert()
    if distiller_config.model_type == 'theseus-bert':
        #Initialize successor BERT weights
        scc_n_layer = model.bert.encoder.scc_n_layer
        model.bert.encoder.scc_layer = nn.ModuleList(
            [deepcopy(model.bert.encoder.layer[ix]) for ix in range(scc_n_layer)])
    if distiller_config.model_type == 'theseus-gpt2':
        scc_n_layer = config.scc_n_layer
        if config.fp16:
            model.module.transformer.scc_n_layer = scc_n_layer
            model.module.transformer.prd_n_layer = config.num_layers
            model.module.transformer.compress_ratio = model.module.transformer.prd_n_layer // model.module.transformer.scc_n_layer
            model.module.transformer.scc_layer = nn.ModuleList(
                [deepcopy(model.module.transformer.layers[ix]) for ix in range(scc_n_layer)])
        else:
            model.transformer.scc_n_layer = scc_n_layer
            model.transformer.prd_n_layer = config.num_layers
            model.transformer.compress_ratio = model.transformer.prd_n_layer // model.transformer.scc_n_layer
            model.transformer.scc_layer = nn.ModuleList(
                [deepcopy(model.transformer.layers[ix]) for ix in range(scc_n_layer)])


    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    prefix = "loaded ckpt"
    # TODO 这里读取ckpt，依然会重新初始化scc layers，不会读训练好的scc layers
    result = evaluate(args, model, tokenizer, prefix=prefix)
    preds = result['preds']
    print('results: ')
    for pred in preds:
        print(' '.join(tokenizer.convert_ids_to_tokens(pred)))
    # result = dict((k + '_{}'.format(prefix), v) for k, v in result.items())
    # results.update(result)

    # #training
    # if distiller_config.do_train:
    #     train_dataset = load_and_cache_examples(args, distiller_config.task_name, tokenizer, evaluate=False)
    #     global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    #     logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    #
    # # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if args.do_train and (args.local_rank  in [-1,0] or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)
    #
    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = model.module if hasattr(model,
    #                                             'module') else model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)
    #
    #     # distiller_config.save_pretrained(distiller_config.output_dir)
    #
    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    #
    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = model_class.from_pretrained(args.output_dir)
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     model.to(args.device)

    # # Evaluation
    # results = {}
    # if distiller_config.do_eval and (args.local_rank  in [-1,0] or torch.distributed.get_rank() == 0):
    #     logger.info("start eval")
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in
    #                            sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         # TODO 这里读取ckpt，依然会重新初始化scc layers，不会读训练好的scc layers
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         result = evaluate(args, model, tokenizer, prefix=global_step)
    #         result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         results.update(result)
    #
    #     return results

if __name__ == '__main__':
    main()
