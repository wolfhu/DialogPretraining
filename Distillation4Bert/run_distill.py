

import argparse
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
import math

from models import DCBertForSequenceClassification, TheseusBertForSequenceClassification
from models import ConstantReplacementScheduler, LinearReplacementScheduler
from models import Distiller, DistillBertForMaskedLM, DistillBertForSequenceClassification
from models import BLSTMForSequenceClassification
from transformers import BertModel, BertForSequenceClassification, BertTokenizer, BertConfig,\
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, BertForMaskedLM
from transformers import WEIGHTS_NAME
from transformers import AdamW, get_linear_schedule_with_warmup

from data_utils import compute_metrics, load_and_cache_examples, \
    processors, output_modes,create_lm_seq_dataset

import arguments as arguments
from config.distiller_config import DistilConfig
from utils import init_gpu_params
from config import DISTILLER_MODEL_CONFIGURATION, DistillConfigutation

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO )
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_CLASSES = {
    'bert-general-distill': (BertConfig, DistillBertForMaskedLM, BertTokenizer),
    'bert-task-distill': (BertConfig, DistillBertForSequenceClassification, BertTokenizer),
    'rnn-distill': (BertConfig, BLSTMForSequenceClassification, BertTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank in [-1, 0] else DistributedSampler(eval_dataset)
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
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet', 'dc-bert'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                outputs = model(**inputs)
                logits = outputs[0]
                logits = torch.nn.Softmax()(logits)

                if args.output_mode == "classification":
                    loss_fct = nn.CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(logits.view(-1, model.num_labels), inputs['labels'].view(-1))
                elif args.output_mode == "regression":
                    loss_fct = nn.MSELoss()
                    tmp_eval_loss = loss_fct(logits.view(-1), inputs['labels'].view(-1))

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
        result = compute_metrics(eval_task, preds, out_label_ids)
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        logger.info('eval loss: {}'.format(eval_loss))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        result['precision'] = result['precision'].tolist()
        result['recall'] = result['recall'].tolist()
        result['f1'] = result['f1'].tolist()
        result['true_sum'] = result['true_sum'].tolist()
        result['logits'] = pred_logits.tolist()
        result['preds'] = preds.tolist()
        result['labels'] = out_label_ids.tolist()
        logger.info('save result json to {}'.format(os.path.join(eval_output_dir, "eval_results.json")))
        with open(os.path.join(eval_output_dir, "eval_results.json"), 'w', encoding='utf-8') as fout:
            import json
            json.dump(result, fout)
        results.update(result)

        with open(os.path.join(eval_output_dir, "eval_results_4eva.txt"), 'w', encoding='utf-8') as fout:
            for score, label in zip(pred_logits, out_label_ids):
                fout.write('\t'.join([str(s) for s in score]) + '\t' + str(label))
                fout.write('\n')
    return results

def main():
    arguments.parse()
    args = arguments.args

    # arguments check
    device, n_gpu = arguments.args_check(args)
    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    distiller_config = DistilConfig.from_json_file('config/distill.example.1.json')
    arguments.setup_args_from_output(args, distiller_config)

    #delete this line when checkin
    args.output_dir = os.path.join(distiller_config.output_dir, distiller_config.model_type)

    # Set seed
    set_seed(args)

    #init gpu
    init_gpu_params(args)

    #init output dir
    arguments.path_check(args)

    # prepare task
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

    if 'distill' not in args.model_type.lower():
        return None

    student_config_class, student_model_class, _ = MODEL_CLASSES[args.student_model_type]
    teacher_config_class, teacher_model_class, teacher_tokenizer_class = MODEL_CLASSES[args.teacher_model_type]

    # TOKENIZER #
    tokenizer = teacher_tokenizer_class.from_pretrained(args.teacher_model_name_or_path)
    special_tok_ids = {}
    for tok_name, tok_symbol in tokenizer.special_tokens_map.items():
        idx = tokenizer.all_special_tokens.index(tok_symbol)
        special_tok_ids[tok_name] = tokenizer.all_special_ids[idx]
    logger.info(f"Special tokens {special_tok_ids}")
    args.special_tok_ids = special_tok_ids
    args.max_model_input_size = args.max_seq_length

    #teacher model
    logger.info(f'Start init teacher model from {args.teacher_model_name_or_path}')
    teacher_config = teacher_config_class.from_json_file(args.teacher_model_config)
    teacher_config.output_hidden_states = True
    teacher_config.output_attention = True
    teacher_config.num_labels = num_labels
    teacher_model = teacher_model_class.from_pretrained(args.teacher_model_name_or_path, config=teacher_config)
    logger.info('Successfully init teacher model')

    #student_model
    logger.info(f'Start init student model from {args.student_config}')
    student_config = student_config_class.from_json_file(args.student_config)
    student_config.output_hidden_states = True
    student_config.output_attention = True
    student_config.num_labels = num_labels
    # setattr(student_config, 'is_student', True)
    if args.student_model_name_or_path:
        logger.info(f'Loading pretrained weights for student model from {args.student_model_name_or_path}')
        student_model = student_model_class.from_pretrained(args.student_model_name_or_path, config=student_config,
                                                            is_student = True)
    else:
        student_model = student_model_class(student_config, is_student = True, fit_size = teacher_config.hidden_size)
    logger.info('Successfully init student model')

    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    teacher_model.to(args.device)
    student_model.to(args.device)

    logger.info("Training/evaluation parameters %s", distiller_config)

    if args.model_type in DISTILLER_MODEL_CONFIGURATION:
        configuration = DISTILLER_MODEL_CONFIGURATION[args.model_type]
    else:
        configuration = DistillConfigutation.from_json_file(args.configuration_file)

    arguments.setup_args_from_output(args, configuration)
    token_probs = None
    if configuration.general_distill:
        train_dataset, counts = create_lm_seq_dataset(args, tokenizer)

        token_probs = np.maximum(counts, 1) ** -args.mlm_smoothing
        for idx in special_tok_ids.values():
            token_probs[idx] = 0.0  # do not predict special tokens
        token_probs = torch.from_numpy(token_probs)
    elif configuration.task_distill:
        train_dataset = load_and_cache_examples(args, distiller_config.task_name, tokenizer, \
                                                evaluate=False)
    else:
        raise AttributeError("must set general_distill or task_distill to be true")

    # training
    if args.do_train:
        # train_dataset = load_and_cache_examples(distiller_config, distiller_config.task_name, teacher_tokenizer, evaluate=False)
        args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

        #torch.cuda.empty_cache()
        distiller = Distiller(
            params=args,
            dataset=train_dataset,
            token_probs=token_probs,
            student=student_model,
            teacher=teacher_model,
            distill_configuration=configuration
        )

        distiller.train()
        logger.info('wow')

    if configuration.general_distill and args.do_eval:
        raise ValueError("We do not suggest evaluation in general distill, please take it after the finetune or "
                         "task distill")

    if args.do_eval:
        results = evaluate(args, student_model, tokenizer, prefix="")
        logger.info(f"Evaluation result {results}")

if __name__ == '__main__':
    main()