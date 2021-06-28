

from .utils_glue import *
from .utils_dc import *
from .metrics import compute_metrics
from .lm_seq_dataset import  LmSeqsDataset
from .grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups

import torch
from torch.utils.data import TensorDataset
import numpy as np
from collections import Counter

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    'nq': NqSentProcessor,
    'nq_para': NqParaProcessor,
    'chat': ChatQAProcessor,
    'chat-generation': ChatGenerationProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    'nq': 'classification',
    'nq_para': 'classification',
    'chat':'classification',
    'chat-generation': 'generation',
}

GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "nq":2,
    'nq_para':2,
    'chat':2
}

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        # list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)

        if args.model_type == 'dc-bert':
            features = dc_convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                         cls_token_at_end=bool(args.model_type in ['xlnet']),
                                         # xlnet has a cls token at the end
                                         cls_token=tokenizer.cls_token,
                                         cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                         sep_token=tokenizer.sep_token,
                                         sep_token_extra=bool(args.model_type in ['roberta']),
                                         # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                         pad_on_left=bool(args.model_type in ['xlnet']),  # pad on the left for xlnet
                                         pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                         pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                         )
        else:
            features = glue_convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer,
                                                       output_mode,
                                                       cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                       # xlnet has a cls token at the end
                                                       cls_token=tokenizer.cls_token,
                                                       cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                       sep_token=tokenizer.sep_token,
                                                       # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                       pad_on_left=bool(args.model_type in ['xlnet']),
                                                       # pad on the left for xlnet
                                                       pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                           0],
                                                       pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                       )


        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # if args.local_rank == 0 and not evaluate:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if args.model_type == 'dc-bert':
        all_input_ids_a = torch.tensor([f.input_ids_a for f in features], dtype=torch.long)
        all_input_mask_a = torch.tensor([f.input_mask_a for f in features], dtype=torch.long)
        all_segment_ids_a = torch.tensor([f.segment_ids_a for f in features], dtype=torch.long)

        all_input_ids_b = torch.tensor([f.input_ids_b for f in features], dtype=torch.long)
        all_input_mask_b = torch.tensor([f.input_mask_b for f in features], dtype=torch.long)
        all_segment_ids_b = torch.tensor([f.segment_ids_b for f in features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    elif output_mode == 'generation':
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    if args.model_type == 'dc-bert':
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                all_input_ids_a, all_input_mask_a, all_segment_ids_a,
                                all_input_ids_b, all_input_mask_b, all_segment_ids_b)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def create_lm_seq_dataset(args,  tokenizer):
    logger.info('create lm_seq_dataset')
    # processor = processors[task]()
    data = []
    counter = Counter()

    with open(args.data_dir, 'r', encoding='utf-8') as f:
        for i, line in enumerate( f):
            if args.mlm:
                cls_id, sep_id = '[CLS]', '[SEP]'
            else:
                cls_id, sep_id ='[BOS]', '[EOS]'
            line = cls_id + line + sep_id
            ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize( line))
            data.append(np.array(ids))
            counter.update(ids)

    dataset = LmSeqsDataset(args, data)
    logger.info("Counting occurences for MLM.")
    counts = [0] * tokenizer.vocab_size
    for k, v in counter.items():
        counts[k] = v

    return dataset, counts






