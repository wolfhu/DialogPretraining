
import argparse
from data_utils import processors
import os
import logging
import torch
from config import DistilConfig
import shutil
import json

logger = logging.getLogger(__name__)
args = None

def parse(opt = None):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str,
                        # required=True,
                        help="Model type")
    parser.add_argument("--task_name", default=None, type=str,
                        # required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str,
                        # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--student_config", type=str,
                        # required=True,
                        help="Path to the student configuration.")
    parser.add_argument("--temperature", default=2.0, type=float, help="Temperature for the softmax temperature.")
    parser.add_argument("--cache_dir", default="cache", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_false',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument(
        "--mlm_mask_prop",default=0.15,type=float,
        help="Proportion of tokens for which we need to make a prediction.",
    )
    parser.add_argument("--word_mask", default=0.8, type=float, help="Proportion of tokens to mask out.")
    parser.add_argument("--word_keep", default=0.1, type=float, help="Proportion of tokens to keep.")
    parser.add_argument("--word_rand", default=0.1, type=float, help="Proportion of tokens to randomly replace.")
    parser.add_argument(
        "--mlm_smoothing",default=0.7,type=float,
        help="Smoothing parameter to emphasize more rare tokens (see XLM, similar to word2vec).",
    )

    parser.add_argument(
        "--freeze_pos_embs",action="store_true",help="Freeze positional embeddings during distillation. For student_type in ['roberta', 'gpt2'] only.",
    )
    parser.add_argument(
        "--freeze_token_type_embds",action="store_true", help="Freeze token type embeddings during distillation if existent. For student_type in ['roberta'] only.")
    parser.add_argument(
        "--group_by_size",action="store_true",
        help="If true, group sequences that have similar length into the same batch. Default is false.",)
    parser.add_argument("--restrict_ce_to_mask",action="store_true",
                        help="If true, compute the distilation loss only the [MLM] prediction distribution.",)

    parser.add_argument("--n_epoch", type=int, default=3, help="Number of pass on the whole dataset.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_percentage", default=0.1, type=float, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_false',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--bert_without_grad', action='store_true')
    parser.add_argument('--later_model_type', default='bilinear')

    global args
    if not opt:
        args = parser.parse_args()
    else:
        args = parser.parse_args(opt)

def args_check(args):
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     logger.warning("Output directory () already exists and is not empty.")
    # if args.gradient_accumulation_steps < 1:
    #     raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
    #                         args.gradient_accumulation_steps))
    # if not args.do_train and not args.do_predict:
    #     raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))
    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu

def setup_args_from_output(args, config):
    if config is None:
        return

    for key in config.__dict__:
        setattr(args, key, getattr(config, key))

    if hasattr(config, 'teacher_model') and config.teacher_model:
        args.teacher_model_type = config.teacher_model['model_type']
        args.teacher_model_config = config.teacher_model['config_file']
        args.teacher_model_name_or_path = config.teacher_model['model_name_or_path']

    if hasattr(config, 'student_model') and config.student_model:
        args.student_model_type = config.student_model['model_type']
        args.student_config = config.student_model['config_file']
        args.student_model_name_or_path = config.student_model['model_name_or_path']


def path_check(args):
    if args.is_master:
        if os.path.exists(args.output_dir):
            if not args.overwrite_output_dir:
                raise ValueError(
                    f"Serialization dir {args.output_dir} already exists, but you have not precised wheter to overwrite it"
                    "Use `--force` if you want to overwrite it"
                )
            else:
                shutil.rmtree(args.output_dir)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info(f"Experiment will be dumped and logged in {args.output_dir}")

        # # SAVE PARAMS #
        # logger.info(f"Param: {args}")
        # with open(os.path.join(args.output_dir, "parameters.json"), "w") as f:
        #     json.dump(vars(args), f, indent=4)

if __name__ == '__main__':
    parse()