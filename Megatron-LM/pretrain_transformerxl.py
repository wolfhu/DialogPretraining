# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain TransformerXL"""

from datetime import datetime
import os
import random
import math
import numpy as np
import torch

from arguments import get_parser, get_args_with_parser
from configure_data import configure_data
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import MemTransformerLM
from model import transformerxl_get_params_for_weight_decay_optimization
from model import DistributedDataParallel as LocalDDP
from model.transformer_xl_modeling import TransfoXLInitConfig
import mpu
from apex.optimizers import FusedAdam as Adam
from utils import Timers
from utils import save_checkpoint
from utils import load_checkpoint
from utils import report_memory
from utils import print_args
from utils import print_params_min_max_norm
from utils import print_rank_0
from utils import enable_adlr_autoresume
from utils import check_adlr_autoresume_termination


def add_transformerxl_config(parser):
    group = parser.add_argument_group('transformerxl', 'xl configurations')

    # model config
    group.add_argument('--transoxl_n_layer', type=int, default=12,
                        help='number of total layers')
    group.add_argument('--transoxl_n_head', type=int, default=10,
                        help='number of heads')
    group.add_argument('--transoxl_d_head', type=int, default=50,
                        help='head dimension')
    group.add_argument('--transoxl_d_embed', type=int, default=-1,
                        help='embedding dimension')
    group.add_argument('--transoxl_d_model', type=int, default=500,
                        help='model dimension')
    group.add_argument('--transoxl_d_inner', type=int, default=1000,
                        help='inner dimension in FF')
    group.add_argument('--transoxl_dropout', type=float, default=0.0,
                        help='global dropout rate')
    group.add_argument('--transoxl_dropatt', type=float, default=0.0,
                        help='attention probability dropout rate')
    
    group.add_argument('--transoxl_same_length', action='store_true',
                        help='use the same attn length for all tokens')
    group.add_argument('--transoxl_attn_type', type=int, default=0,
                        help='attention type. 0 for ours, 1 for Shaw et al,'
                        '2 for Vaswani et al, 3 for Al Rfou et al.')
    group.add_argument('--transoxl_clamp_len', type=int, default=-1,
                        help='use the same pos embeddings after clamp_len')

    group.add_argument('--transoxl_not_tied', action='store_true',
                        help='do not tie the word embedding and softmax weights')

    group.add_argument('--transoxl_div_val', type=int, default=1,
                        help='divident value for adapative input and softmax')
    group.add_argument('--transoxl_pre_lnorm', action='store_true',
                        help='apply LayerNorm to the input instead of the output')

    group.add_argument('--transoxl_tgt_len', type=int, default=70,
                        help='number of tokens to predict')
    group.add_argument('--transoxl_eval_tgt_len', type=int, default=50,
                        help='number of tokens to predict for evaluation')
    group.add_argument('--transoxl_ext_len', type=int, default=0,
                        help='length of the extended context')
    group.add_argument('--transoxl_mem_len', type=int, default=0,
                        help='length of the retained previous heads')

    group.add_argument('--transoxl_sample_softmax', type=int, default=-1,
                        help='number of samples in sampled softmax')

    # model init
    group.add_argument('--transoxl_init', default='normal', type=str,
                        help='parameter initializer to use.')
    group.add_argument('--transoxl_init_range', type=float, default=0.1,
                        help='parameters initialized by U(-init_range, init_range)')
    group.add_argument('--transoxl_init_std', type=float, default=0.02,
                        help='parameters initialized by N(0, init_std)')
    group.add_argument('--transoxl_proj_init_std', type=float, default=0.01,
                        help='parameters initialized by N(0, init_std)')


    group.add_argument('--mem_reset_interval', type=int, default=-1,
                        help='mems reset to None interval, default=-1:never reset mems')

    return parser


class XLDataIterator(object):
    def __init__(self, dataloader, bptt, ext_len=None):
        self.data_iterator = iter(dataloader)

        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.data_key = 'text'

    def iterator(self):
        while True:
            reset_value = 1
            test_count = 0
            while test_count < 10:
                data = next(self.data_iterator)
                if data and self.data_key in data:
                    data = data[self.data_key]
                    break
                test_count += 1
            
            if test_count >= 10:
                break

            cur_data_iterator = LMOrderedIterator(data, self.bptt, self.ext_len)
            for batch in cur_data_iterator:
                batch['reset'] = torch.LongTensor([reset_value])
                yield batch
                reset_value = 0


class LMOrderedIterator(object):
    def __init__(self, data, bptt, ext_len=None):
        """
            data -- size (bsz, seq_len) -- the LongTensor is strictly ordered
        """

        bsz, seq_len = data.size()

        self.bsz = bsz
        self.seq_len = seq_len

        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        # Evenly divide the data across the bsz batches.
        self.data = data.t().contiguous()

    def get_fixlen_iter(self):
        for i in range(0, self.seq_len - 1, self.bptt):
            if i + 1 + self.bptt > self.seq_len:
                break
            
            end_idx = i + self.bptt
            beg_idx = max(0, i - self.ext_len)

            data = self.data[beg_idx:end_idx]
            target = self.data[i+1:i+1+self.bptt]
            yield {'data':data.contiguous(), 'target': target.contiguous()}

    def __iter__(self):
        return self.get_fixlen_iter()


def get_args():
    parser = get_parser()

    parser = add_transformerxl_config(parser)

    args = get_args_with_parser(parser)

    # args.runtime_batch_size = args.batch_size
    # args.batch_size = 1

    return args


def get_model(args):
    """Build the model."""

    print_rank_0('building TransformerXL model ...')

    args.transoxl_tied = not args.transoxl_not_tied

    if args.transoxl_d_embed < 0:
        args.transoxl_d_embed = args.transoxl_d_model


    cutoffs, tie_projs = [], [False]

    init_config = TransfoXLInitConfig(init=args.transoxl_init, init_range=args.transoxl_init_range, 
                                        init_std=args.transoxl_init_std, proj_init_std=args.transoxl_proj_init_std)

    model = MemTransformerLM(args.vocab_size, 
                            args.transoxl_n_layer, 
                            args.transoxl_n_head, 
                            args.transoxl_d_model, 
                            args.transoxl_d_head, 
                            args.transoxl_d_inner, 
                            args.transoxl_dropout, 
                            args.transoxl_dropatt, 
                            tie_weight=args.transoxl_tied, 
                            d_embed=args.transoxl_d_embed, 
                            div_val=args.transoxl_div_val, 
                            tie_projs=tie_projs, 
                            pre_lnorm=args.transoxl_pre_lnorm, 
                            tgt_len=args.transoxl_tgt_len, 
                            ext_len=args.transoxl_ext_len, 
                            mem_len=args.transoxl_mem_len, 
                            cutoffs=cutoffs, 
                            same_length=args.transoxl_same_length, 
                            attn_type=args.transoxl_attn_type, 
                            clamp_len=args.transoxl_clamp_len, 
                            sample_softmax=args.transoxl_sample_softmax, 
                            checkpoint_activations=args.checkpoint_activations,
                            init_config=init_config)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        args.DDP_type = torch.nn.parallel.distributed.DistributedDataParallel
        model = args.DDP_type(model, device_ids=[i], output_device=i,
                              process_group=mpu.get_data_parallel_group())
    elif args.DDP_impl == 'local':
        args.DDP_type = LocalDDP
        model = args.DDP_type(model)
    else:
        print_rank_0('Unknown DDP implementation specified: {}. '
                     'Exiting.'.format(args.DDP_impl))
        exit()

    return model


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (args.DDP_type, FP16_Module)):
        model = model.module

    param_groups = transformerxl_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    # Use Adam.
    optimizer = Adam(param_groups,
                     lr=args.lr, weight_decay=args.weight_decay)

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale':args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               min_lr=args.min_lr,
                               use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
                               override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def setup_model_and_optimizer(args):
    """Setup model and optimizer."""

    model = get_model(args)
    
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)

        base_model = model
        while isinstance(base_model, (args.DDP_type, FP16_Module)):
            base_model = base_model.module
        base_model.tie_weights()

        torch.distributed.barrier()
        if mpu.get_data_parallel_rank() == 0:
            print('successfully loaded tie weights')
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler


def get_batch(data_iterator, args, timers):
    '''
    '''
    # Items and their type.
    keys = ['data', 'target', 'reset']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()

    # disable model parallel here
    if args.model_parallel_size != 1:
        data_b = mpu.broadcast_data(keys, data, datatype)
    else:
        data_b =  {key: data[key].cuda() for key in keys}

    # Unpack.
    tokens = data_b['data'].long().contiguous()
    labels = data_b['target'].long().contiguous()
    is_reseted = data_b['reset'].long().contiguous()[0] >= 1

    return tokens, labels, is_reseted


def update_mems(mems, is_reseted, args, timers):
    if is_reseted:
        return tuple()

    if args.model_parallel_size == 1:
        return mems

    if not mems:
        return mems

    keys = ['mems']
    data = {'mems':torch.stack(mems)}
    datatype = mems[0].dtype

    data_b = mpu.broadcast_data(keys, data, datatype)

    mems = data_b['mems']

    return [mems[i].contiguous() for i in range(mems.size()[0])]


def forward_step(data_iterator, mems, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, is_reseted = get_batch(data_iterator, args, timers)
    mems = update_mems(mems=mems, is_reseted=is_reseted, args=args, timers=timers)
    timers('batch generator').stop()

    # Forward model.
    output = model(tokens, labels, mems)
    loss, mems = output[0], output[1:]
    loss = loss.float().mean().type_as(loss)

    return loss, mems


def backward_step(optimizer, model, lm_loss, args, timers):
    """Backward step."""

    # Total loss.
    loss = lm_loss

    # Backward pass.
    optimizer.zero_grad()
    if args.fp16:
        optimizer.backward(loss, update_master_grads=False)
    else:
        loss.backward()

    # Reduce across processes.
    lm_loss_reduced = lm_loss

    reduced_losses = lm_loss.view(1)
    torch.distributed.all_reduce(reduced_losses.data)
    reduced_losses.data = reduced_losses.data / args.world_size
    if args.DDP_impl == 'local':
        timers('allreduce').start()
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=args.fp32_allreduce)
        timers('allreduce').stop()
    lm_loss_reduced = reduced_losses

    # Update master gradients.
    if args.fp16:
        optimizer.update_master_grads()

    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0:
        if not args.fp16:
            mpu.clip_grad_norm(model.parameters(), args.clip_grad)
        else:
            optimizer.clip_master_grads(args.clip_grad)

    return lm_loss_reduced


def train_step(data_iterator, mems, model, optimizer, lr_scheduler,
               args, timers):
    """Single training step."""

    # Forward model for one step.
    timers('forward').start()
    lm_loss, mems = forward_step(data_iterator=data_iterator, 
                                mems=mems, 
                                model=model, 
                                args=args, 
                                timers=timers)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    lm_loss_reduced = backward_step(optimizer, model, lm_loss, args, timers)
    timers('backward').stop()

    # Update parameters.
    timers('optimizer').start()
    optimizer.step()
    timers('optimizer').stop()

    # Update learning rate.
    skipped_iter = 0
    if not (args.fp16 and optimizer.overflow):
        lr_scheduler.step()
    else:
        skipped_iter = 1

    return lm_loss_reduced, skipped_iter, mems


def train(model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args, writer):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0

    # Iterations.
    iteration = args.iteration
    skipped_iters = 0

    mems = tuple()

    timers('interval time').start()
    report_memory_flag = True
    while iteration < args.train_iters:

        lm_loss, skipped_iter, mems = train_step(data_iterator=train_data_iterator, 
                                            mems=mems, 
                                            model=model,
                                            optimizer=optimizer, 
                                            lr_scheduler=lr_scheduler, 
                                            args=args, 
                                            timers=timers)
        skipped_iters += skipped_iter
        iteration += 1

        # Update losses.
        current_lm_loss = lm_loss.data.detach().float()
        total_lm_loss += current_lm_loss

        # Logging.

        if args.DDP_impl == 'torch':
            timers_to_log = ['forward', 'backward', 'optimizer',
                            'batch generator', 'data loader']
        else:
            timers_to_log = ['forward', 'backward', 'allreduce', 'optimizer',
                             'batch generator', 'data loader']

        learning_rate = optimizer.param_groups[0]['lr']

        if writer and args.rank == 0:
            writer.add_scalar('learning_rate', learning_rate, iteration)
            writer.add_scalar('train_loss', current_lm_loss, iteration)
            if args.fp16:
                writer.add_scalar('loss_scale', optimizer.loss_scale, iteration)
            normalizer = iteration % args.log_interval
            if normalizer == 0:
                normalizer = args.log_interval
            timers.write(timers_to_log, writer, iteration,
                         normalizer=normalizer)

        if iteration % args.log_interval == 0:
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            if writer and args.rank == 0:
                writer.add_scalar('iteration_time',
                                  elapsed_time / args.log_interval, iteration)
            log_string = ' iteration {:8d}/{:8d} |'.format(iteration,
                                                            args.train_iters)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
                elapsed_time * 1000.0 / args.log_interval)
            log_string += ' learning rate {:.3E} |'.format(learning_rate)
            log_string += ' lm loss {:.6E} |'.format(avg_lm_loss)
            log_string += ' lm ppl: {:.6E} |'.format(math.exp(min(20, avg_lm_loss)))
            if args.fp16:
                log_string += ' loss scale {:.1f} |'.format(
                    optimizer.loss_scale)
            print_rank_0(log_string)
            total_lm_loss = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(iteration))
                report_memory_flag = False
            timers.log(timers_to_log, normalizer=args.log_interval)

        # Autoresume
        if (iteration % args.adlr_autoresume_interval == 0) and args.adlr_autoresume:
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              lr_scheduler, args)

        # Checkpointing
        if args.save and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, val_data_iterator, model, args,
                                       writer, iteration, timers, False)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, iteration), flush=True)
            exit()

    return iteration, skipped_iters

def evaluate(data_iterator, model, args, timers, verbose=False):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    base_model = model
    while isinstance(base_model, (args.DDP_type, FP16_Module)):
        base_model = base_model.module

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.transoxl_mem_len == 0:
        base_model.reset_length(args.transoxl_eval_tgt_len,
            args.transoxl_ext_len+args.transoxl_tgt_len-args.transoxl_eval_tgt_len, args.transoxl_mem_len)
    else:
        base_model.reset_length(args.transoxl_eval_tgt_len,
            args.transoxl_ext_len, args.transoxl_mem_len+args.transoxl_tgt_len-args.transoxl_eval_tgt_len)

    total_lm_loss = 0

    with torch.no_grad():
        mems = tuple()
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
            # Forward evaluation.
            lm_loss, mems = forward_step(data_iterator=data_iterator, 
                                        mems=mems, 
                                        model=model, 
                                        args=args, 
                                        timers=timers)
            # Reduce across processes.
            if isinstance(model, args.DDP_type):
                torch.distributed.all_reduce(lm_loss.data)
                lm_loss.data = lm_loss.data / args.world_size

            total_lm_loss += lm_loss.data.detach().float().item()

    
    # Switch back to the training mode
    base_model.reset_length(args.transoxl_tgt_len, args.transoxl_ext_len, args.transoxl_mem_len)
    # Move model back to the train mode.
    model.train()

    total_lm_loss /= args.eval_iters
    return total_lm_loss


def evaluate_and_print_results(prefix, data_iterator, model,
                               args, writer, iteration,
                               timers, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    lm_loss = evaluate(data_iterator, model, args, timers, verbose)
    lm_ppl = math.exp(min(20, lm_loss))
    print_rank_0('-' * 100)
    string = ' validation loss at {} | '.format(prefix)
    string += 'LM loss: {:.6E} | '.format(lm_loss)
    string += 'LM PPL: {:.6E}'.format(lm_ppl)
    length = len(string) + 1
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)

    if writer and args.rank == 0:
        writer.add_scalar('val_loss', lm_loss, iteration)
        writer.add_scalar('val_ppl', lm_ppl, iteration)

    return lm_loss


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    # disable model parallel(model parllel size == 1), enable data parallel
    # TODO : @kaizh enable model parallel
    if args.model_parallel_size != 1:
        raise ValueError('only support args.model_parallel_size == 1 now')

    mpu.initialize_model_parallel(args.model_parallel_size)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def get_train_val_test_data(args):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, val_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        data_config = configure_data()
        data_config.set_defaults(data_set_type='TransformerXL', transpose=False)
        (train_data, val_data, test_data), tokenizer = data_config.apply(
            args)
        num_tokens = tokenizer.num_tokens
        eod_token = tokenizer.get_command('eos').Id

        before = num_tokens
        after = before
        multiple = args.make_vocab_size_divisible_by * \
                   mpu.get_model_parallel_world_size()
        while (after % multiple) != 0:
            after += 1
        print_rank_0('> padded vocab (size: {}) with {} dummy '
                     'tokens (new size: {})'.format(
                         before, after - before, after))
        print_rank_0('> found end-of-document token: {}'.format(eod_token))
        token_counts = torch.cuda.LongTensor([after, eod_token, int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    eod_token = token_counts[1].item()
    args.do_train = token_counts[2].item()
    args.do_valid = token_counts[3].item()
    args.do_test = token_counts[4].item()

    return train_data, val_data, test_data, num_tokens, eod_token


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    writer = None
    if args.tensorboard_dir and args.rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir = args.tensorboard_dir)
        except ModuleNotFoundError:
            print_rank_0('WARNING: TensorBoard writing requested but is not '
                         'available (are you using PyTorch 1.1.0 or later?), '
                         'no TensorBoard logs will be written.')
            writer = None

    # Pytorch distributed.
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Pretrain TransformerXL model')
        print_args(args, writer)

    # Autoresume.
    torch.distributed.barrier()
    if args.adlr_autoresume:
        enable_adlr_autoresume(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    train_data, val_data, test_data, args.vocab_size, \
        args.eod_token = get_train_val_test_data(args)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)

    # Resume data loader if necessary.
    if args.resume_dataloader:
        if train_data is not None:
            train_data.batch_sampler.start_iter = args.iteration % \
                                                  len(train_data)
            print_rank_0('setting training data start iteration to {}'.
                         format(train_data.batch_sampler.start_iter))
        if val_data is not None:
            start_iter_val = (args.iteration // args.eval_interval) * \
                             args.eval_iters
            val_data.batch_sampler.start_iter = start_iter_val % \
                                                len(val_data)
            print_rank_0('setting validation data start iteration to {}'.
                         format(val_data.batch_sampler.start_iter))
    if train_data is not None:
        train_data_iterator = XLDataIterator(train_data, args.transoxl_tgt_len, args.transoxl_ext_len).iterator()
    else:
        train_data_iterator = None
    if val_data is not None:
        val_data_iterator = XLDataIterator(val_data, args.transoxl_eval_tgt_len, args.transoxl_ext_len).iterator()
    else:
        val_data_iterator = None

    #TODO: figure out how to properly set this especially when resuming training
    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            iteration, skipped = train(model, optimizer,
                                       lr_scheduler,
                                       train_data_iterator,
                                       val_data_iterator,
                                       timers, args, writer)

        if args.do_valid:
            prefix = 'the end of training for val data'
            val_loss = evaluate_and_print_results(prefix, val_data_iterator,
                                                  model, args, writer, iteration,
                                                  timers, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer,
                        lr_scheduler, args)

    if test_data is not None:
        test_data_iterator = XLDataIterator(test_data, args.transoxl_eval_tgt_len, args.transoxl_ext_len).iterator()
    else:
        test_data_iterator = None

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, test_data_iterator,
                                   model, args, None, 0, timers, True)


if __name__ == "__main__":
    main()
