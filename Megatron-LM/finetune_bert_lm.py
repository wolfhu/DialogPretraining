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

"""Pretrain BERT"""

from datetime import datetime
import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from arguments import get_args
from configure_data import configure_data
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import BertModel
from model import get_params_for_weight_decay_optimization
from model import gpt2_get_params_for_weight_decay_optimization
from model import DistributedDataParallel as LocalDDP
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

def get_model(args):
    """Build the model."""

    print_rank_0('building BERT model ...')
    model = BertModel(args)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)
        if args.fp32_embedding:
            model.module.model.embedding.word_embeddings.float()
            model.module.model.embedding.position_embeddings.float()
            model.module.model.embedding.token_type_embeddings.float()
        if args.fp32_tokentypes:
            model.module.model.embedding.token_type_embeddings.float()
        if args.fp32_layernorm:
            for name, _module in model.named_modules():
                if 'LayerNorm' in name:
                    _module.float()

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
    # layers = model.model.bert.encoder.layer
    # generator_layer = model.model.g_encoder.layer
    # discriminitor = model.model.d_encoder.layer
    #
    # # pooler = model.model.bert.pooler
    # lmheads = model.model.cls.predictions
    # # nspheads = model.model.cls.seq_relationship
    # embeddings = model.model.embedding
    # discriminitor_ouput_layer = model.model.d_outputlayer
    # param_groups = []
    # param_groups += list(get_params_for_weight_decay_optimization(generator_layer))
    # param_groups += list(get_params_for_weight_decay_optimization(discriminitor))
    # # param_groups += list(get_params_for_weight_decay_optimization(pooler))
    # # param_groups += list(get_params_for_weight_decay_optimization(nspheads))
    # param_groups += list(get_params_for_weight_decay_optimization(embeddings))
    # param_groups += list(get_params_for_weight_decay_optimization(discriminitor_ouput_layer))
    # param_groups += list(get_params_for_weight_decay_optimization(
    #     lmheads.transform))
    # param_groups[1]['params'].append(lmheads.bias)
    param_groups = get_params_for_weight_decay_optimization(model)
    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    # Use Adam.
    betas = (0.9, 0.999)
    optimizer = Adam(param_groups, betas=betas,
                     lr=args.lr, weight_decay=args.weight_decay)

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
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
    else:
        args.iteration = 0
    # optimizer.param_groups[0]['lr'] /=10
    return model, optimizer, lr_scheduler


def get_batch(data_iterator, timers, train_data = None):
    ''' get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    '''
    # Items and their type.
    keys = ['text', 'types', 'label', 'mask', 'mask_labels', 'pad_mask']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is None and train_data is not None:
        data_iterator = iter(train_data)
    if data_iterator is not None:
        try:
            data = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_data)
            data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['text'].long()
    types = data_b['types'].long()
    label = data_b['label'].long()
    loss_mask = data_b['mask'].float()
    lm_labels = data_b['mask_labels'].long()
    padding_mask = data_b['pad_mask'].byte()

    return tokens, types, label, loss_mask, lm_labels, padding_mask


def forward_step(data_iterator, model, args, timers, data = None):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, types, next_sentence, loss_mask, lm_labels, \
        padding_mask = get_batch(data_iterator, timers, train_data=data)
    timers('batch generator').stop()

    sentence_label = next_sentence
    # Forward model.
    discriminator_output = model(tokens, types, 1-padding_mask, lm_labels,
                        checkpoint_activations=args.checkpoint_activations)

    # print(' discriminator_output shape is ', discriminator_output.shape)
    # print(' sentence_label shape is ', sentence_label.shape)
    dsp_loss = F.cross_entropy(discriminator_output.view(-1, 3).contiguous().float(),
                               sentence_label.view(-1).contiguous(),
                               ignore_index=-1)

    _logits = torch.argmax(F.softmax(discriminator_output), -1)
    _acc = (_logits == sentence_label).float().mean()

    # losses = mpu.vocab_parallel_cross_entropy(
    #     output.contiguous().float(), lm_labels.contiguous())
    # loss_mask = loss_mask.contiguous()
    # lm_loss = torch.sum(
    #     losses.view(-1) * loss_mask.view(-1).float()) / loss_mask.sum()

    return dsp_loss, dsp_loss, _acc


def backward_step(optimizer, model, lm_loss, nsp_loss, args, timers, factor = 50):
    """Backward step."""

    # Total loss.
    loss = nsp_loss

    # Backward pass.
    optimizer.zero_grad()
    if args.fp16:
        optimizer.backward(loss, update_master_grads=False)
    else:
        loss.backward()

    # Reduce across processes.
    lm_loss_reduced = lm_loss
    nsp_loss_reduced = nsp_loss

    reduced_losses = torch.cat((lm_loss.view(1), nsp_loss.view(1)))
    torch.distributed.all_reduce(reduced_losses.data)
    reduced_losses.data = reduced_losses.data / args.world_size
    if args.DDP_impl == 'local':
        timers('allreduce').start()
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=args.fp32_allreduce)
        timers('allreduce').stop()
    lm_loss_reduced = reduced_losses[0]
    nsp_loss_reduced = reduced_losses[1]

    # Update master gradients.
    if args.fp16:
        optimizer.update_master_grads()

    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0:
        if not args.fp16:
            mpu.clip_grad_norm(model.parameters(), args.clip_grad)
        else:
            optimizer.clip_master_grads(args.clip_grad)

    return lm_loss_reduced, nsp_loss_reduced


def train_step(data_iterator, model, optimizer, lr_scheduler,
               args, timers, train_data = None):
    """Single training step."""

    # Forward model for one step.
    timers('forward').start()
    lm_loss, nsp_loss, dsp_acc = forward_step(data_iterator, model,
                                     args, timers, data= train_data)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    lm_loss_reduced, nsp_loss_reduced = backward_step(optimizer, model, lm_loss,
                                                      nsp_loss, args, timers)
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

    return lm_loss_reduced, nsp_loss_reduced, skipped_iter, dsp_acc


def train(model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args, writer, train_data = None, val_data = None):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0
    total_nsp_loss = 0.0
    total_dsp_acc = 0.0

    # Iterations.
    iteration = args.iteration
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    while iteration < args.train_iters:

        if train_data_iterator is None and train_data is not None:
            train_data_iterator =  iter(train_data)
        if val_data_iterator is None and val_data is not None:
            val_data_iterator = iter(val_data)

        lm_loss, nsp_loss, skipped_iter, dsp_acc = train_step(train_data_iterator,
                                                     model,
                                                     optimizer,
                                                     lr_scheduler,
                                                     args, timers,
                                                     train_data = train_data
                                                     )
        skipped_iters += skipped_iter
        iteration += 1

        # Update losses.
        current_lm_loss = lm_loss.data.detach().float()
        current_nsp_loss = nsp_loss.data.detach().float()
        current_dsp_acc = dsp_acc.data.detach().float()
        total_lm_loss += current_lm_loss
        total_nsp_loss += current_nsp_loss
        total_dsp_acc += current_dsp_acc

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
            writer.add_scalar('lm_loss', current_lm_loss, iteration)
            writer.add_scalar('nsp_loss', current_nsp_loss, iteration)
            writer.add_scalar('dsp_acc', current_dsp_acc, iteration)
            if args.fp16:
                writer.add_scalar('loss_scale', optimizer.loss_scale, iteration)
            normalizer = iteration % args.log_interval
            if normalizer == 0:
                normalizer = args.log_interval
            timers.write(timers_to_log, writer, iteration,
                         normalizer=normalizer)

        if iteration % args.log_interval == 0:
            avg_nsp_loss = total_nsp_loss.item() / args.log_interval
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            avg_ds_acc = total_dsp_acc.item() / args.log_interval
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
            log_string += ' dsp loss {:.6E} |'.format(avg_nsp_loss)
            if args.fp16:
                log_string += ' loss scale {:.1f} |'.format(
                    optimizer.loss_scale)
            log_string += ' dsp acc {:.6E} |'.format(avg_ds_acc)
            print_rank_0(log_string)
            total_nsp_loss = 0.0
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
                                       writer, iteration, timers, False, val_data=val_data)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, iteration), flush=True)
            exit()

    return iteration, skipped_iters


def evaluate(data_iterator, model, args, timers, verbose = False, val_data = None):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_lm_loss = 0
    total_nsp_loss = 0
    total_dsp_acc = 0

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
            # Forward evaluation.
            lm_loss, nsp_loss, dsp_acc = forward_step(data_iterator, model,
                                             args, timers, data=val_data)
            # Reduce across processes.
            if isinstance(model, args.DDP_type):
                reduced_losses = torch.cat((lm_loss.view(1), nsp_loss.view(1), dsp_acc.view(1)))
                torch.distributed.all_reduce(reduced_losses.data)
                reduced_losses.data = reduced_losses.data/args.world_size
                lm_loss = reduced_losses[0]
                nsp_loss = reduced_losses[1]
                dsp_acc = reduced_losses[2]

            total_lm_loss += lm_loss.data.detach().float().item()
            total_nsp_loss += nsp_loss.data.detach().float().item()
            total_dsp_acc += dsp_acc.data.detach().float().item()

    # Move model back to the train mode.
    model.train()

    total_lm_loss /= args.eval_iters
    total_nsp_loss /= args.eval_iters
    total_dsp_acc /= args.eval_iters
    return total_lm_loss, total_nsp_loss, total_dsp_acc


def evaluate_and_print_results(prefix, data_iterator, model,
                               args, writer, iteration,
                               timers, verbose=False, factor = 50, val_data = None):
    """Helper function to evaluate and dump results on screen."""
    lm_loss, nsp_loss, dsp_acc = evaluate(data_iterator, model,
                                 args, timers, verbose, val_data=val_data)
    val_loss =  nsp_loss
    print_rank_0('-' * 100)
    string = ' validation loss at {} | '.format(prefix)
    string += 'LM loss: {:.6E} | '.format(lm_loss)
    string += 'DSP loss: {:.6E} | '.format(nsp_loss)
    string += 'DSP acc : {:.6E} | '.format(dsp_acc)
    string += 'total loss: {:.6E}'.format(val_loss)
    length = len(string) + 1
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)

    if writer and args.rank == 0:
        writer.add_scalar('val_lm_loss', lm_loss, iteration)
        writer.add_scalar('val_dsp_loss', nsp_loss, iteration)
        writer.add_scalar('val_dsp_acc', dsp_acc, iteration)
        writer.add_scalar('val_total_loss', val_loss, iteration)

    return val_loss


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    print("device count is {0}, and local rank is {1}".format(device, args.local_rank))
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
        ds_type = 'BERT-finetune'
        data_config.set_defaults(data_set_type=ds_type, transpose=False)
        (train_data, val_data, test_data), tokenizer = data_config.apply(args)
        before = tokenizer.num_tokens
        after = before
        multiple = args.make_vocab_size_divisible_by * \
                   mpu.get_model_parallel_world_size()
        while (after % multiple) != 0:
            after += 1
        print_rank_0('> padded vocab (size: {}) with {} dummy '
                     'tokens (new size: {})'.format(
                         before, after - before, after))
        # Need to broadcast num_tokens and num_type_tokens.
        token_counts = torch.cuda.LongTensor([after,
                                              tokenizer.num_type_tokens,
                                              int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    num_type_tokens = token_counts[1].item()
    args.do_train = token_counts[2].item()
    args.do_valid = token_counts[3].item()
    args.do_test = token_counts[4].item()

    return train_data, val_data, test_data, num_tokens, num_type_tokens


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
        print('Pretrain BERT model')
        print_args(args, writer)

    # Autoresume.
    torch.distributed.barrier()
    if args.adlr_autoresume:
        enable_adlr_autoresume(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    train_data, val_data, test_data, args.tokenizer_num_tokens, \
        args.tokenizer_num_type_tokens = get_train_val_test_data(args)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)

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
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None

    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            iteration, skipped = train(model, optimizer,
                                       lr_scheduler,
                                       train_data_iterator,
                                       val_data_iterator,
                                       timers, args, writer,
                                       train_data = train_data,
                                       val_data = val_data
                                       )
        if args.do_valid:
            prefix = 'the end of training for val data'
            val_loss = evaluate_and_print_results(prefix, val_data_iterator,
                                                  model, args, writer, iteration,
                                                  timers, False, val_data=val_data)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

    if test_data is not None:
        test_data_iterator = iter(test_data)
    else:
        test_data_iterator = None

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, test_data_iterator,
                                   model, args, None, 0, timers, True, val_data=test_data)


if __name__ == "__main__":
    main()
