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

"""GPT-2 model."""

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
import random
from copy import deepcopy

from transformers import mpu
from fp16 import FP16_Module
# from model import DistributedDataParallel as LocalDDP
from transformers.mpu import model_parallel_cuda_manual_seed
from transformers.mpu.model_utils import Timers
from transformers.mpu.model_utils import save_checkpoint
from transformers.mpu.model_utils import load_checkpoint
from transformers.mpu.model_utils import report_memory
from transformers.mpu.model_utils import print_args
from transformers.mpu.model_utils import print_params_min_max_norm
from transformers.mpu.model_utils import print_rank_0
from transformers.mpu.model_utils import enable_adlr_autoresume
from transformers.mpu.model_utils import check_adlr_autoresume_termination


def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class MegatronGPT2Model(torch.nn.Module):
    """GPT-2 Language model.

    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True):

        super(MegatronGPT2Model, self).__init__()
        set_random_seed(1234)

        self.parallel_output = parallel_output

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(max_sequence_length,
                                                      hidden_size)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self.tokentype_embeddings = None
        self.hidden_size = hidden_size

        # Initialize the position embeddings.
        init_method(self.position_embeddings.weight)

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # Transformer
        self.transformer = mpu.GPT2ParallelTransformer(num_layers,
                                                       hidden_size,
                                                       num_attention_heads,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers)

    def add_tokentype_embeddings(self, num_tokentypes):
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes),
                  flush=True)
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes,
                                                       self.hidden_size)

    # 这里更改了 Megatron-LM GPT2 的 forward 参数
    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            use_cache=True,
    ):
        layer_past = past
        get_present = use_cache
        tokentype_ids = token_type_ids
        head_mask = None
        inputs_embeds = None
        batch_size, seq_length = input_ids.size()
        # Attention mask (lower triangular).
        # TODO 这里没有reset attention mask 和 position ids
        # huggingface的attention mask有问题，覆盖掉
        att_mask_batch = 1
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=input_ids.device)).view(
            att_mask_batch, 1, seq_length, seq_length)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        # Transformer.
        transformer_output = self.transformer(embeddings, attention_mask,
                                              layer_past=layer_past,
                                              get_present=get_present)
        if get_present:
            transformer_output, presents = transformer_output

        # Parallel logits.
        transformer_output_parallel = mpu.copy_to_model_parallel_region(
            transformer_output)
        logits_parallel = F.linear(transformer_output_parallel,
                                   self.word_embeddings.weight)

        if self.parallel_output:
            output = logits_parallel
        else:
            output = mpu.gather_from_model_parallel_region(logits_parallel)
        if get_present:
            output = [output, presents]
        return output

    @classmethod
    def get_model(cls, args, tokenizer):
        """Build the model."""

        print_rank_0('building GPT2 model ...')
        initialize_distributed(args)
        model = cls(num_layers=args.num_layers,
                    vocab_size=args.vocab_size,
                    hidden_size=args.hidden_size,
                    num_attention_heads=args.num_attention_heads,
                    embedding_dropout_prob=args.hidden_dropout,
                    attention_dropout_prob=args.attention_dropout,
                    output_dropout_prob=args.hidden_dropout,
                    max_sequence_length=args.max_position_embeddings,
                    checkpoint_activations=args.checkpoint_activations,
                    checkpoint_num_layers=args.checkpoint_num_layers,
                    parallel_output=True)

        # EOS token
        model.eod_token = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0]

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
            args.DDP_type = DistributedDataParallel
            model = args.DDP_type(model)
        elif args.DDP_impl == 'no_distributed':
            pass
        else:
            print_rank_0('Unknown DDP implementation specified: {}. '
                         'Exiting.'.format(args.DDP_impl))
            exit()

        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 填充args
        args = kwargs['config']
        tokenizer = kwargs['tokenizer']
        model = cls.get_model(args, tokenizer)
        model.args = args
        iteration = load_checkpoint(model, None, None, args)
        model.iteration = iteration
        return model

    def save_pretrained(self, output_dir):
        # 填充args
        self.args.save = output_dir
        save_checkpoint(self.iteration, self, None,
                        None, self.args)


class TheseusMegatronGPT2Model(MegatronGPT2Model):

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True):
        super(TheseusMegatronGPT2Model, self).__init__(num_layers,
                                                       vocab_size,
                                                       hidden_size,
                                                       num_attention_heads,
                                                       embedding_dropout_prob,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       max_sequence_length,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers=checkpoint_num_layers,
                                                       parallel_output=parallel_output)
        # Theseus Transformer
        self.transformer = mpu.TheseusGPT2ParallelTransformer(num_layers,
                                                              hidden_size,
                                                              num_attention_heads,
                                                              attention_dropout_prob,
                                                              output_dropout_prob,
                                                              checkpoint_activations,
                                                              checkpoint_num_layers)



class TheseusMegatronGPT2ModelForGeneration(TheseusMegatronGPT2Model):

    def forward(self,
                input_ids=None,
                past=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                use_cache=True,):
        output = super(TheseusMegatronGPT2ModelForGeneration, self).forward(input_ids=input_ids,
                                                                            past=past,
                                                                            attention_mask=attention_mask,
                                                                            token_type_ids=token_type_ids,
                                                                            position_ids=position_ids,
                                                                            head_mask=head_mask,
                                                                            inputs_embeds=inputs_embeds,
                                                                            labels=labels,
                                                                            use_cache=use_cache)
        get_present = use_cache
        if get_present:
            logits, presents = output
        losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(),
                                                  labels)
        # Loss mask.
        loss_mask = torch.ones(input_ids.size(), dtype=torch.float, device=input_ids.device)
        loss_mask[input_ids == self.eod_token] = 0.0
        loss_mask = loss_mask.view(-1)
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return (loss, logits)


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    # device = args.rank % torch.cuda.device_count()
    # if args.local_rank is not None:
    #     device = args.local_rank
    # torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=1, rank=0,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)


def gpt2_get_params_for_weight_decay_optimization(module):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params


class DistributedDataParallel(Module):

    def __init__(self, module):
        super(DistributedDataParallel, self).__init__()
        self.warn_on_half = True if dist._backend == dist.dist_backend.GLOO else False

        self.module = module
        self.data_parallel_group = mpu.get_data_parallel_group()
        src_rank = mpu.get_model_parallel_rank()
        for p in self.module.parameters():
            if torch.is_tensor(p):
                dist.broadcast(p, src_rank, group=self.data_parallel_group)

        def allreduce_params(reduce_after=True, no_scale=False, fp32_allreduce=False):
            if (self.needs_reduction):
                self.needs_reduction = False
                buckets = {}
                for name, param in self.module.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        tp = (param.data.type())
                        if tp not in buckets:
                            buckets[tp] = []
                        buckets[tp].append(param)
                if self.warn_on_half:
                    if torch.cuda.HalfTensor in buckets:
                        print("WARNING: gloo dist backend for half parameters may be extremely slow." +
                              " It is recommended to use the NCCL backend in this case.")
                        self.warn_on_half = False
                for tp in buckets:
                    bucket = buckets[tp]
                    grads = [param.grad.data for param in bucket]
                    coalesced = _flatten_dense_tensors(grads)
                    if fp32_allreduce:
                        coalesced = coalesced.float()
                    if not no_scale and not reduce_after:
                        coalesced /= dist.get_world_size(group=self.data_parallel_group)
                    dist.all_reduce(coalesced, group=self.data_parallel_group)
                    torch.cuda.synchronize()
                    if not no_scale and reduce_after:
                        coalesced /= dist.get_world_size(group=self.data_parallel_group)
                    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                        buf.copy_(synced)

        self.hook_handles = []
        self.hooks = []
        for param in list(self.module.parameters()):
            def allreduce_hook(*unused):
                Variable._execution_engine.queue_callback(allreduce_params)
        #    handle = param.register_hook(allreduce_hook)
        # self.hooks.append(allreduce_hook)
        # self.hook_handles.append(handle)
        self.allreduce_params = allreduce_params

    def forward(self, *inputs, **kwargs):
        self.needs_reduction = True
        return self.module(*inputs, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # [h.remove() for h in self.hook_handles]
        sd = self.module.state_dict(destination, prefix, keep_vars)
        # for handle, hook in zip(self.hook_handles, self.hooks):
        #     d = handle.hooks_dict_ref()
        #     d[handle.id] = hook

        return sd

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)

    '''
    def _sync_buffers(self):
        buffers = list(self.module._all_buffers())
        if len(buffers) > 0:
            # cross-node buffer sync
            flat_buffers = _flatten_dense_tensors(buffers)
            dist.broadcast(flat_buffers, 0)
            for buf, synced in zip(buffers, _unflatten_dense_tensors(flat_buffers, buffers)):
                buf.copy_(synced)
    def train(self, mode=True):
        # Clear NCCL communicator and CUDA event cache of the default group ID,
        # These cache will be recreated at the later call. This is currently a
        # work-around for a potential NCCL deadlock.
        if dist._backend == dist.dist_backend.NCCL:
            dist._clear_group_cache()
        super(DistributedDataParallel, self).train(mode)
        self.module.train(mode)
    '''


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)
