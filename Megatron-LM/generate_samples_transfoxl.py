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

"""Sample Generate TransformerXL"""

import os
import random
import json
import copy
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from utils import Timers

from pretrain_transformerxl import initialize_distributed
from pretrain_transformerxl import set_random_seed
from pretrain_transformerxl import get_train_val_test_data
from pretrain_transformerxl import get_args
from pretrain_transformerxl import get_model

from utils import load_checkpoint
from data_utils import make_tokenizer
from configure_data import configure_data
import mpu

from fp16 import FP16_Module
from model import MemTransformerLM
from model import DistributedDataParallel as DDP

from utils import print_rank_0


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)

    if args.load is not None:
        _ = load_checkpoint(
            model, None, None, args)

        base_model = model
        while isinstance(base_model, (args.DDP_type, FP16_Module)):
            base_model = base_model.module
        base_model.tie_weights()

    return model

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        #convert to 1D
        # logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value
        #going back to 2D
        # logits=logits.view(1, -1).contiguous()
    
    return logits

def generate_samples_interactive(model, tokenizer, args):

    print_frequency = 24 

    context_count=0
    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs=0

            if mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                raw_text = input("\nContext prompt (stop to exit) >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("\nContext prompt (stop to exit) >>> ")
           
                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.EncodeAsIds(raw_text).tokenization
                    context_length = len(context_tokens)
            else:
                context_tokens = tokenizer.EncodeAsIds("EMPTY TEXT").tokenization
                context_length = len(context_tokens)
            
            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            start_time = time.time()
            token_stream = get_token_stream(model, context_tokens, tokenizer, args)
            for counter, decode_tokens in enumerate(token_stream):
                # token_end = decode_tokens.find("<|endoftext|>")
                # if token_end > 0:
                #     break
                
                cur_decode_tokens = decode_tokens.cpu().numpy().tolist()[context_length:]

                if mpu.get_model_parallel_rank() == 0 and counter % print_frequency == 0:
                    os.system('clear')
                    #print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                    print("\nContext:", raw_text, flush=True)
                    trim_decode_tokens = tokenizer.DecodeIds(cur_decode_tokens)
                    #print("\nGPT2:", trim_decode_tokens, flush=True)
                    #print("\nMegatron-LM:", trim_decode_tokens.replace("\n", "\n\n"), flush=True)
                    print("\nMegatron-LM:", trim_decode_tokens, flush=True)

            if mpu.get_model_parallel_rank() == 0 and cur_decode_tokens:
                os.system('clear')
                #print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                print("\nContext:", raw_text, flush=True)
                trim_decode_tokens = tokenizer.DecodeIds(cur_decode_tokens)
                #print("\nGPT2:", trim_decode_tokens, flush=True)
                #print("\nMegatron-LM:", trim_decode_tokens.replace("\n", "\n\n"), flush=True)
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

            raw_text = None

            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1
            
            if mpu.get_model_parallel_rank() == 0:
                input("\nPress any key to continue >>>")


def pad_batch(tokens, tokenizer, args):
    context_length = len(tokens)
    if context_length < args.seq_length:
        tokens.extend([0]*(args.seq_length-context_length))
    return tokens, context_length

def get_token_stream(model, context_tokens, tokenizer, args):

    context_tokens, context_length = pad_batch(context_tokens, tokenizer, args)

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor([context_length])

    torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
    torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())

    org_context_length = context_length

    batch_token_iterator = sample_sequence_batch(model, context_tokens_tensor, context_length, tokenizer, args)
    for tokens, length in batch_token_iterator:
        yield tokens[:length]

def sample_sequence_batch(model, context_tokens, context_length, tokenizer, args):
    base_model = model
    while isinstance(base_model, (args.DDP_type, FP16_Module)):
        base_model = base_model.module

    org_context_length = context_length

    maxlen = args.seq_length - 1
    if maxlen > (org_context_length + args.out_seq_length):
        maxlen = org_context_length + args.out_seq_length

    batch_size = 1

    model.eval()
    with torch.no_grad():
        tokens = context_tokens

        target = tuple()
        mems = tuple()

        # update mems:
        base_model.reset_length(args.transoxl_mem_len, 0, args.transoxl_mem_len)
        for i in range(0, context_length, args.transoxl_mem_len):
            if i + args.transoxl_mem_len >= context_length:
                break
            startid = i
            endid = i + args.transoxl_mem_len
            output = model(tokens[startid:endid].view(-1,1).contiguous(), target, mems)
            logits, mems = output[0], output[1:]

        base_model.reset_length(args.transoxl_tgt_len, 0, args.transoxl_mem_len)
        while context_length <= (maxlen):
            endid = context_length
            tgt_len = endid % args.transoxl_mem_len
            if tgt_len == 0:
                tgt_len = args.transoxl_mem_len
            startid = endid - tgt_len
            base_model.reset_length(tgt_len, 0, args.transoxl_mem_len)
            output = model(tokens[startid:endid].view(-1,1).contiguous(), target, mems)
            logits, new_mems = output[0], output[1:]

            if context_length % args.transoxl_mem_len == 0:
                mems = new_mems

            logits = logits[-1,:,:].view(1, -1).contiguous()

            if args.greedy:
                prev = torch.argmax(logits, dim=-1).view(-1)
            else:
                logits /= args.temperature
                logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)            
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1).view(-1)


            tokens[context_length] = prev[-1]
            context_length += 1

            yield tokens, context_length

def prepare_tokenizer(args):

    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    tokenizer = make_tokenizer(**tokenizer_args)

    args.tokenizer_num_tokens = tokenizer.num_tokens
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    after = tokenizer.num_tokens
    multiple = args.make_vocab_size_divisible_by * \
                   mpu.get_model_parallel_world_size()
    if multiple != 0:
        while (after % multiple) != 0:
            after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer

def main():
    """Main training program."""

    print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    #get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    #setting default batch size to 1
    # args.batch_size = 1

    args.device = torch.cuda.current_device()

    #generate samples
    args.batch_size = 1
    generate_samples_interactive(model, tokenizer, args)

if __name__ == "__main__":
    main()



