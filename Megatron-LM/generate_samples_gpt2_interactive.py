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

"""Sample Generate GPT2"""

import os
import random
import json
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
import re
from arguments import get_parser, get_args_with_parser
from utils import Timers
import unicodedata
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from pretrain_gpt2 import get_train_val_test_data
from pretrain_gpt2 import get_masks_and_position_ids
from utils import load_checkpoint
from data_utils import make_tokenizer
from configure_data import configure_data
import mpu
import editdistance

from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0

from flask import Flask, request, jsonify

MODEL_INSTANCE = None
TOKENIZER_INSTANCE = None
ARGS_INSTANCE = None
MODEL_OUT_DTYPE = None

app = Flask(__name__)

def get_args():
    parser = get_parser()

    parser = add_server_config(parser)
    parser = add_generate_interactive_config(parser)

    args = get_args_with_parser(parser)

    return args

def add_server_config(parser):
    group = parser.add_argument_group('generate server', 'server configurations')
    group.add_argument('--server-port', default='9999', type=str,
                        help='flask port to use.')

    return parser

def add_generate_interactive_config(parser):
    group = parser.add_argument_group('generate interactive configurations', 'generate interactive configurations')
    group.add_argument('--max-context-turn', default=7, type=int,
                        help='context turns to use')
    group.add_argument('--repeat-count', default=20, type=int,
                        help='max candidate sample case')
    group.add_argument('--not-norm-lm-with-length', action='store_true',
                       help='not-norm-lm-with-length')
    group.add_argument('--write-all-results', action='store_true',
                       help='write all results to file')
    group.add_argument('--infer-end-token', default='eos', type=str, 
                        choices=['eos', 'sep'],
                        help='infer end token')
    return parser


def get_model(args):
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      max_sequence_length=args.max_position_embeddings,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=False,
                      no_parallel=args.model_parallel_size)
    if args.model_parallel_size == 1 or mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank() if args.model_parallel_size > 1 else 0,
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if args.world_size > 1:
        model = DDP(model)

    return model

def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)
    if args.load is not None:
        _ = load_checkpoint(
            model, None, None, args)

    return model

def get_batch(context_tokens, args):
    tokens = context_tokens
    # tokens = tokens.view(args.batch_size, -1).contiguous()
    device = args.device
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, attention_mask, position_ids

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

def top_p_sampling(logits, top_k, top_p, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k <= 0 and top_p <= 0.:
        position_ids = torch.arange(logits.size(1), dtype=torch.long,
                                device=logits.device)
        position_ids = position_ids.unsqueeze(0).expand_as(logits)

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        logits, position_ids = torch.topk(logits, top_k)
        # print('top_p_sample logits', logits, 'positionid', position_ids)


    if top_p > 0.0:
        # convert to 1D
        # logits=logits.view(logits.size()[1]).contiguous()

        if top_k <= 0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        else:
            sorted_logits = logits
            sorted_indices = torch.arange(sorted_logits.size(1), dtype=torch.long,
                                    device=sorted_logits.device)
            sorted_indices = sorted_indices.unsqueeze(0).expand_as(sorted_logits)

        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()

        # keep also the first token above the threshold
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

        # going back to 2D
        # logits=logits.view(1, -1).contiguous()
    
    # print('top_p_sample logits', logits, 'positionid', position_ids)
    return logits, position_ids

def parse_request():
    req_data = request.data
    if not req_data:
        return None
    req_json = json.loads(req_data.decode("utf-8"))
    if not req_json:
        return None

    return req_json

    args = ARGS_INSTANCE

    input_text = req_json.get('input', None)
    if not input_text:
        return None

    end_token = req_json.get('eos', 'eos')
    out_seq_length = req_json.get('out_seq_length', args.out_seq_length)

    temperature = req_json.get('temperature', args.temperature)
    top_k = req_json.get('top_k', args.top_k)
    top_p = req_json.get('top_p', args.top_p)


    return input_text, end_token, out_seq_length

@app.route('/generate', methods=['POST'])
def generate():
    args = ARGS_INSTANCE
    tokenizer = TOKENIZER_INSTANCE
    model = MODEL_INSTANCE

    if args.world_size != 1:
        raise Exception('Only support world size == 1')

    req_json = parse_request()
    if not req_json:
        return jsonify({"Error": "Bad Request"}), 400

    input_text = req_json.get('input', None)
    if not input_text:
        return None

    end_token = req_json.get('eos', 'eos')
    end_token_id = tokenizer.get_command(end_token).Id

    out_seq_length = req_json.get('out_seq_length', args.out_seq_length)

    temperature = req_json.get('temperature', args.temperature)
    top_k = req_json.get('top_k', args.top_k)
    top_p = req_json.get('top_p', args.top_p)

    mem_length = req_json.get('mem_length', -1)
    # print('mem length', mem_length)
    max_mem_length = max(args.seq_length // 2, args.seq_length - 100)
    # print("max mem length", max_mem_length)
    if mem_length >= max_mem_length:
        return jsonify({"Error": "mem_lenght must less than {}, but we get {}".format(max_mem_length, mem_length)}), 400

    if out_seq_length > args.seq_length and mem_length <= 0:
        return jsonify({"Error": "When mem_length <= 0, out_seq_length must less than or equal to {}. However, we get out_seq_length={}, mem_length={}. Plese set mem_length value in your request, if you want to get a result length > {}, ".format(args.seq_length, out_seq_length, mem_length, args.seq_length)}), 400

    do_convert_sep = req_json.get('do_convert_sep', True)

    model.eval()

    output_tokens = []
    input_text = input_text.replace('\n', ' [SEP] ')
    context_tokens = tokenizer.EncodeAsIds(input_text).tokenization

    if len(context_tokens) >= args.seq_length:
        return jsonify({"Error": "input bpe tokens length must less than {}, but we get {}".format(args.seq_length, len(context_tokens))}), 400

    while True:
        with torch.no_grad():
            context_length = len(context_tokens)

            maxlen = args.seq_length
            if maxlen > (context_length + out_seq_length):
                maxlen = context_length + out_seq_length

            decode_tokens = context_tokens

            token_stream = get_token_stream(model, [context_tokens], tokenizer, args, temperature, top_k, top_p)
            for counter, decode_tokens in enumerate(token_stream):
                decode_tokens, _, probs = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()
 
                if decode_tokens[-1] == end_token_id or len(decode_tokens) >= maxlen or len(decode_tokens) + len(output_tokens) >= out_seq_length:
                    break

            output_tokens.extend(decode_tokens)
            if mem_length <= 0 or len(output_tokens) >= out_seq_length or output_tokens[-1] == end_token_id:
                break

            chunk_id = _find_valid_chunk_id(output_tokens, tokenizer, mem_length, max_mem_length)
            # print('chunk id', chunk_id)
            context_tokens = output_tokens[chunk_id:]
            # print('context token len', len(context_tokens))
            output_tokens = output_tokens[:chunk_id]

    out_seq_length = min(len(output_tokens), out_seq_length)
    tokens = tokenizer.text_tokenizer.convert_ids_to_tokens(output_tokens[:out_seq_length])

    tokens = format_zh_bpe_result(tokens)
    if do_convert_sep:
        return ''.join(tokens).replace(' [SEP] ', '\n')
    return ''.join(tokens)

def _find_valid_chunk_id(tokens, tokenizer, mem_length, max_mem_length):
    sep_id = tokenizer.get_command('sep').Id

    start_id = max(len(tokens) - mem_length - 1, -1)
    min_id = max(len(tokens) - max_mem_length - 1, -1)
    while start_id > min_id:
        if tokens[start_id] == sep_id:
            break

        token = tokenizer.IdToToken(tokens[start_id])
        if len(token) == 1 and (_is_punctuation(token)):
            break

        start_id -= 1

    return start_id + 1

def generate_chat_samples_interactive(model, tokenizer, args):
    args.infer_end_token = 'sep'
    model.eval()
    with torch.no_grad():
        context_texts = []

        while True:
            if args.world_size > 1:
                torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs=0

            if args.model_parallel_size == 1 or mpu.get_model_parallel_rank() == 0:
                raw_text = input("\n Me:")            
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("\nMe (stop to exit) >>> ")

                if 'clear' == raw_text:
                    os.system('clear')
                    context_texts = []
                    continue

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_texts.append(raw_text)
                    startid = max(0, len(context_texts) - args.max_context_turn)
                    cur_context_text = context_texts[startid:]
                    raw_text = "".join([x + " [SEP] " for x in cur_context_text])
                    context_tokens = tokenizer.EncodeAsIds(raw_text).tokenization
                    context_length = len(context_tokens)
            else:
                context_tokens = tokenizer.EncodeAsIds("EMPTY TEXT").tokenization
                context_length = len(context_tokens)
            print("input {0}".format(raw_text))
            if args.world_size > 1:
                terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
                torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
                terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            results = generate_multi_case(model, tokenizer, context_tokens, context_length, args, print_intermediate_result=True)

            if (args.model_parallel_size == 1 or mpu.get_model_parallel_rank() == 0) and results:
                context_tokens, response, log_prob = get_one_legal_case(results, context_texts, args, print_intermediate_result=True)
                context_texts.append(response)
                for id, res in enumerate(context_texts):
                    if id % 2 == 0:
                        print("\nMe :", res, flush=True)
                    else:
                        print("\n\tRobot :", res, flush=True)
            if args.world_size > 1:
                torch.distributed.barrier(group=mpu.get_model_parallel_group())

def generate_chat_samples_input_from_file(model, tokenizer, args):
    args.infer_end_token = 'sep'

    if args.sample_input_file == "":
        if args.model_parallel_size == 1 or mpu.get_model_parallel_rank() == 0:
            print("args.sample_input_file CAN NOT BE empty!\n")
        return
    
    if args.model_parallel_size == 1 or mpu.get_model_parallel_rank() == 0:
        fname = open(args.sample_input_file, "r")
        all_raw_text = fname.readlines()
        input_count = len(all_raw_text)
        input_pos = 0
        if args.sample_output_file == "":
            print("Argument: sample-output-file can't be empty, setting it to\n")
            print("\t args.sample_input_file.out")
            args.sample_output_file = args.sample_input_file+".out"
        fname_out = open(args.sample_output_file, "w+")

    context_count=0
    model.eval()
    with torch.no_grad():
        while input_pos < len(all_raw_text):
            if args.world_size > 1:
                torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs=0

            if args.model_parallel_size == 1 or mpu.get_model_parallel_rank() == 0:
                raw_text = all_raw_text[input_pos].strip()
                input_pos += 1
                if input_pos == input_count:
                    raw_text = "stop"

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.EncodeAsIds(raw_text).tokenization
                    if context_tokens[-1] != tokenizer.get_command('sep').Id:
                        context_tokens.append(tokenizer.get_command('sep').Id)
                    context_length = len(context_tokens)
            else:
                context_tokens = tokenizer.EncodeAsIds("EMPTY TEXT").tokenization
                context_length = len(context_tokens)
            
            if args.world_size > 1:
                terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
                torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
                terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            # print("\nraw text split:", raw_text.split(" [SEP] "), flush=True)
            results = generate_multi_case(model, tokenizer, context_tokens, context_length, args, print_intermediate_result=False)
            
            if (args.model_parallel_size == 1 or mpu.get_model_parallel_rank() == 0) and results:
                if not args.write_all_results:
                    all_tokens, response, log_prob = get_one_legal_case(results, raw_text.split(" [SEP] "), args, print_intermediate_result=False)
                    if response:
                        print("\nInput:", raw_text, flush=True)
                        print("\tRobot:", response, log_prob, flush=True)
                        fname_out.write("{}\t{}\n".format(raw_text, response))
                else:
                    results.sort(key=lambda x:x[2], reverse=True)
                    for decode_tokens, response, log_prob in results:
                        if response:
                            print("\nInput:", raw_text, flush=True)
                            print("\tRobot:", response, log_prob, flush=True)
                            fname_out.write("{}\t{}\t{}\n".format(raw_text, response, log_prob))
 
            raw_text = None

            if args.world_size > 1:
                torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1

def generate_multi_case(model, tokenizer, context_tokens, context_length, args, print_intermediate_result=False):
        results = []
        if print_intermediate_result:
            print()
        # for _ in range(args.repeat_count):
        #     decode_tokens = [context_tokens.copy()]
        #     case_tokens, log_probs = generate_one_case(model, tokenizer, decode_tokens, context_length, args)
        #     # case_tokens, log_probs = generate_one_case(model, tokenizer, context_tokens, context_length, args)
        #     if (args.model_parallel_size == 1 or mpu.get_model_parallel_rank() == 0) and case_tokens and log_probs:
        #         if print_intermediate_result:
        #             for id in range(len(log_probs)):
        #                 print("\t", id, ":", case_tokens[id], log_probs[id], flush=True)
        #         for id in range(len(log_probs)):
        #             results.append((None, case_tokens[id], log_probs[id],))
        #     if args.world_size > 1:
        #         torch.distributed.barrier(group=mpu.get_model_parallel_group())
        decode_tokens = [context_tokens.copy() for _ in range(args.repeat_count)]
        case_tokens, log_probs = generate_one_case(model, tokenizer, decode_tokens, context_length, args)
        # case_tokens, log_probs = generate_one_case(model, tokenizer, context_tokens, context_length, args)
        if (args.model_parallel_size == 1 or mpu.get_model_parallel_rank() == 0) and case_tokens and log_probs:
            if print_intermediate_result:
                for id in range(len(log_probs)):
                    print("\t", id, ":", case_tokens[id], log_probs[id], flush=True)
            for id in range(len(log_probs)):
                results.append((None, case_tokens[id], log_probs[id],))
        if args.world_size > 1:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            
        return results

_repeat1_regex = re.compile(r"(.)\1{3,}")
_repeat2_regex = re.compile(r"(.{2,})\1{2,}")
_repeat3_regex = re.compile(r"(.{3,}).*\1")

def get_one_legal_case(generate_results, context_texts, args, print_intermediate_result=False):
    if args.model_parallel_size > 1:
        if (mpu.get_model_parallel_rank() != 0) or (not generate_results):
            return None, None, None

    global _repeat1_regex, _repeat2_regex, _repeat3_regex

    generate_results_with_penalty = []
    for id, res in enumerate(generate_results):
        if res[1] in context_texts[-1] or context_texts[-1] in res[1]:
            generate_results_with_penalty.append((res[0],res[1],res[2] - 100))
        elif len(context_texts[-1]) >= 5 and editdistance.eval(context_texts[-1], res[1]) / min(len(context_texts[-1]), len(res[1]) ) <= 0.3:
            generate_results_with_penalty.append((res[0],res[1],res[2] - 100))
        else:
            generate_results_with_penalty.append(res)

    generate_results  = generate_results_with_penalty

    generate_results.sort(key=lambda x:x[2], reverse=True)

    if print_intermediate_result:
        print("\nsorted:")
        for id, res in enumerate(generate_results):
            print("\t", id, ":", res[1], res[2], flush=True)

    cur_context_texts = context_texts[max(0, len(context_texts) - args.max_context_turn):]
    for id, cur_gen_res in enumerate(generate_results):
        decode_tokens, response, log_prob = cur_gen_res
        # if id == 0 and log_prob > -1:
        #     continue
        if response in cur_context_texts:
            continue
        if _repeat1_regex.search(response) or _repeat2_regex.search(response) or _repeat3_regex.search(response):
            continue
        return decode_tokens, response, log_prob

    decode_tokens, response, log_prob = generate_results[0]
    return decode_tokens, response, log_prob
    
    
def generate_one_case(model, tokenizer, context_tokens, context_length, args):
        decode_tokens = context_tokens

        start_time = time.time()
        token_stream = get_token_stream(model, decode_tokens, tokenizer, args, args.temperature, args.top_k, args.top_p)

        geneate_result = None
        for counter, cur_decode_result in enumerate(token_stream):
            geneate_result = cur_decode_result
            if counter > 50:
                break

        case_tokens = []
        log_probs = []
        if (args.model_parallel_size == 1 or mpu.get_model_parallel_rank() == 0) and geneate_result:
            sep_id = tokenizer.get_command('sep').Id
            gen_tokens, lengths, probs = geneate_result
            for i in range(gen_tokens.size(0)):
                decode_tokens, cur_len, cur_probs = gen_tokens[i], lengths[i].item(), probs[i]
                decode_tokens = decode_tokens[:cur_len].cpu().numpy().tolist()

                if len(context_tokens) != len(decode_tokens):
                    pre_sep_id = max(0, len(decode_tokens) - 2)
                    while pre_sep_id >= 0 and decode_tokens[pre_sep_id] != sep_id:
                        pre_sep_id = pre_sep_id - 1
                    print(decode_tokens)

                    if pre_sep_id < len(decode_tokens) - 1:
                        # trim_decode_tokens = tokenizer.DecodeIds(decode_tokens[pre_sep_id + 1:-1])
                        trim_decode_tokens = tokenizer.text_tokenizer.convert_ids_to_tokens(decode_tokens[pre_sep_id + 1:-1])
                        trim_decode_tokens = format_zh_bpe_result(trim_decode_tokens)

                        log_prob = torch.sum(torch.log(cur_probs[pre_sep_id + 1 : cur_len]))
                        cur_len = 1
                        if trim_decode_tokens:
                            cur_len = len(trim_decode_tokens) + 1
                            trim_decode_tokens = "".join(trim_decode_tokens)
                        if args.not_norm_lm_with_length:
                            cur_len = 1
                        if trim_decode_tokens:
                            case_tokens.append(trim_decode_tokens)
                            log_probs.append(log_prob.item() / cur_len)
                            # log_probs.append(log_prob / cur_len)

        return case_tokens, log_probs

def pad_batch(batch, tokenizer, args):
    pad_id = tokenizer.get_command('pad').Id
    context_tokens = []
    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        cur_tokens = tokens.copy()
        if context_length < args.seq_length:
            cur_tokens.extend([pad_id]*(args.seq_length-context_length))
        context_lengths.append(context_length)
        context_tokens.append(cur_tokens)
    return context_tokens, context_lengths

def get_token_stream(model, context_tokens, tokenizer, args, temperature, top_k, top_p):
    pad_id = tokenizer.get_command('pad').Id
    # context_length = len(context_tokens)
    # if context_length < args.seq_length:
    #     context_tokens = context_tokens + [pad_id] * (args.seq_length - context_length)
    context_tokens, context_lengths = pad_batch(context_tokens, tokenizer, args)

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)
    # context_length_tensor = torch.cuda.LongTensor([context_length])

    if args.world_size > 1:
        torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
   # import pdb
    #pdb.set_trace()
    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids=get_batch(context_tokens_tensor, args)

    # print('position id shape', position_ids.size())

    counter = 0
    org_context_length = context_length

    layer_past = None

    batch_token_iterator = sample_sequence_batch(model, context_tokens_tensor, context_length_tensor, attention_mask, position_ids, tokenizer, args, temperature, top_k, top_p)
    for tokens, lengths, probs in batch_token_iterator:
        context_length += 1
        yield tokens[:, :context_length], lengths, probs

def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1-boolean)*val1 + boolean*val2

def sample_sequence_batch(model, context_tokens, context_lengths, attention_mask, position_ids, tokenizer, args, temperature, top_k, top_p, maxlen=None):
    global MODEL_OUT_DTYPE
    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()
        eos_id = tokenizer.get_command('eos').Id
        cus_eos_id = tokenizer.get_command(args.infer_end_token).Id

        counter = 0
        org_context_length = context_length

        layer_past = None
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        probs = torch.ones_like(tokens, dtype=MODEL_OUT_DTYPE)

        if maxlen is None:
            maxlen = args.seq_length - 1
            if maxlen > (org_context_length + args.out_seq_length):
                maxlen = org_context_length + args.out_seq_length

        lengths = torch.ones([batch_size]).long().cuda()*maxlen
        
        while context_length <= (maxlen):
            if args.recompute:
                logits = model(tokens, position_ids, attention_mask)
                logits = logits[:, context_length - 1, :] 
            else:
                if counter == 0:
                    tokens2use = tokens[:, :context_length]
                    positions2use = position_ids[:, :context_length]
                else:
                    tokens2use = tokens[:, context_length - 1].view(batch_size, -1)
                    positions2use = position_ids[:, context_length - 1].view(batch_size, -1)
                logits, layer_past = model(tokens2use, positions2use, attention_mask, layer_past=layer_past, get_present=True)
                logits = logits[:, -1].view(batch_size,-1).contiguous()

            if args.greedy:
                prev = torch.argmax(logits, dim=-1).view(-1)
                log_probs = F.softmax(logits, dim=-1)
                select_logits = torch.ones([batch_size], dtype=MODEL_OUT_DTYPE).cuda()
                for i in range(batch_size):
                    select_logits[i] = log_probs[i, prev[i]]
                logits = log_probs
            else:
                logits /= temperature
                logits = top_k_logits(logits, top_k=top_k, top_p=top_p)            
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1).view(-1)
                select_logits = torch.ones([batch_size], dtype=MODEL_OUT_DTYPE).cuda()
                for i in range(batch_size):
                    select_logits[i] = log_probs[i, prev[i]]
                logits = log_probs
            # else:
            #     logits /= temperature
            #     logits, pos_topk_ids = top_p_sampling(logits, top_k=top_k, top_p=top_p)
            #     log_probs = F.softmax(logits, dim=-1)
            #     prev_fake = torch.multinomial(log_probs, num_samples=1).view(-1)
                
            #     prev = torch.zeros_like(prev_fake)
            #     select_logits = torch.ones_like(prev_fake, dtype=MODEL_OUT_DTYPE)
            #     for i in range(batch_size):
            #         prev[i] = pos_topk_ids[i, prev_fake[i]]
            #         select_logits[i] = log_probs[i, prev_fake[i]]

            #     logits = log_probs


            started = context_lengths <= context_length
            tokens[:, context_length] = switch(tokens[:, context_length].view(-1), prev, started)
            probs[:, context_length] = switch(probs[:, context_length].view(-1), select_logits, started)

            context_length += 1
            counter += 1

            done_token = (prev == eos_id).byte() | (prev == cus_eos_id).byte()
            just_finished = (done_token & ~is_done).to(torch.bool)
            lengths[just_finished.view(-1)] = context_length
            was_done = is_done
            is_done = is_done | done_token
            done = torch.all(is_done)

            yield tokens, lengths, probs
            if done:
                break

def format_zh_bpe_result(tokens):
    new_tokens = []

    for id, token in enumerate(tokens):
        if len(token) == 1 and (_is_chinese_char(token) or _is_punctuation(token)):
            new_tokens.append(token)
        elif token == "[SEP]":
            new_tokens.append(" " + token +" ")
        elif token.startswith('##'):
            new_tokens.append(''.join(token[2:]))
        else:
            new_tokens.append(' ' + token)

    return new_tokens

def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

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
    if args.model_parallel_size > 1:
        multiple = args.make_vocab_size_divisible_by * \
                    mpu.get_model_parallel_world_size()
    else:
        multiple = args.make_vocab_size_divisible_by
    if multiple != 0:
        while (after % multiple) != 0:
            after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer

def main():
    global MODEL_INSTANCE, TOKENIZER_INSTANCE, ARGS_INSTANCE, MODEL_OUT_DTYPE

    """Main training program."""

    print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()
    ARGS_INSTANCE = args

    # Pytorch distributed.
    if args.world_size > 1:
        initialize_distributed(args)
    else:
        print('torch distribute not init')

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)
    TOKENIZER_INSTANCE = tokenizer

    # Model, optimizer, and learning rate.
    model = setup_model(args)
    MODEL_INSTANCE = model

    param = next(model.parameters())
    MODEL_OUT_DTYPE = param.dtype

    model.eval()

    # setting default batch size to 1
    # args.batch_size = 1

    args.device = torch.cuda.current_device()

    # generate samples
    args.batch_size = 1
    
    if args.num_samples == 0:
        if args.sample_input_file:
            generate_chat_samples_input_from_file(model, tokenizer, args)
        else:
            generate_chat_samples_interactive(model, tokenizer, args)
    else:
        if args.world_size != 1:
            raise Exception('Only support world size == 1')
        app.run(host='0.0.0.0', port=args.server_port)


if __name__ == "__main__":
    main()



