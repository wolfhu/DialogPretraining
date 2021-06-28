#!/bin/bash

CHECKPOINT_PATH=~/xiaoicechatexp/kaizh/transfoxl_345m_poem/
MPSIZE=1
MAXSEQLEN=100
OUTSEQLEN=100

#SAMPLING ARGS
TEMP=1
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=50
TOPP=0.95

python generate_samples_transfoxl.py \
       --model-parallel-size $MPSIZE \
       --transoxl_n_layer 24 \
       --transoxl_d_model 1024 \
       --transoxl_n_head 8 \
       --transoxl_d_head 128 \
       --transoxl_d_inner 3072 \
       --transoxl_tgt_len 16 \
       --transoxl_mem_len 16 \
       --transoxl_pre_lnorm \
       --load $CHECKPOINT_PATH \
       --batch-size 1 \
       --tokenizer-type GPT2WordPieceTokenizer \
       --tokenizer-model-type gpt2-large-chinese-refine \
       --fp16 \
       --cache-dir ~/xiaoicechatexp/kaizh/cache \
       --seq-length $MAXSEQLEN \
       --out-seq-length $OUTSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP 
