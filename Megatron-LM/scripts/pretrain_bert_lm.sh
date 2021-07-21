#!/bin/bash

RANK=0
WORLD_SIZE=1

python3 pretrain_bert_lm.py \
       --num-layers 5 \
        --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 128 \
       --seq-length 128 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 128 \
       --train-iters 1000000 \
       --save checkpoints/bert_lm_base_v1 \
       --load checkpoints/bert_lm_base_v1 \
       --resume-dataloader \
       --train-data wiki \
       --lazy-loader \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-base-chinese \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 2e-4 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01
