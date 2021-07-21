#!/bin/bash

RANK=0
WORLD_SIZE=1

python3.6 finetune_bert_lm.py \
       --num-layers 6 \
        --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 16 \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters 100000 \
       --save checkpoints/bert_lm_small_finetune \
       --load checkpoints/bert_lm_small \
       --resume-dataloader \
       --train-data finetune \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-base-chinese \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 2e-4 \
       --lr-decay-style linear \
       --lr-decay-iters 99000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --finetune
