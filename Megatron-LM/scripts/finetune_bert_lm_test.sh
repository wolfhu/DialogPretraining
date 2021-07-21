#!/bin/bash

RANK=0
WORLD_SIZE=1

python3 finetune_bert_lm.py \
       --num-layers 5 \
        --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 128 \
       --seq-length 128 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 128 \
       --save checkpoints/bert_lm_base_finetune \
       --load checkpoints/bert_lm_base_finetune \
       --resume-dataloader \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-base-chinese \
       --presplit-sentences \
       --cache-dir cache \
       --distributed-backend nccl \
       --lr 2e-4 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --finetune \
       --test-data finetune_test
