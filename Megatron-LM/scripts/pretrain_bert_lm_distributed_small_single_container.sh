#!/bin/bash

#cd /blob/electral
#pip3 install -r docker/requirements.txt
pip3 install regex

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6699
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
#WORLD_SIZE=2

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3.6 -m torch.distributed.launch $DISTRIBUTED_ARGS \
      pretrain_bert_lm.py \
       --num-layers 6 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 128 \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save checkpoints/bert_lm_small_1c \
       --load checkpoints/bert_lm_small_1c \
       --resume-dataloader \
       --train-data wiki baidu_baike news zhaichaowang \
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
       --warmup .01 \
       --fp16 \
       --fp32-layernorm \
       --fp32-embedding \
       || exit 1
