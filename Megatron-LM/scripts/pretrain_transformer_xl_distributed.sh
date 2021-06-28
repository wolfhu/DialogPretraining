#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_transformerxl.py \
       --transoxl_n_layer 24 \
       --transoxl_d_model 1024 \
       --transoxl_n_head 8 \
       --transoxl_d_head 128 \
       --transoxl_d_inner 3072 \
       --transoxl_dropout 0.15 \
       --transoxl_dropatt 0.15 \
       --transoxl_tgt_len 16 \
       --transoxl_mem_len 16 \
       --transoxl_eval_tgt_len 16 \
       --transoxl_pre_lnorm \
       --batch-size 600 \
       --seq-length 165 \
       --train-iters 2000 \
       --save ~/kaizh/checkpoint/transfo_xl_345m_poem/ \
       --load ~/kaizh/checkpoint/transfo_xl_345m_poem/ \
       --resume-dataloader \
       --train-data ~/kaizh/data/poem/train.1w.json \
       --text-key text \
       --lazy-loader \
       --loose-json \
       --tokenizer-type GPT2WordPieceTokenizer \
       --tokenizer-model-type gpt2-large-chinese-refine \
       --cache-dir ~/xiaoicechatexp/kaizh/cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --log-interval 10 \
       --save-interval 1000 \
       --fp16


set +x
