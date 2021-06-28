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

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 40 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters  515803\
       --save checkpoints/news \
       --load checkpoints/news \
       --resume-dataloader \
       --train-data /home/t-yuniu/xiaoice/yuniu/dataset/processed/monolingual/news/news.txt.gpt2.json \
       --text-key text \
       --lazy-loader \
       --loose-json \
       --reset-position-ids \
       --reset-attention-mask \
       --eod-mask-loss \
       --tokenizer-type GPT2WordPieceTokenizer \
       --tokenizer-model-type gpt2-large-chinese \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16


set +x
