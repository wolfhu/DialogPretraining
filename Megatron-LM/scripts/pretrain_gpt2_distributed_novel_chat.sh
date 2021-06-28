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

# TRAINING PARAMS
TRAINING_SET_SIZE=893000
BATCH_SIZE=40
EPOCH_NUM=5
TRAIN_ITERS=$((TRAINING_SET_SIZE*EPOCH_NUM/BATCH_SIZE/GPUS_PER_NODE/NNODES))
DATA_PATH=/home/t-yuniu/xiaoice/yuniu/dataset/zhfiction/chitchat/chat.formatted_sep.94w
TRAINING_SET_SUFFIX=train.json
VALID_SET_SUFFIX=val.json
TEST_SET_SUFFIX=test.json
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       --finetune \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size $BATCH_SIZE \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters $TRAIN_ITERS \
       --save checkpoints/gpt2_345m_novel_chat_niu_20200512 \
       --load checkpoints/gpt2_345m_novel_chat_kaizh \
       --resume-dataloader \
       --train-data $DATA_PATH.$TRAINING_SET_SUFFIX \
       --valid-data $DATA_PATH.$VALID_SET_SUFFIX \
       --test-data $DATA_PATH.$TEST_SET_SUFFIX \
       --text-key text \
       --lazy-loader \
       --loose-json \
       --reset-position-ids \
       --reset-attention-mask \
       --eod-mask-loss \
       --tokenizer-type GPT2WordPieceTokenizer \
       --tokenizer-model-type gpt2-large-chinese \
       --cache-dir cache \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16


set +x
