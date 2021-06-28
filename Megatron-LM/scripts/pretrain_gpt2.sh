#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 12 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 320000 \
       --save checkpoints/gpt2_345m \
       --load checkpoints/gpt2_345m \
       --resume-dataloader \
       --train-data /home/t-yuniu/xiaoice/yuniu/dataset/processed/monolingual/zhaichaowang/all.v1.txt.sample.10000.json \
       --text-key text \
       --lazy-loader \
       --loose-json \
       --tokenizer-type GPT2WordPieceTokenizer \
       --tokenizer-model-type gpt2-large-chinese-refine \
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
