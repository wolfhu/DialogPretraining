#!/bin/bash

CHECKPOINT_PATH=~/xiaoicechatexp/yuwu/Model_130G/

MPSIZE=1
NLAYERS=24
NHIDDEN=1024
NATT=16

# as 90% 130G training data length larger than 960, we set this value to 960
# you can change this value according to yourself training data
MAXSEQLEN=960

#SAMPLING ARGS
TEMP=1
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=50
TOPP=0.95

#flask server confg
SERVER_PORT=9999

# model pparallel config
GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=7777
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       generate_samples_gpt2_interactive.py \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --max-position-embeddings 1024 \
       --batch-size 1 \
       --tokenizer-type GPT2WordPieceTokenizer \
       --tokenizer-model-type gpt2-large-chinese-refine \
       --fp16 \
       --cache-dir cache \
       --seq-length $MAXSEQLEN \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --num-samples 1 \
       --top_p $TOPP \
       --server-port $SERVER_PORT \
       --no-load-rng

       # --sample-input-file ~/xiaoicechatexp/kaizh/test/rewrite_query_set.txt \
       # --sample-output-file ~/xiaoicechatexp/kaizh/test/rewrite_query_set.novelchat.result.txt \
       # --recompute
