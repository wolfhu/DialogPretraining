#!/bin/bash

# candiate Philly Docker
# 1. 
#      Docker Registry Name : phillyregistry.azurecr.io 
#      Docker Repo Name     : philly/jobs/custom/pytorch-tts v1.4-py36-cuda10
#      Tag                  : v1.4-py36-cuda10

nvidia-smi
sudo env PATH=$PATH python3 -m pip install boto3 tqdm flask jieba pandas nltk sentencepiece regex zhconv
# Check Environment
echo 'pytorch: '
python3 -m pip list | grep torch
echo 'CUDA: '
cat /usr/local/cuda/version.txt
echo 'CUDNN: '
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

# Apex
#sudo env PATH=$PATH python3 -m pip uninstall -y apex
#cd /blob/apex
# git clone https://github.com/NVIDIA/apex.git
# cd apex
#git checkout f3a960f
#sudo env PATH=$PATH python3 setup.py install --cpp_ext --cuda_ext

# into code dir
cd /blob/Megatron-LM || exit 1
NPROC_PER_NODE=4
# Change for multinode config
# MASTER_ADDR=localhost
MASTER_ADDR=$(head -n 1 $HOME/mpi-hosts)
MASTER_PORT=7777
# NNODES=1
# NODE_RANK=0
NNODES=$OMPI_COMM_WORLD_SIZE
NODE_RANK=$OMPI_COMM_WORLD_RANK
unset OMPI_COMM_WORLD_LOCAL_RANK
unset OMPI_COMM_WORLD_LOCAL_SIZE

# DEBUG
echo 'NPROC_PER_NODE: '$NPROC_PER_NODE
echo 'MASTER_ADDR: '$MASTER_ADDR
echo 'MASTER_PORT: '$MASTER_PORT
echo 'NNODES: '$NNODES
echo 'NODE_RANK: '$NODE_RANK

# OMPI PARAMS
echo 'OMPI_COMM_WORLD_SIZE: '$OMPI_COMM_WORLD_SIZE
echo 'OMPI_COMM_WORLD_RANK: '$OMPI_COMM_WORLD_RANK
echo 'OMPI_COMM_WORLD_LOCAL_RANK: '$OMPI_COMM_WORLD_LOCAL_RANK
echo 'OMPI_COMM_WORLD_LOCAL_SIZE: '$OMPI_COMM_WORLD_LOCAL_SIZE
echo 'SLURM_JOB_NUM_NODES: '$SLURM_JOB_NUM_NODES
echo 'SLURM_NODEID: '$SLURM_NODEID

DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# DEBUG
echo 'DISTRIBUTED_ARGS: '$DISTRIBUTED_ARGS

## CHANGE FOLLOWING PARAMS
#LOCAL_DATASET_DIR=/home/t-yuniu/xiaoice/yuniu/dataset/binary  # For Local
#LOCAL_DATASET_DIR=/blob/data  # For Philly
#LOCAL_DATASET_DIR=/home/shujliu/yuniu/dataset/processed  # For DXG2
CORPUS_PATH=/blob/data/test/chat.formatted_sep.val.json
#LOAD_MODEL=gpt_special_cls_corpus_16g
SAVE_MODEL_PATH=gpt2_345m_test
CORPUS_SIZE=100000
EPOCH_NUM=30
BATCH_SIZE=20
TRAIN_ITER_NUM=$((CORPUS_SIZE*EPOCH_NUM/BATCH_SIZE/NPROC_PER_NODE/NNODES))
# CHANGE PARAMS UP HERE

# DEBUG
echo 'CORPUS_PATH: '$CORPUS_PATH
echo 'SAVE_MODEL_PATH: '$SAVE_MODEL_PATH
echo 'CORPUS_SIZE: '$CORPUS_SIZE
echo 'EPOCH_NUM: '$EPOCH_NUM
echo 'BATCH_SIZE: '$BATCH_SIZE
echo 'TRAIN_ITER_NUM: '$TRAIN_ITER_NUM

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size $BATCH_SIZE \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters $TRAIN_ITER_NUM \
       --save $SAVE_MODEL_PATH \
       --load $SAVE_MODEL_PATH \
       --resume-dataloader \
       --train-data $CORPUS_PATH \
       --text-key text \
       --lazy-loader \
       --loose-json \
       --reset-position-ids \
       --reset-attention-mask \
       --eod-mask-loss \
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
