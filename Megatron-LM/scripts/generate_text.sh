#!/bin/bash

CHECKPOINT_PATH=/home/t-yuniu/xiaoice/yuniu/testing/Dialog_Pretraining/Megatron-LM/checkpoints/poem_1k_repeat_1w/
MPSIZE=1
NLAYERS=24
NHIDDEN=1024
NATT=16
MAXSEQLEN=50
BATCH_SIZE=240

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=1
TOPP=0
# TODO 37行设置了greedy search用于测试logits，测试结束后revert代码
python3 generate_samples.py \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --max-position-embeddings 1024 \
       --batch-size $BATCH_SIZE \
       --tokenizer-type GPT2WordPieceTokenizer \
       --tokenizer-model-type gpt2-large-chinese-refine \
       --fp16 \
       --cache-dir cache \
       --seq-length $MAXSEQLEN \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --sample-input-file /home/t-yuniu/xiaoice/yuniu/dataset/poem/train.1k.test.2 \
       --sample-output-file /home/t-yuniu/xiaoice/yuniu/dataset/poem/train.1k.test.2.infer \
       --num-samples 0 \
       --top_p $TOPP \
       --no-load-rng \
       --greedy \
       --recompute
