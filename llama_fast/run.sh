#!/bin/bash

TOKENIZER_PATH=/data/tokenizer.model
CKPT_DIR=/data/
CACHE_DIR=/data/cache
NPROCS=4

cd "$(dirname "$0")"

start=$(date +%s.%N)

torchrun --nproc_per_node $NPROCS example.py --tokenizer_path $TOKENIZER_PATH --ckpt_dir $CKPT_DIR --cache_dir $CACHE_DIR

finish=$(date +%s.%N)
time=$( echo "$finish - $start" | bc -l )
echo 'time:' $time