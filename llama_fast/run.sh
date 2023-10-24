#!/bin/bash

TOKENIZER_PATH=/data/tokenizer.model
CKPT_DIR=/data/
CACHE_DIR=/data/cache
NPROCS=4

cd "$(dirname "$0")"
# prevent dataset download from online being included in time measurement
python ../tools/download_dataset.py --cache_dir $CACHE_DIR
# Cleanup shared memory and semaphore, if the previous run was not properly shut down
rm /dev/shm/teamh* /dev/shm/sem.teamh*

start=$(date +%s.%N)

#nsys sessions list
#nsys profile --cuda-memory-usage true -o /data/test --force-overwrite true --trace-fork-before-exec=true \
python example.py --nproc_per_node $NPROCS --tokenizer_path $TOKENIZER_PATH --ckpt_dir $CKPT_DIR --cache_dir $CACHE_DIR

finish=$(date +%s.%N)
time=$( echo "$finish - $start" | bc -l )

echo '[Report from shell]'
echo 'time:' $time