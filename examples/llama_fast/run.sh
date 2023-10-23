#!/bin/bash

MODEL_DIR=/data
CACHE_DIR=/data/cache
OUTPUT_DIR=/data/tensorrt

mpirun -n 4 --allow-run-as-root \
  python3 example.py --max_output_len=50 \
                --tokenizer_dir $MODEL_DIR \
                --cache_dir $CACHE_DIR \
                --engine_dir=$OUTPUT_DIR