#!/bin/bash

MODEL_DIR=/data
OUTPUT_DIR=/data

mpirun -n 4 --allow-run-as-root \
  python3 run.py --max_output_len=50 \
                --tokenizer_dir $MODEL_DIR \
                --engine_dir=$OUTPUT_DIR