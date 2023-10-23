
#!/bin/bash

MODEL_DIR=/data
OUTPUT_DIR=/data

# use_rms norm plugin?

python build.py --model_dir $MODEL_DIR \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir $OUTPUT_DIR \
                --gpus_per_node 4 \
                --world_size 4 \
                --pp_size 4 \
                --parallel_build \