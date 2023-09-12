TOKENIZER_PATH=../local_disk/llama_preprocessed/tokenizer.model
CKPT_DIR=../local_disk/llama_preprocessed
CACHE_DIR=../local_disk/datasets
NPROCS=1

torchrun --nproc_per_node $NPROCS debug.py --tokenizer_path $TOKENIZER_PATH --ckpt_dir $CKPT_DIR --cache_dir $CACHE_DIR
