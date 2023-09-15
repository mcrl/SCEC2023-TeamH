TARGET_FOLDER=../local_disk/llama_pretrained
MP=4
MODEL_SIZE=30B
torchrun --nproc_per_node $MP example.py --ckpt_dir $TARGET_FOLDER/$MODEL_SIZE --tokenizer_path $TARGET_FOLDER/tokenizer.model
