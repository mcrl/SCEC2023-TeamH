# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import torch
import time
import json
import argparse
import os

from pathlib import Path

from model import ModelArgs, Transformer
from tokenizer import Tokenizer
from generation import LLaMA

def load(
    ckpt_dir: str,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    print("Loading")
    checkpoint = torch.load(os.path.join(ckpt_dir, '30B_cpu.pth'), map_location="cpu")
    with open(os.path.join(ckpt_dir, 'params.json'), "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=os.path.join(ckpt_dir, 'tokenizer.model'))
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.FloatTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def main(
    ckpt_dir: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    torch.manual_seed(1)

    generator = load(
        ckpt_dir, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
    ]
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt_dir", type=str, default="local_disk/llama_preprocessed")
  args = parser.parse_args()

  main(ckpt_dir=args.ckpt_dir)