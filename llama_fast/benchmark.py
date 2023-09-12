# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import numpy as np

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from model import ModelArgs, TransformerBlocks, PreTransformer, PostTransformer
from tokenizer import Tokenizer
from generation import LLaMA

from datasets import load_dataset
import re
import math
import logging
import torch.distributed as dist

logger = logging.getLogger('llama_fast')
sh = logging.StreamHandler()
formatter = logging.Formatter('[%(process)d][%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d:%(funcName)s] %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.INFO)

def process_example(example):
  def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text
  ctx = example['ctx_a'] + " " + example['ctx_b'].capitalize()
  out_example = {
    "query": preprocess(example["activity_label"] + ": " + ctx),
    "choices": [preprocess(ending) for ending in example["endings"]],
    "gold": int(example["label"]),
  }
  return out_example

def encode_input(tokenizer, example):
  def encode_pair(context, continuation):
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    bos = False
    whole_enc = tokenizer.encode(context + continuation, bos=bos, eos=False)
    context_enc = tokenizer.encode(context, bos=bos, eos=False)
    context_enc_len = len(context_enc)
    continuation_enc = whole_enc[context_enc_len:]
    return context_enc, continuation_enc
  reqs = [(example['query'], ' {}'.format(choice)) for choice in example['choices']]
  new_reqs = []
  for ctx, cont in reqs:
    ctx_enc, cont_enc = encode_pair(ctx, cont)
    new_reqs.append((ctx, cont, ctx_enc, cont_enc))
  return new_reqs

def encoded_input_to_tensor(eis):
  full_encs = [ei[2] + ei[3] for ei in eis]
  bsz = len(full_encs)
  max_len = max([len(enc) for enc in full_encs])
  tokens = torch.full((bsz, max_len), 0).cuda().long() # pad with 0, as it does not matter
  for i in range(bsz):
    tokens[i, :len(full_encs[i])] = torch.tensor(full_encs[i]).long()
  return tokens

def grade_output(logprobs, eis, pe):
  bsz = len(eis)
  ss = []
  logproblist = [[] for _ in range(bsz)]
  for i in range(bsz):
    ctx = eis[i][2]
    cont = eis[i][3]
    s = 0
    for j in range(len(cont)):
      logprob = logprobs[i, len(ctx) + j - 1, cont[j]].item()
      s += logprob
      logproblist[i].append(logprob)
    ss.append(s)

  gold = pe["gold"]
  acc = 1.0 if np.argmax(ss) == gold else 0.0
  completion_len = np.array([float(len(i)) for i in pe["choices"]])
  acc_norm = 1.0 if np.argmax(ss / completion_len) == gold else 0.0

  num_choices = 4
  for j in range(num_choices):
    encoded_input = eis[j]
    print(f'ctx="{encoded_input[0]}" cont="{encoded_input[1]}" ctx_enc={encoded_input[2]} cont_enc={encoded_input[3]} logproblist={logproblist[j]}')
  print(f'results: {ss}, normed_results: {ss / completion_len}, gold: {gold}')

  return acc, acc_norm

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def main(
  tokenizer_path: str,
  ckpt_dir: str,
  cache_dir: str,
  max_seq_len: int = 512,
  max_batch_size: int = 32,
):
  local_rank, world_size = setup_model_parallel()
  #if local_rank > 0:
  #    sys.stdout = open(os.devnull, "w")
  #    sys.stderr = open(os.devnull, "w")

  start_time = time.time()
  dataset = load_dataset("hellaswag", cache_dir=cache_dir, split='validation')
  checkpoint = torch.load(os.path.join(ckpt_dir, f'30B_cpu_{local_rank}.pth'), map_location="cpu")
  with open(os.path.join(ckpt_dir, 'params.json'), "r") as f:
    params = json.loads(f.read())
  model_args: ModelArgs = ModelArgs(
    max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
  )
  tokenizer = Tokenizer(model_path=tokenizer_path)
  model_args.vocab_size = tokenizer.n_words
  print(f"Loaded dataset, ckpt, tokenizer in {time.time() - start_time:.2f} seconds")

  start_time = time.time()
  torch.set_default_tensor_type(torch.cuda.HalfTensor)
  if local_rank == 0:
    pretb = PreTransformer(model_args)
    tb = TransformerBlocks(model_args, 0, 15)
    pretb.custom_load(checkpoint)
    tb.custom_load(checkpoint)
  elif local_rank == 1:
    tb = TransformerBlocks(model_args, 15, 15)
    tb.custom_load(checkpoint)
  elif local_rank == 2:
    tb = TransformerBlocks(model_args, 30, 15)
    tb.custom_load(checkpoint)
  elif local_rank == 3:
    tb = TransformerBlocks(model_args, 45, 15)
    posttb = PostTransformer(model_args)
    tb.custom_load(checkpoint)
    posttb.custom_load(checkpoint)
  torch.set_default_tensor_type(torch.FloatTensor)
  print(f"Loaded model in gpu in {time.time() - start_time:.2f} seconds")

  # max S = 170
  B = 1
  S = 170
  H = model_args.dim
  num_iter = 5
  for B in [1, 2, 4, 8, 16, 32, 64, 128]:
    for S in [170]:
    #for S in [1, 2, 4, 8, 16, 32, 64, 128, 170]:
      avg = 0
      for iter in range(num_iter):
        h = torch.empty((B, S, H)).cuda().half()
        torch.cuda.synchronize()
        st = time.time()
        h = tb(h, 0)
        torch.cuda.synchronize()
        et = time.time()
        elapsed = et - st
        avg += elapsed
      avg /= num_iter
      print(f'B={B} S={S} avg={avg}')


  #for param in tb.parameters():
  #  logger.info(f'Rank {local_rank} {type(param)} {param.size()} {param.device} {param.dtype}')


if __name__ == "__main__":
    fire.Fire(main)
