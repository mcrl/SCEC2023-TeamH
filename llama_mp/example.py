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

from model import ModelArgs, Transformer
from tokenizer import Tokenizer
from generation import LLaMA

from datasets import load_dataset
import re
import math

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

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

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    dataset = load_dataset("hellaswag", cache_dir="~/SCEC2023-TeamH/local_disk/datasets", split='validation')
    print(f"Loaded in {time.time() - start_time:.2f} seconds")

    # TODO
    #for i in range(2):
    #  ctx_enc=[1528, 974, 528, 292, 280, 28744, 29901, 319, 767, 338, 16246, 373, 263, 17526, 29889, 940]
    #  if i == 0:
    #    cont_enc=[338, 773, 12244, 304, 12244, 263, 5101, 310, 2071, 275, 29889]
    #  elif i == 1:
    #    cont_enc=[338, 1, 2, 3, 4, 5, 6, 7]
    #    #cont_enc=[338, 10107, 3262, 3233, 260, 5475, 1283, 29889]
    #  full_enc = ctx_enc + cont_enc
    #  tokens = torch.full((1, max_seq_len), tokenizer.pad_id).cuda().long()
    #  tokens[0, : len(full_enc)] = torch.tensor(full_enc).long()
    #  logits = model.forward(tokens[:, : len(full_enc)], 0) # [B, S, V]
    #  prob = torch.softmax(logits, dim=-1)
    #  print(f'prob={prob[0, len(ctx_enc) - 1, cont_enc[0]]}')
    #return

    print(f'pad_id={tokenizer.pad_id}')

    num_choices = 4
    sum_acc = 0
    sum_acc_norm = 0
    count = 0
    bsz = max_batch_size
    assert bsz == 4 # single batch process
    for ex_idx in range(0, len(dataset), bsz // num_choices):
      # last batch
      if ex_idx + bsz // num_choices > len(dataset):
        bsz = (len(dataset) - ex_idx) * num_choices

      processed_examples = []
      encoded_inputs = []
      for i in range(ex_idx, ex_idx + bsz // num_choices):
        processed_example = process_example(dataset[i])
        processed_examples.append(processed_example)
        encoded_inputs.extend(encode_input(tokenizer, processed_example))
      assert len(processed_examples) == bsz // num_choices
      assert len(encoded_inputs) == bsz

      min_prompt_size = min([len(i[2]) for i in encoded_inputs])
      max_prompt_size = max([len(i[2]) for i in encoded_inputs])
      max_gen_len = max([len(i[3]) for i in encoded_inputs])
      total_len = min(max_seq_len, max_gen_len + max_prompt_size)

      tokens = torch.full((bsz, total_len), tokenizer.eos_id).cuda().long()
      for k, t in enumerate(encoded_inputs):
        ctx = t[2]
        cont = t[3]
        tokens[k, : len(ctx)] = torch.tensor(ctx).long()
        tokens[k, len(ctx) : len(ctx) + len(cont)] = torch.tensor(cont).long()
      start_pos = min_prompt_size
      prev_pos = 0
      sumlogprobs = [0.0] * bsz
      plist = [[] for _ in range(bsz)]
      logproblist = [[] for _ in range(bsz)]
      cur_pos = total_len
      #assert bsz == 4 # single batch process
      for i in range(bsz):
        ctx = encoded_inputs[i][2]
        cont = encoded_inputs[i][3]
        logits = model.forward(tokens[i:i+1, 0:len(ctx)+len(cont)], 0)
        logprobs = torch.log_softmax(logits, dim=-1) # [1, S, V]
        for cur_pos in range(start_pos, total_len):
          if cur_pos >= len(ctx) and cur_pos < len(ctx) + len(cont):
            #p = probs[i, cont[cur_pos - len(ctx)]].item()
            #logprob = math.log(p)
            logprob = logprobs[0, cur_pos - 1, cont[cur_pos - len(ctx)]].item()
            sumlogprobs[i] += logprob
            #plist[i].append(p)
            logproblist[i].append(logprob)
      #probs = torch.softmax(logits, dim=-1)

      # eval
      for i in range(bsz // num_choices):
        example = processed_examples[i]
        gold = example["gold"]
        results = sumlogprobs[i * num_choices : (i + 1) * num_choices]
        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in example["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        sum_acc += acc
        sum_acc_norm += acc_norm
        count += 1
        for j in range(num_choices):
          encoded_input = encoded_inputs[i * num_choices + j]
          print(f'ctx="{encoded_input[0]}" cont="{encoded_input[1]}" ctx_enc={encoded_input[2]} cont_enc={encoded_input[3]} plist={plist[i * num_choices + j]} logproblist={logproblist[i * num_choices + j]}')
        print(f'results: {results}, normed_results: {results / completion_len}, gold: {gold}')
        print(f'acc: {acc}, acc_norm: {acc_norm}, avg_acc: {sum_acc / count}, avg_acc_norm: {sum_acc_norm / count}, count: {count}')
         

      #results = []
      #for ctx, cont, ctx_enc, cont_enc in encoded_inputs:
      #  full_enc = ctx_enc + cont_enc
      #  full_len = len(full_enc)
      #  tokens[0, : full_len] = torch.tensor(full_enc).long()
      #  sum = 0
      #  plist = []
      #  logproblist = []
      #  prev_pos = 0
      #  for i in range(len(ctx_enc), full_len):
      #    logits = model.forward(tokens[:, : full_len], 0) # [B, S, V]
      #    prob = torch.softmax(logits, dim=-1)
      #    if i == full_len:
      #      p = prob[0, i - 1, tokenizer.eos_id].item()
      #    else:
      #      p = prob[0, i - 1, full_enc[i]].item()
      #    plist.append(p)
      #    logprob = math.log(p)
      #    logproblist.append(logprob)
      #    sum += logprob
      #  results.append(sum)
      #  print(f'ctx={ctx} cont={cont} ctx_enc={ctx_enc} cont_enc={cont_enc} plist={plist} logproblist={logproblist}')
      #gold = example["gold"]
      #acc = 1.0 if np.argmax(results) == gold else 0.0
      #completion_len = np.array([float(len(i)) for i in example["choices"]])
      #acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

      #sum_acc += acc
      #sum_acc_norm += acc_norm
      #count += 1
      #print(f'results: {results}, normed_results: {results / completion_len}, gold: {gold}')
      #print(f'acc: {acc}, acc_norm: {acc_norm}, avg_acc: {sum_acc / count}, avg_acc_norm: {sum_acc_norm / count}, count: {count}')

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

if __name__ == "__main__":
    fire.Fire(main)
