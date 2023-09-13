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

import schedule

NUM_CHOICES = 4
#DATA_LIMIT = 22222
#CTX_GRP_LIMIT = 22222
DATA_LIMIT = 1000
CTX_GRP_LIMIT = 2
SCHED_THR = 1024
DEBUG_SCHEDULE = True
DEBUG_ANSWER = False

logger = logging.getLogger('llama_fast')
sh = logging.StreamHandler()
formatter = logging.Formatter('[%(process)d][%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d:%(funcName)s] %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.INFO)

def make_input_cpu_tensor_from_docs(docs, phase, cached_len):
  if phase == 0:
    encs = [doc['ctx'] for doc in docs]
    S = min([len(enc) for enc in encs])
  if phase == 1:
    encs = [doc['ctx'] + doc['cont'][:-1] for doc in docs]
    S = max([len(enc) for enc in encs]) - cached_len
  bsz = len(encs)
  tokens = torch.full((bsz, S), 0).long() # pad with 0, as it does not matter
  for i in range(bsz):
    if phase == 0:
      tokens[i, :] = torch.tensor(encs[i][: S]).long()
    if phase == 1:
      tokens[i, :len(encs[i]) - cached_len] = torch.tensor(encs[i][cached_len:]).long()
  return tokens

def encoded_input_to_tensor(eis):
  full_encs = [ei[2] + ei[3] for ei in eis]
  bsz = len(full_encs)
  max_len = max([len(enc) for enc in full_encs])
  tokens = torch.full((bsz, max_len), 0).cuda().long() # pad with 0, as it does not matter
  for i in range(bsz):
    tokens[i, :len(full_encs[i])] = torch.tensor(full_encs[i]).long()
  return tokens

def encoded_ctx_to_tensor(eis):
  ctxs = [ei[0] for ei in eis]
  bsz = len(ctxs)
  min_len = min([len(ctx) for ctx in ctxs])
  tokens = torch.full((bsz, min_len), 0).long() # pad with 0, as it does not matter
  for i in range(bsz):
    tokens[i, :min_len] = torch.tensor(ctxs[i][:min_len]).long()
  return tokens

def encoded_ctx_cont_to_tensor(eis, cached_len):
  full_encs = [ei[0] + ei[1][:-1] for ei in eis] # do not use last token
  bsz = len(full_encs)
  max_len = max([len(enc) for enc in full_encs])
  tokens = torch.full((bsz, max_len - cached_len), 0).long() # pad with 0, as it does not matter
  for i in range(bsz):
    tokens[i, :len(full_encs[i]) - cached_len] = torch.tensor(full_encs[i][cached_len:]).long()
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

  if DEBUG_ANSWER:
    num_choices = 4
    for j in range(num_choices):
      encoded_input = eis[j]
      print(f'ctx="{encoded_input[0]}" cont="{encoded_input[1]}" ctx_enc={encoded_input[2]} cont_enc={encoded_input[3]} logproblist={logproblist[j]}')
    print(f'results: {ss}, normed_results: {ss / completion_len}, gold: {gold}')

  return acc, acc_norm

def record_logprobs(ctx_logprobs, logprobs, eis, pes, cached_len, cont2ctx, query_idx, logprobsum_db):
  N = len(eis)

  logproblist = [[] for _ in range(N)]
  for i in range(N):
    pe = pes[i]
    ei = eis[i]
    ctx = ei['ctx']
    cont = ei['cont']
    s = 0
    for j in range(len(cont)):
      k = len(ctx) + j - cached_len - 1
      if k == -1:
        logprob = ctx_logprobs[cont2ctx[i], cont[j]].item()
      else:
        logprob = logprobs[i, k, cont[j]].item()
      s += logprob
      logproblist[i].append(logprob)
    logprobsum_db[cont2ctx[i]][query_idx[i]] = s
  
  if DEBUG_ANSWER:
    for i in range(N):
      print(f'ctx="{pes[i]["query"]}" cont="{pes[i]["choices"][query_idx[i]]}" ctx_enc={eis[i]["ctx"]} cont_enc={eis[i]["cont"]} logproblist={logproblist[i]}')

def grade_from_db(pes, logprobsum_db):
  acc, acc_norm = 0, 0
  for i, pe in enumerate(pes):
    ss = logprobsum_db[i]
    gold = pe["gold"]
    acc += 1.0 if np.argmax(ss) == gold else 0.0
    completion_len = np.array([float(len(i)) for i in pe["choices"]])
    acc_norm += 1.0 if np.argmax(ss / completion_len) == gold else 0.0

  if DEBUG_ANSWER:
    print(f'ctx="{pe["query"]}"')
    print(f'results: {ss}, normed_results: {ss / completion_len}, gold: {gold}')

  return acc, acc_norm

def setup_model_parallel() -> Tuple[int, int]:
  global local_rank, world_size
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

  #for param in tb.parameters():
  #  logger.info(f'Rank {local_rank} {type(param)} {param.size()} {param.device} {param.dtype}')

  docs, tokenized_docs, batches = schedule.preprocess_and_schedule_dataset(dataset, tokenizer, DATA_LIMIT, SCHED_THR)

  d2h_stream = torch.cuda.Stream()
  prev_cont_args = None
  prev_ctx_args = None
  nccl_handle = None

  sum_acc = 0
  sum_acc_norm = 0
  count = 0
  start_time = time.time()

  kv_cache = {}
  output_cache = {}
  logprobsumdb_cache = {}

  for batch_idx, batch in enumerate(batches):
    docs_in_batch = (tokenized_docs[i] for i in batch.data_idx)
    tokens = make_input_cpu_tensor_from_docs(docs_in_batch, batch.phase, batch.cache_len).pin_memory().cuda(non_blocking=True)
    cont2ctx_gpu = None
    if batch.phase == 1:
      cont2ctx_gpu = torch.Tensor(batch.cache_mapping).long().pin_memory().cuda(non_blocking=True)
    B = tokens.size(0)
    S = tokens.size(1)
    H = model_args.dim
    if DEBUG_SCHEDULE:
      logger.info(f'Rank {local_rank} ctx group id={batch_idx} size={(B, S, H)}')

    # load cache
    cache_k_list, cache_v_list = None, None
    if batch.phase == 1:
      cache_k_list, cache_v_list = kv_cache[batch.cache_dep]
      if local_rank == world_size - 1:
        ctx_logprobs = output_cache[batch.cache_dep]
        logprobsum_db = logprobsumdb_cache[batch.cache_dep]

    # run prolog
    if local_rank == 0:
      h = pretb.forward(tokens)
    else:
      h = torch.empty((B, S, H), dtype=torch.float16, device='cuda')
      dist.recv(h, local_rank - 1)

    # run transformer
    h, new_cache_k_list, new_cache_v_list = tb.forward(
      h, batch.cache_len, phase = batch.phase,
      cache_k_list = cache_k_list, cache_v_list = cache_v_list, cont2ctx = cont2ctx_gpu)

    # run epilog
    if local_rank < world_size - 1:
      dist.send(h, local_rank + 1)
    else:
      if batch.phase == 0:
        ctx_h = h[:, -1, :] # [B, V]
        ctx_logits = posttb.forward(ctx_h)
        ctx_logprobs = torch.log_softmax(ctx_logits, dim=-1)
      if batch.phase == 1:
        logits = posttb.forward(h)
        logprobs = torch.log_softmax(logits, dim=-1)
        pes = [docs[i // NUM_CHOICES] for i in batch.data_idx]
        eis = [tokenized_docs[i] for i in batch.data_idx]
        query_idx = [i % NUM_CHOICES for i in batch.data_idx]
        record_logprobs(ctx_logprobs.cpu(), logprobs.cpu(), eis, pes, batch.cache_len, batch.cache_mapping, query_idx, logprobsum_db)
        if batch_idx + 1 == len(batches) or batches[batch_idx + 1].phase == 0:
          pes = [docs[i // NUM_CHOICES] for i in batches[batch.cache_dep].data_idx]
          acc, acc_norm = grade_from_db(pes, logprobsum_db)
          sum_acc += acc
          sum_acc_norm += acc_norm
          count += len(pes)
          print(f'acc: {acc}, acc_norm: {acc_norm}, avg_acc: {sum_acc / count}, avg_acc_norm: {sum_acc_norm / count}, count: {count}')

          elapsed = time.time() - start_time
          print(f'elapsed={elapsed}, throughput(example/s)={count / elapsed}')

          # TODO
          if count == 133:
            break

    # update cache
    if batch.phase == 0:
      # TODO currently only keep one element in cache
      kv_cache.clear()
      kv_cache[batch_idx] = (new_cache_k_list, new_cache_v_list)
      if local_rank == world_size - 1:
        output_cache.clear()
        logprobsumdb_cache.clear()
        output_cache[batch_idx] = ctx_logprobs
        logprobsumdb_cache[batch_idx] = [[0 for _ in range(NUM_CHOICES)] for _ in range(len(batch.data_idx))]

if __name__ == "__main__":
  if True:
    from torch.profiler import profile, record_function, ProfilerActivity
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_flops=True) as prof:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
      fire.Fire(main)
    prof.export_chrome_trace(f'trace_{local_rank}.json')
  else:
    fire.Fire(main)
