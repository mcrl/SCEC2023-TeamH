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

NC = 4
DATA_LIMIT = 22222
CTX_GRP_LIMIT = 22222
DEBUG_SCHEDULE = True

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
    new_reqs.append((ctx_enc, cont_enc))
  return new_reqs

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
    ctx = ei[0]
    cont = ei[1]
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
  
  #for i in range(N):
  #  print(f'ctx="{pes[i]["query"]}" cont="{pes[i]["choices"][query_idx[i]]}" ctx_enc={eis[i][0]} cont_enc={eis[i][1]} logproblist={logproblist[i]}')

def grade_from_db(pes, logprobsum_db):
  acc, acc_norm = 0, 0
  for i, pe in enumerate(pes):
    ss = logprobsum_db[i]
    gold = pe["gold"]
    acc += 1.0 if np.argmax(ss) == gold else 0.0
    completion_len = np.array([float(len(i)) for i in pe["choices"]])
    acc_norm += 1.0 if np.argmax(ss / completion_len) == gold else 0.0

    #print(f'ctx="{pe["query"]}"')
    #print(f'results: {ss}, normed_results: {ss / completion_len}, gold: {gold}')

  return acc, acc_norm

def setup_model_parallel() -> Tuple[int, int]:
  global local_rank
  local_rank = int(os.environ.get("LOCAL_RANK", -1))
  world_size = int(os.environ.get("WORLD_SIZE", -1))

  torch.distributed.init_process_group("nccl")
  initialize_model_parallel(world_size)
  torch.cuda.set_device(local_rank)

  # seed must be the same in all processes
  torch.manual_seed(1)
  return local_rank, world_size

# length is minimum of each block
def schedule_min(lengths, thr):
  N = len(lengths)
  idx = [i for i in range(N)]
  idx.sort(key=lambda x: lengths[x])
  # init
  D = []
  E = []
  total_area = 0
  for i in range(N):
    total_area += lengths[idx[i]]
    rect_area = lengths[idx[0]] * (i + 1)
    penalty = total_area - rect_area
    if rect_area >= thr:
      D.append(penalty)
      E.append(-1)
    else:
      D.append(2 ** 30)
      E.append(-1)
  # DP
  for i in range(N):
    total_area = 0
    for j in range(i - 1, -1, -1):
      total_area += lengths[idx[j + 1]]
      rect_area = lengths[idx[j + 1]] * (i - j)
      penalty = total_area - rect_area
      if rect_area >= thr and D[i] > D[j] + penalty:
        D[i] = D[j] + penalty
        E[i] = j
  blocks = []
  i = N - 1
  while i >= 0:
    blocks.append(i - E[i])
    i = E[i]
  blocks.reverse()

  #logging
  #print(f'total penalty: {D[N - 1]}')
  #s = 0
  #for i, block in enumerate(blocks):
  #  s += block
  #  cur_s = lengths[idx[s - block]]
  #  print(f'block {i}: {block} x {cur_s} = {block * cur_s}')
  #print(blocks)
  #print(sum(blocks))

  return idx, blocks

# length is maximum of each block
def schedule_max(lengths, thr):
  N = len(lengths)
  idx = [i for i in range(N)]
  idx.sort(key=lambda x: lengths[x])
  # init
  D = []
  E = []
  total_area = 0
  for i in range(N):
    total_area += lengths[idx[i]]
    rect_area = lengths[idx[i]] * (i + 1)
    penalty = rect_area - total_area
    if rect_area >= thr:
      D.append(penalty)
      E.append(-1)
    else:
      D.append(2 ** 30)
      E.append(-1)
  # DP
  for i in range(N):
    total_area = 0
    for j in range(i - 1, -1, -1):
      total_area += lengths[idx[j + 1]]
      rect_area = lengths[idx[i]] * (i - j)
      penalty = rect_area - total_area
      if rect_area >= thr and D[i] > D[j] + penalty:
        D[i] = D[j] + penalty
        E[i] = j
  blocks = []
  i = N - 1
  while i >= 0:
    blocks.append(i - E[i])
    i = E[i]
  blocks.reverse()

  #logging
  #print(f'total penalty: {D[N - 1]}')
  #s = 0
  #for i, block in enumerate(blocks):
  #  s += block
  #  cur_s = lengths[idx[s - 1]]
  #  print(f'block {i}: {block} x {cur_s} = {block * cur_s}')
  #print(blocks)
  #print(sum(blocks))

  return idx, blocks


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

  # encode the whole dataset
  whole_pe = []
  whole_ei = []
  for i, data in enumerate(dataset):
    # TODO
    if i == DATA_LIMIT:
      break
    if i % 100 == 0:
      print(f'Processing {i}...')
    pe = process_example(data)
    whole_pe.append(pe)
    whole_ei.extend(encode_input(tokenizer, pe))

  # schedule context blocks and cont blocks
  NC = 4
  THR = 1024
  ctx_lengths = [len(whole_ei[i][0]) for i in range(0, len(whole_ei), NC)]
  ctx_idx, ctx_blocks = schedule_min(ctx_lengths, THR)

  schedule_info = []
  s = 0
  for i, ctx_block in enumerate(ctx_blocks):
    s += ctx_block
    ctx_min_len = len(whole_ei[ctx_idx[s - ctx_block] * NC][0])
    cur_ctx_grp = []
    cur_grp = []
    for j in range(s - ctx_block, s):
      cur_ctx_grp.append(ctx_idx[j] * NC)
      cur_grp.extend([ctx_idx[j] * NC + k for k in range(NC)])
    cont_lengths = [len(whole_ei[j][0]) + len(whole_ei[j][1]) - ctx_min_len for j in cur_grp]
    cont_idx, cont_blocks = schedule_max(cont_lengths, THR)
    # TODO
    schedule_info.append((cur_ctx_grp, cur_grp, cont_idx, cont_blocks))

  d2h_stream = torch.cuda.Stream()
  prev_cont_args = None
  prev_ctx_args = None
  nccl_handle = None

  sum_acc = 0
  sum_acc_norm = 0
  count = 0
  start_time = time.time()
  for ctx_grp_idx, (cur_ctx_grp, cur_grp, cont_idx, cont_blocks) in enumerate(schedule_info):
    if ctx_grp_idx == CTX_GRP_LIMIT:
      break

    torch.cuda.nvtx.range_push('ctx group input gen')

    #for i in range(len(cur_ctx_grp)):
    #  if local_rank == 0:
    #    print(f'ctx group {i} : {whole_pe[cur_ctx_grp[i] // NC]}')
    eis = [whole_ei[i] for i in cur_ctx_grp]
    tokens = encoded_ctx_to_tensor(eis)
    tokens = tokens.pin_memory()
    tokens = tokens.cuda(non_blocking=True)
    B = tokens.size(0)
    S = tokens.size(1)
    H = model_args.dim
    if DEBUG_SCHEDULE:
      logger.info(f'Rank {local_rank} ctx group id={ctx_grp_idx} size={(B, S, H)} min_len={len(eis[0][0])}')

    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push('ctx group forward')

    if local_rank == 0:
      h = pretb.forward(tokens)
      h, cache_k_list, cache_v_list = tb.forward(h, 0, phase = 0)
      dist.isend(h, local_rank + 1)
    elif local_rank == 1:
      h = torch.empty((B, S, H), dtype=torch.float16, device='cuda')
      dist.recv(h, local_rank - 1)
      h, cache_k_list, cache_v_list = tb.forward(h, 0, phase = 0)
      dist.isend(h, local_rank + 1)
    elif local_rank == 2:
      h = torch.empty((B, S, H), dtype=torch.float16, device='cuda')
      dist.recv(h, local_rank - 1)
      h, cache_k_list, cache_v_list = tb.forward(h, 0, phase = 0)
      dist.isend(h, local_rank + 1)
    elif local_rank == 3:
      h = torch.empty((B, S, H), dtype=torch.float16, device='cuda')
      dist.recv(h, local_rank - 1)
      h, cache_k_list, cache_v_list = tb.forward(h, 0, phase = 0)
      ctx_h = h[:, -1, :] # [B, V]
      ctx_logits = posttb.forward(ctx_h)
      ctx_logprobs = torch.log_softmax(ctx_logits, dim=-1)

    torch.cuda.nvtx.range_pop()
    
    ei_idx_end = 0
    cached_len = S
    # TODO
    #cached_len = 0
    logprobsum_db = [[0 for _ in range(NC)] for _ in range(len(cur_ctx_grp))]
    for cont_block_idx, cont_block in enumerate(cont_blocks):
      torch.cuda.nvtx.range_push('cont group input gen')

      ei_idx_end += cont_block
      eis = []
      cont2ctx = []
      # cur_ctx_grp point to document idx [0, len(dataset) * 4) and is multiple of 4
      # cur_grp point to choice idx [0, len(dataset) * 4)
      # len(cur_ctx_grp) * 4 == len(cur_grp)
      # cont_idx point to index to current ctx grp [0, len(cur_grp))
      for ei_idx in range(ei_idx_end - cont_block, ei_idx_end):
        eis.append(whole_ei[cur_grp[cont_idx[ei_idx]]])
        cont2ctx.append(cont_idx[ei_idx] // NC)
        #if local_rank == 0:
        #  print(f'cont {ei_idx} : {whole_ei[cur_grp[cont_idx[ei_idx]]]}')
      #if local_rank == 0:
      #  for cont, ctx in enumerate(cont2ctx):
      #    print(f'cont {cont} -> ctx {ctx}')
      tokens = encoded_ctx_cont_to_tensor(eis, cached_len)
      tokens = tokens.pin_memory()
      tokens = tokens.cuda(non_blocking=True)
      #if local_rank == 0:
      #  print(tokens)
      cont2ctx_gpu = torch.Tensor(cont2ctx).long()
      cont2ctx_gpu = cont2ctx_gpu.pin_memory()
      cont2ctx_gpu = cont2ctx_gpu.cuda(non_blocking=True)
      B = tokens.size(0)
      S = tokens.size(1)
      H = model_args.dim

      torch.cuda.nvtx.range_pop()

      torch.cuda.nvtx.range_push('cont group forward')

      if local_rank == 0:
        h = pretb.forward(tokens)
        h = tb.forward(h, cached_len, phase = 1, cache_k_list = cache_k_list, cache_v_list = cache_v_list, cont2ctx = cont2ctx_gpu)

        #h, _, _ = tb.forward(h, cached_len, phase = 0)

        #h_new, cache_k_list, cache_v_list = tb.forward(h[:, 0:1, :], 0, phase = 0)
        #tmp = torch.Tensor([i for i in range(B)]).cuda().long()
        #h_new2 = tb.forward(h[:, 1:, :], 1, phase = 1, cache_k_list = cache_k_list, cache_v_list = cache_v_list, cont2ctx = tmp)
        #h = torch.cat((h_new, h_new2), dim = 1)
        #logger.info(h.shape)

        dist.isend(h, local_rank + 1)
      elif local_rank == 1:
        #if nccl_handle is not None:
        #  nccl_handle.wait()
        #  nccl_handle = None
        #  h = h_prefetch
        #else:
        #  h = torch.empty((B, S, H), dtype=torch.float16, device='cuda')
        #  nccl_handle = dist.irecv(h, local_rank - 1)
        #  nccl_handle.wait()

        #if cont_block_idx < len(cont_blocks) - 1: # not last
        #  h_prefetch = torch.empty((1, 1, 1), dtype=torch.float16, device='cuda')
        #  nccl_handle = dist.irecv(h_prefetch, local_rank - 1)
        h = torch.empty((B, S, H), dtype=torch.float16, device='cuda')
        dist.recv(h, local_rank - 1)
        h = tb.forward(h, cached_len, phase = 1, cache_k_list = cache_k_list, cache_v_list = cache_v_list, cont2ctx = cont2ctx_gpu)
        #h, _, _ = tb.forward(h, cached_len, phase = 0)
        dist.isend(h, local_rank + 1)
      elif local_rank == 2:
        h = torch.empty((B, S, H), dtype=torch.float16, device='cuda')
        dist.recv(h, local_rank - 1)
        h = tb.forward(h, cached_len, phase = 1, cache_k_list = cache_k_list, cache_v_list = cache_v_list, cont2ctx = cont2ctx_gpu)
        #h, _, _ = tb.forward(h, cached_len, phase = 0)
        dist.isend(h, local_rank + 1)
      elif local_rank == 3:
        h = torch.empty((B, S, H), dtype=torch.float16, device='cuda')
        dist.recv(h, local_rank - 1)
        h = tb.forward(h, cached_len, phase = 1, cache_k_list = cache_k_list, cache_v_list = cache_v_list, cont2ctx = cont2ctx_gpu)
        #h, _, _ = tb.forward(h, cached_len, phase = 0)
        logits = posttb.forward(h)
        logprobs = torch.log_softmax(logits, dim=-1)

        # process previous cpu job if exists MODIFY ALL COPIES
        if prev_cont_args is not None:
          d2h_stream.synchronize() # logits are ready
          record_logprobs(ctx_logprobs_cpu, logprobs_cpu, *prev_cont_args)
          prev_cont_args = None

        # process previous cpu job if exists MODIFY ALL COPIES
        if prev_ctx_args is not None:
          pes, logprobsum_db = prev_ctx_args
          acc, acc_norm = grade_from_db(*prev_ctx_args)
          prev_ctx_args = None

          sum_acc += acc
          sum_acc_norm += acc_norm
          count += len(pes)
          print(f'acc: {acc}, acc_norm: {acc_norm}, avg_acc: {sum_acc / count}, avg_acc_norm: {sum_acc_norm / count}, count: {count}')

          elapsed = time.time() - start_time
          print(f'elapsed={elapsed}, throughput(example/s)={count / elapsed}')

        # async call d2h
        with torch.cuda.stream(d2h_stream):
          torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
          logprobs_cpu = torch.empty_like(logprobs, device='cpu', pin_memory=True).copy_(logprobs, non_blocking=True)
          ctx_logprobs_cpu = torch.empty_like(ctx_logprobs, device='cpu', pin_memory=True).copy_(ctx_logprobs, non_blocking=True)

        # plan next cpu job 
        pes = [whole_pe[cur_grp[cont_idx[ei_idx]] // NC] for ei_idx in range(ei_idx_end - cont_block, ei_idx_end)]
        query_idx = [cont_idx[ei_idx] % NC for ei_idx in range(ei_idx_end - cont_block, ei_idx_end)]
        prev_cont_args = (eis, pes, cached_len, cont2ctx, query_idx, logprobsum_db)

      torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push('grading')

    # outside cont loop
    if local_rank == 3:
      pes = [whole_pe[i // NC] for i in cur_ctx_grp]
      prev_ctx_args = (pes, logprobsum_db)

    torch.cuda.nvtx.range_pop()

  # outside ctx loop

  # process previous cpu job if exists MODIFY ALL COPIES
  if prev_cont_args is not None:
    d2h_stream.synchronize() # logits are ready
    logprobs = torch.log_softmax(logits, dim=-1)
    ctx_logprobs = torch.log_softmax(ctx_logits, dim=-1)
    record_logprobs(ctx_logprobs, logprobs, *prev_cont_args)
    prev_cont_args = None

  # process previous cpu job if exists MODIFY ALL COPIES
  if prev_ctx_args is not None:
    pes, logprobsum_db = prev_ctx_args
    acc, acc_norm = grade_from_db(*prev_ctx_args)
    prev_ctx_args = None

    sum_acc += acc
    sum_acc_norm += acc_norm
    count += len(pes)
    print(f'acc: {acc}, acc_norm: {acc_norm}, avg_acc: {sum_acc / count}, avg_acc_norm: {sum_acc_norm / count}, count: {count}')

    elapsed = time.time() - start_time
    print(f'elapsed={elapsed}, throughput(example/s)={count / elapsed}')


if __name__ == "__main__":
  if False:
    from torch.profiler import profile, record_function, ProfilerActivity
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_flops=True) as prof:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
      fire.Fire(main)
    prof.export_chrome_trace(f'trace_{local_rank}.json')
  else:
    fire.Fire(main)
