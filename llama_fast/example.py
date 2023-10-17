# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
from multiprocessing import Process, Queue
import os
import sys
import torch
import fire
import time
import json
import numpy as np

from model import ModelArgs, TransformerBlocks, PreTransformer, PostTransformer
from tokenizer import Tokenizer

from datasets import load_dataset
import logging
import torch.distributed as dist

import schedule
import cProfile
import pstats

NUM_CHOICES = 4
DATA_LIMIT = 22222
CTX_GRP_LIMIT = 22222
CTX_THR = 10000
CTX_MINIBATCH_THR = 2048
CONT_THR = 2048
DEBUG_SCHEDULE = False
DEBUG_ANSWER = False
DEBUG_PERFORMANCE = False
DEBUG_PROFILE = False
DEBUG_PYTHON_PROFILE = False
DEBUG_COMPARE_GOLD = False

if DEBUG_COMPARE_GOLD:
  gold_answer = []
  with open('gold_answer.txt') as f:
    lines = f.readlines()
    assert len(lines) == 10042
    for line in lines:
      normed_results, gold = line.strip().split(';')
      normed_results, gold = eval(normed_results), int(gold)
      gold_answer.append({'normed_results': normed_results, 'gold': gold})

logger = logging.getLogger('llama_fast')
sh = logging.StreamHandler()
formatter = logging.Formatter('[%(process)d][%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d:%(funcName)s] %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.INFO)

def make_input_cpu_tensor_from_docs(docs, batch):
  encs = [doc['ctx'] + doc['cont'] for doc in docs]
  S = batch.seq_len
  bsz = len(encs)
  tokens = torch.full((bsz, S), 0).long() # pad with 0, as it does not matter
  for i in range(bsz):
    l = min(len(encs[i]), batch.cache_len + S) - batch.cache_len
    tokens[i, :l] = torch.tensor(encs[i][batch.cache_len : batch.cache_len + l]).long()
  return tokens

def record_logprobs(ctx_logprobs, logprobs, batch, docs, tokenized_docs, logprobsum_db):
  pes = [docs[i // NUM_CHOICES] for i in batch.data_idx]
  eis = [tokenized_docs[i] for i in batch.data_idx]
  query_idx = [i % NUM_CHOICES for i in batch.data_idx]
  cached_len = batch.cache_len
  cont2ctx = batch.cache_mapping
        
  N = len(eis)

  done_idx = []
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
    logprobsum_db[batch.data_idx[i] // NUM_CHOICES][query_idx[i]] = s
    if all(x != None for x in logprobsum_db[batch.data_idx[i] // NUM_CHOICES]):
      done_idx.append(batch.data_idx[i] // NUM_CHOICES)
  
  if DEBUG_ANSWER:
    for i in range(N):
      print(f'ctx="{pes[i]["query"]}" cont="{pes[i]["choices"][query_idx[i]]}" ctx_enc={eis[i]["ctx"]} cont_enc={eis[i]["cont"]} logproblist={logproblist[i]}')
  
  return done_idx

def grade_from_db(data_idx, docs, logprobsum_db):
  pes = [docs[i] for i in data_idx]

  acc, acc_norm = 0, 0
  sq_acc, sq_acc_norm = 0, 0
  for i, pe in enumerate(pes):
    ss = logprobsum_db[data_idx[i]]
    gold = pe["gold"]
    if np.argmax(ss) == gold:
      acc += 1.0
      sq_acc += 1.0 ** 2
    completion_len = np.array([float(len(i)) for i in pe["choices"]])
    if np.argmax(ss / completion_len) == gold:
      acc_norm += 1.0
      sq_acc_norm += 1.0 ** 2
    if DEBUG_COMPARE_GOLD:
      if gold_answer[data_idx[i]]['gold'] != np.argmax(ss / completion_len):
        print(f'gold mismatch: {gold_answer[data_idx[i]]} != {ss / completion_len}')

  if DEBUG_ANSWER:
    print(f'ctx="{pe["query"]}"')
    print(f'results: {ss}, normed_results: {ss / completion_len}, gold: {gold}')

  return acc, acc_norm, sq_acc, sq_acc_norm

def run_grade(grade_queue, grade_state, d2h_stream):
  if len(grade_queue) > 0:
    d2h_stream.synchronize()
  while len(grade_queue) > 0:
    ctx_logprobs_cpu, logprobs_cpu, batch, logprobsum_db, docs, tokenized_docs = grade_queue.pop(0)
    done_idx = record_logprobs(ctx_logprobs_cpu, logprobs_cpu, batch, docs, tokenized_docs, logprobsum_db)

    if len(done_idx) > 0:
      acc, acc_norm, sq_acc, sq_acc_norm = grade_from_db(done_idx, docs, logprobsum_db)
      grade_state["sum_acc"] += acc
      grade_state["sum_sq_acc"] += sq_acc
      grade_state["sum_acc_norm"] += acc_norm
      grade_state["sum_sq_acc_norm"] += sq_acc_norm
      grade_state["count"] += len(done_idx)
      #print(f'acc: {acc}, acc_norm: {acc_norm}, sum_acc: {grade_state["sum_acc"]}, sum_acc_norm: {grade_state["sum_acc_norm"]}, avg_acc: {grade_state["sum_acc"] / grade_state["count"]}, avg_acc_norm: {grade_state["sum_acc_norm"] / grade_state["count"]}, count: {grade_state["count"]}')
      print(f'acc_norm: {acc_norm}, sum_acc_norm: {grade_state["sum_acc_norm"]}, avg_acc_norm: {grade_state["sum_acc_norm"] / grade_state["count"]}, count: {grade_state["count"]}')

      elapsed = time.time() - grade_state["start_time"]
      print(f'elapsed={elapsed}, throughput(example/s)={grade_state["count"] / elapsed}')

def print_result(grade_state):
  acc = grade_state['sum_acc'] / grade_state['count']
  acc_norm = grade_state['sum_acc_norm'] / grade_state['count']
  acc_stddev = ((grade_state['sum_sq_acc'] / grade_state['count']) - acc ** 2) ** 0.5
  acc_norm_stddev = ((grade_state['sum_sq_acc_norm'] / grade_state['count']) - acc_norm ** 2) ** 0.5
  acc_stderr = acc_stddev / (grade_state['count'] ** 0.5)
  acc_norm_stderr = acc_norm_stddev / (grade_state['count'] ** 0.5)
  print(f'|  Task   |Version| Metric |Value |   |Stderr|')
  print(f'|---------|------:|--------|-----:|---|-----:|')
  print(f'|hellaswag|      0|acc     |{acc:.4f}|±  |{acc_stderr:.4f}|')
  print(f'|         |       |acc_norm|{acc_norm:.4f}|±  |{acc_norm_stderr:.4f}|')

def setup_model_parallel() -> Tuple[int, int]:
  global local_rank, world_size
  local_rank = int(os.environ.get("LOCAL_RANK", -1))
  world_size = int(os.environ.get("WORLD_SIZE", -1))

  torch.distributed.init_process_group("nccl")
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

  # if local_rank > 0:
  #    sys.stdout = open(os.devnull, "w")
  #    sys.stderr = open(os.devnull, "w")

  # Load tokenizer
  start_time = time.time()
  tokenizer = Tokenizer(model_path=tokenizer_path)
  print(f"Loaded tokenizer in {time.time() - start_time:.2f} seconds")

  # Load and preprocess dataset with multiprocessing
  def _load_preprocess_and_schedule_dataset(q, cache_dir, tokenizer):
    start_time = time.time()
    dataset = load_dataset("hellaswag", cache_dir=cache_dir, split='validation')
    docs, tokenized_docs, batches = schedule.preprocess_and_schedule_dataset(dataset, tokenizer, DATA_LIMIT, CTX_THR, CTX_MINIBATCH_THR, CONT_THR)
    q.put((docs, tokenized_docs, batches))
    print(f"Loaded dataset and preprocessed in {time.time() - start_time:.2f} seconds")

  q = Queue()
  p = Process(target=_load_preprocess_and_schedule_dataset, args=(q, cache_dir, tokenizer))
  p.start()

  # Load model
  start_time = time.time()
  checkpoint = torch.load(os.path.join(ckpt_dir, f'30B_cpu_{local_rank}.pth'), map_location="cpu")
  with open(os.path.join(ckpt_dir, 'params.json'), "r") as f:
    params = json.loads(f.read())
  model_args: ModelArgs = ModelArgs(
    max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
  )
  model_args.vocab_size = tokenizer.n_words
  print(f"Loaded ckpt in {time.time() - start_time:.2f} seconds")

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

  # Join dataset process
  docs, tokenized_docs, batches = q.get()

  d2h_stream = torch.cuda.Stream()

  print(f'Rank {local_rank} waiting for other ranks...')
  dist.barrier()
  start_time = time.time()
  print(f'Rank {local_rank} computation starting...')

  grade_state = {
    "sum_acc": 0,
    "sum_sq_acc": 0,
    "sum_acc_norm": 0,
    "sum_sq_acc_norm": 0,
    "count": 0,
    "start_time": start_time,
  }

  kv_cache = {}
  output_cache = {}
  grade_queue = []
  ctx_grp_count = 0
  
  logprobsum_db = [[None for _ in range(NUM_CHOICES)] for _ in range(len(docs))]

  for batch_idx, batch in enumerate(batches):
    if not batch.use_cache:
      ctx_grp_count += 1
      if ctx_grp_count > CTX_GRP_LIMIT:
        break

    docs_in_batch = (tokenized_docs[i] for i in batch.data_idx)
    tokens = make_input_cpu_tensor_from_docs(docs_in_batch, batch).pin_memory().cuda(non_blocking=True)
    cont2ctx_gpu = None
    if batch.use_cache:
      cont2ctx_gpu = torch.Tensor(batch.cache_mapping).long().pin_memory().cuda(non_blocking=True)
    B = tokens.size(0)
    S = tokens.size(1)
    H = model_args.dim
    if DEBUG_SCHEDULE:
      logger.info(f'Rank {local_rank} ctx group id={batch_idx} size={(B, S, H)}')

    # load cache
    cache_k_list, cache_v_list, ctx_logprobs = [], [], None
    if batch.use_cache:
      if kv_cache[batch.cache_dep[0]]['merged']:
        cache_k_list = kv_cache[batch.cache_dep[0]]['k']
        cache_v_list = kv_cache[batch.cache_dep[0]]['v']
      else:
        cache_k_list = [torch.cat(tuple(kv_cache[cache_dep]['k'][j] for cache_dep in batch.cache_dep)) for j in range(15)]
        cache_v_list = [torch.cat(tuple(kv_cache[cache_dep]['v'][j] for cache_dep in batch.cache_dep)) for j in range(15)]
        for cache_dep in batch.cache_dep[1:]:
          kv_cache.pop(cache_dep)
        kv_cache[batch.cache_dep[0]]['k'] = cache_k_list
        kv_cache[batch.cache_dep[0]]['v'] = cache_v_list
        kv_cache[batch.cache_dep[0]]['merged'] = True
      if local_rank == world_size - 1:
        if output_cache[batch.cache_dep[0]]['merged']:
          ctx_logprobs = output_cache[batch.cache_dep[0]]['value']
        else:
          ctx_logprobs = torch.cat(tuple(output_cache[cache_dep]['value'] for cache_dep in batch.cache_dep))
          for cache_dep in batch.cache_dep[1:]:
            output_cache.pop(cache_dep)
          output_cache[batch.cache_dep[0]]['value'] = ctx_logprobs
          output_cache[batch.cache_dep[0]]['merged'] = True

    if batch.gen_cache and batch.first_minibatch:
      kv_cache.clear() 
      if local_rank == world_size - 1:
        output_cache.clear()

    # run prolog
    if local_rank == 0:
      h = pretb.forward(tokens)
    else:
      h = torch.empty((B, S, H), dtype=torch.float16, device='cuda')
      handle = dist.irecv(h, local_rank - 1, tag = batch_idx)
      handle.wait()

    # run transformer
    if DEBUG_PERFORMANCE:
      torch.cuda.synchronize()
      perf_start = time.time()

    h, new_cache_k_list, new_cache_v_list = tb.forward(
      h, batch.cache_len, use_cache = batch.use_cache, gen_cache = batch.gen_cache,
      cache_k_list = cache_k_list, cache_v_list = cache_v_list, cont2ctx = cont2ctx_gpu)

    if DEBUG_PERFORMANCE:
      torch.cuda.synchronize()
      elapsed = time.time() - perf_start
      FLOPs = 15 * (8 * B * S * H ** 2 + 4 * B * H * S ** 2 + 6 * B * S * H * 17920)
      TFLOPS = FLOPs / elapsed / 1e12
      BS_thr = B * S / elapsed
      print(f'Rank {local_rank} batch_idx={batch_idx} size={(B, S, H)} elapsed={elapsed} TFLOPS={TFLOPS} BS_thr={BS_thr}')

    run_grade(grade_queue, grade_state, d2h_stream)
    
    # run epilog
    if local_rank < world_size - 1:
      dist.isend(h, local_rank + 1, tag = batch_idx)
    else:
      if batch.gen_cache:
        ctx_h = h[:, -1, :] # [B, V]
        ctx_logits = posttb.forward(ctx_h)
        ctx_logprobs = torch.log_softmax(ctx_logits, dim=-1)
      if not batch.gen_cache:
        logits = posttb.forward(h)
        logprobs = torch.log_softmax(logits, dim=-1)

        with torch.cuda.stream(d2h_stream):
          torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
          logprobs_cpu = torch.empty_like(logprobs, device='cpu', pin_memory=True).copy_(logprobs, non_blocking=True)
          ctx_logprobs_cpu = None
          if ctx_logprobs is not None:
            ctx_logprobs_cpu = torch.empty_like(ctx_logprobs, device='cpu', pin_memory=True).copy_(ctx_logprobs, non_blocking=True)

          grade_queue.append((ctx_logprobs_cpu, logprobs_cpu, batch, logprobsum_db, docs, tokenized_docs))

    # update cache
    if batch.gen_cache:
      kv_cache[batch_idx] = {
        'k': new_cache_k_list,
        'v': new_cache_v_list,
        'merged': False,
      }
      if local_rank == world_size - 1:
        output_cache[batch_idx] = {
          'value': ctx_logprobs,
          'merged': False,
        }

  # empty remaining grade queue
  run_grade(grade_queue, grade_state, d2h_stream)

  torch.cuda.synchronize()
  elapsed = time.time() - grade_state['start_time']
  print(f'Rank {local_rank} finished in {elapsed} seconds')

  dist.barrier() # this prevent isend result in wrong result

  if local_rank == world_size - 1:
    print_result(grade_state)

if __name__ == "__main__":
  if DEBUG_PROFILE:
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
      fire.Fire(main)
    prof.export_chrome_trace(f'trace_{local_rank}.json')
  elif DEBUG_PYTHON_PROFILE:
    with cProfile.Profile() as pr:
      fire.Fire(main)
    pr.dump_stats(f'profile.prof')
    pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE).print_stats()
  else:
    fire.Fire(main)