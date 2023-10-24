import time
time_py_entered = time.time()

import pickle
from tokenizer import Tokenizer
from typing import Tuple
from multiprocessing import Process, Queue
import os
import sys
import fire
import time
import json
import numpy as np
import logging
import cProfile
import argparse

print(f'[MAIN] import done: {time.time() - time_py_entered:.2f} seconds')

NUM_CHOICES = 4
DATA_LIMIT = 22222
CTX_GRP_LIMIT = 22222
CTX_THR = 10000
CTX_MINIBATCH_THR = 2048
CONT_THR = 2048
USE_CUSTOM_COMM = True
DEBUG_SCHEDULE = False
DEBUG_ANSWER = False
DEBUG_PERFORMANCE = False
DEBUG_PROFILE = False
DEBUG_PYTHON_PROFILE = False
DEBUG_COMPARE_GOLD = False
DEBUG_TURN_OFF_COMM = False

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

# Given tokenized examples, retrun a Tensor suitable for input to the model
def make_input_cpu_tensor_from_docs(docs, batch):
  encs = [doc['ctx'] + doc['cont'] for doc in docs]
  S = batch.seq_len
  bsz = len(encs)
  tokens = torch.full((bsz, S), 0).long() # pad with 0, as it does not matter
  for i in range(bsz):
    l = min(len(encs[i]), batch.cache_len + S) - batch.cache_len
    tokens[i, :l] = torch.tensor(encs[i][batch.cache_len : batch.cache_len + l]).long()
  return tokens

# From model outputs, calculate log probabilities of each choice and save it to the table
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

# From the table, calculate accuracy and normalized accuracy
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

# Calculate accuracy for finished examples in the queue. Need to synchronize as the device to host communication is asynchronous.
def run_grade(grade_queue, grade_state, d2h_stream):
  if len(grade_queue) > 0:
    d2h_stream.synchronize()
  while len(grade_queue) > 0:
    ctx_logprobs_cpu, logprobs_cpu, batch, logprobsum_db, docs, tokenized_docs, logprobs_gpu, ctx_logprobs_gpu = grade_queue.pop(0)
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

# Mimic the print format of lm-evaluation-harness
def print_result(grade_state):
  acc = grade_state['sum_acc'] / grade_state['count']
  acc_norm = grade_state['sum_acc_norm'] / grade_state['count']
  acc_stddev = ((grade_state['sum_sq_acc'] / grade_state['count']) - acc ** 2) ** 0.5
  acc_norm_stddev = ((grade_state['sum_sq_acc_norm'] / grade_state['count']) - acc_norm ** 2) ** 0.5
  acc_stderr = acc_stddev / (grade_state['count'] ** 0.5)
  acc_norm_stderr = acc_norm_stddev / (grade_state['count'] ** 0.5)
  print(f'[Round 2 Accuracy Report]')
  print(f'|  Task   |Version| Metric |Value |   |Stderr|')
  print(f'|---------|------:|--------|-----:|---|-----:|')
  print(f'|hellaswag|      0|acc     |{acc:.4f}|±  |{acc_stderr:.4f}|')
  print(f'|         |       |acc_norm|{acc_norm:.4f}|±  |{acc_norm_stderr:.4f}|')

def print_time(time_round2):
  preprocess_time = time_round2['preprocess_done'] - time_round2['start']
  computation_time = time_round2['end'] - time_round2['preprocess_done']
  total_time = time_round2['end'] - time_round2['start']
  print(f'[Round 2 Time Report]')
  print(f'| Phase     | Time          |')
  print(f'|-----------|--------------:|')
  print(f'|Preprocess |{preprocess_time:15.9f}|')
  print(f'|Computation|{computation_time:15.9f}|')
  print(f'|Total      |{total_time:15.9f}|')

def main(
  local_rank,
  world_size,
  tokenizer_path: str,
  ckpt_dir: str,
  cache_dir: str,
  max_seq_len: int = 512,
  max_batch_size: int = 32,
):

  print(f'[Rank {local_rank}] main entered: {time.time() - time_py_entered:.2f} seconds')

  # if local_rank > 0:
  #    sys.stdout = open(os.devnull, "w")
  #    sys.stderr = open(os.devnull, "w")

  # Time measurement complying to Round2 rule
  time_round2 = {}
  time_round2['start'] = time.time()

  # Load and preprocess dataset with multiprocessing
  def _load_preprocess_and_schedule_dataset(q,):
    """================
    Tokenizer Disk I/O
    ================"""
    tokenizer = Tokenizer(model_path=tokenizer_path)
    print(f'[Rank {local_rank}] tokenizer loading done: {time.time() - time_py_entered:.2f} seconds')

    """================
    Dataset Disk I/O
    ================"""
    #dataset = load_dataset("hellaswag", cache_dir=cache_dir, split='validation')
    dataset = pickle.load(open('hellaswag_validation.pkl', 'rb'))
    print(f'[Rank {local_rank}] dataset loading done: {time.time() - time_py_entered:.2f} seconds')

    """================
    Dataset Tokenization and Scheduling
    ================"""
    import schedule
    docs, tokenized_docs, batches = schedule.preprocess_and_schedule_dataset(dataset, tokenizer, DATA_LIMIT, CTX_THR, CTX_MINIBATCH_THR, CONT_THR, False)
    max_bs, max_b, _ = schedule.max_sizes_of_batches(batches)
    q.put((docs, tokenized_docs, batches, max_bs, max_b))
    print(f'[Rank {local_rank}] preprocess_and_schedule_dataset done: {time.time() - time_py_entered:.2f} seconds')

  # As (the dataset preprocess time) ~ (model checkpoint load time), we overlap them
  q = Queue()
  p = Process(target=_load_preprocess_and_schedule_dataset, args=(q,))
  p.start()

  global torch
  import torch # ~ 2 seconds
  from model import ModelArgs, TransformerBlocks, PreTransformer, PostTransformer
  torch.cuda.set_device(local_rank)
  torch.manual_seed(1)
  print(f'[Rank {local_rank}] torch init done: {time.time() - time_py_entered:.2f} seconds')

  import teamh_c_helper # ~0.6 seconds
  if USE_CUSTOM_COMM:
    teamh_c_helper.init(local_rank, world_size)
    teamh_c_helper.init_comm()
  print(f'[Rank {local_rank}] Communication setup done: {time.time() - time_py_entered:.2f} seconds')

  """================
  Model Checkpoint Disk I/O + Send to GPU
  ================"""

  # Load model in cpu
  checkpoint = torch.load(os.path.join(ckpt_dir, f'30B_cpu_{local_rank}.pth'), map_location="cpu", mmap=True)
  with open(os.path.join(ckpt_dir, 'params.json'), "r") as f:
    params = json.loads(f.read())
  model_args: ModelArgs = ModelArgs(
    max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
  )
  model_args.vocab_size = 32000
  print(f'[Rank {local_rank}] checkpoint loading done: {time.time() - time_py_entered:.2f} seconds')

  # Send parameters to gpu
  torch.set_default_tensor_type(torch.cuda.HalfTensor)
  if local_rank == 0:
    pretb = PreTransformer(model_args)
    tb = TransformerBlocks(model_args, 0, 15)
  elif local_rank == 1:
    tb = TransformerBlocks(model_args, 15, 15)
  elif local_rank == 2:
    tb = TransformerBlocks(model_args, 30, 15)
  elif local_rank == 3:
    tb = TransformerBlocks(model_args, 45, 15)
    posttb = PostTransformer(model_args)
  print(f'[Rank {local_rank}] model init done: {time.time() - time_py_entered:.2f} seconds')
  if local_rank == 0:
    pretb.custom_load(checkpoint)
    tb.custom_load(checkpoint)
  elif local_rank == 1:
    tb.custom_load(checkpoint)
  elif local_rank == 2:
    tb.custom_load(checkpoint)
  elif local_rank == 3:
    tb.custom_load(checkpoint)
    posttb.custom_load(checkpoint)
  torch.set_default_tensor_type(torch.FloatTensor)
  print(f'[Rank {local_rank}] model loading done: {time.time() - time_py_entered:.2f} seconds')

  """================
  Warming up GPU with the smallest data
  ================"""
  #torch.cuda.nvtx.range_push('gpu warmup')
  # warmup PREFILL phase
  B, S, H = 1, 1, model_args.dim
  if local_rank == 0:
    # Run pre-transformer blocks
    h = pretb.forward(torch.zeros((B, S), dtype=torch.long, device='cuda'))
  else:
    # Receive tensor from previous rank
    h = torch.empty((B, S, H), dtype=torch.float16, device='cuda')
  partial_cache_k_list = torch.zeros((15, B, S, model_args.n_heads, model_args.dim // model_args.n_heads), dtype=torch.float16, device='cuda')
  partial_cache_v_list = torch.zeros((15, B, S, model_args.n_heads, model_args.dim // model_args.n_heads), dtype=torch.float16, device='cuda')
  h, _, _ = tb.forward(
    h, start_pos = 0, use_cache = False, gen_cache = True,
    cache_k_list = partial_cache_k_list, cache_v_list = partial_cache_v_list, cont2ctx = None)
  ctx_len = S
  # warmup DECODE phase
  B, S, H = 1, 1, model_args.dim
  if local_rank == 0:
    # Run pre-transformer blocks
    h = pretb.forward(torch.zeros((B, S), dtype=torch.long, device='cuda'))
  else:
    # Receive tensor from previous rank
    h = torch.empty((B, S, H), dtype=torch.float16, device='cuda')
  cont2ctx_gpu = torch.zeros((B,), dtype=torch.long, device='cuda')
  h, _, _ = tb.forward(
    h, start_pos = ctx_len, use_cache = True, gen_cache = False,
    cache_k_list = partial_cache_k_list, cache_v_list = partial_cache_v_list, cont2ctx = cont2ctx_gpu)
  #torch.cuda.nvtx.range_pop()
  print(f'[Rank {local_rank}] GPU warmup done: {time.time() - time_py_entered:.2f} seconds')
  # WARMUP END

  """================
  Join with dataset process
  ================"""
  #torch.cuda.nvtx.range_push('Join dataset process')
  docs, tokenized_docs, batches, max_bs, max_b = q.get()
  p.join()
  #torch.cuda.nvtx.range_pop()
  print(f'[Rank {local_rank}] dataset process joining done: {time.time() - time_py_entered:.2f} seconds')

  """================
  Cache setup
  ================"""
  buf_cache_k_list = torch.empty(15 * max_bs * model_args.n_heads * (model_args.dim // model_args.n_heads), dtype=torch.float16, device='cuda')
  buf_cache_v_list = torch.empty(15 * max_bs * model_args.n_heads * (model_args.dim // model_args.n_heads), dtype=torch.float16, device='cuda')
  buf_ctx_logprobs = torch.empty(max_b * model_args.vocab_size, dtype=torch.float16, device='cuda')
  print(f'[Rank {local_rank}] cache setup done: {time.time() - time_py_entered:.2f} seconds')

  """================
  Misc. initialization
  ================"""
  d2h_stream = torch.cuda.Stream()
  grade_queue = []
  ctx_grp_count = 0
  cache_k_list, cache_v_list, ctx_logprobs = None, None, None
  logprobsum_db = [[None for _ in range(NUM_CHOICES)] for _ in range(len(docs))]
  print(f'[Rank {local_rank}] right before computation loop: {time.time() - time_py_entered:.2f} seconds')

  time_round2['preprocess_done'] = time.time()
  grade_state = {
    "sum_acc": 0,
    "sum_sq_acc": 0,
    "sum_acc_norm": 0,
    "sum_sq_acc_norm": 0,
    "count": 0,
    "start_time": time_round2['preprocess_done'],
  }

  for batch_idx, batch in enumerate(batches):
    #batch_str = f'batch_idx={batch_idx} bsz={len(batch.data_idx)} use_cache={batch.use_cache} gen_cache={batch.gen_cache} seq_len={batch.seq_len} cache_len={batch.cache_len}'
    #torch.cuda.nvtx.range_push(batch_str)

    if not batch.use_cache:
      ctx_grp_count += 1
      if ctx_grp_count > CTX_GRP_LIMIT:
        break

    if batch.gen_cache and batch.first_minibatch:
      total_B = 0
      for i in range(batch_idx, len(batches)):
        if not batches[i].gen_cache:
          break
        total_B += len(batches[i].data_idx)
      del cache_k_list, cache_v_list
      sz = 15 * total_B * batch.seq_len * model_args.n_heads * (model_args.dim // model_args.n_heads)
      cache_k_list = buf_cache_k_list[:sz].view(15, total_B, batch.seq_len, model_args.n_heads, model_args.dim // model_args.n_heads)
      cache_v_list = buf_cache_v_list[:sz].view(15, total_B, batch.seq_len, model_args.n_heads, model_args.dim // model_args.n_heads)
      #print(f'Rank {local_rank} cache_k_list.size()={cache_k_list.size()} cache_v_list.size()={cache_v_list.size()}')
      if local_rank == world_size - 1:
        del ctx_logprobs
        sz = total_B * model_args.vocab_size
        ctx_logprobs = buf_ctx_logprobs[:sz].view(total_B, model_args.vocab_size)
      offset_B = 0

    if local_rank == 0:
      docs_in_batch = (tokenized_docs[i] for i in batch.data_idx)
      if DEBUG_TURN_OFF_COMM:
        tokens = torch.randint(0, model_args.vocab_size, (len(batch.data_idx), batch.seq_len), dtype=torch.long, device='cuda')
      else:
        tokens = make_input_cpu_tensor_from_docs(docs_in_batch, batch).pin_memory().cuda(non_blocking=True)
    cont2ctx_gpu = None
    if batch.use_cache:
      cont2ctx_gpu = torch.Tensor(batch.cache_mapping).long().pin_memory().cuda(non_blocking=True)
    B = len(batch.data_idx)
    S = batch.seq_len
    H = model_args.dim
    if DEBUG_SCHEDULE:
      logger.info(f'Rank {local_rank} ctx group id={batch_idx} size={(B, S, H)}')

    # Prolog
    if local_rank == 0:
      # Run pre-transformer blocks
      h = pretb.forward(tokens)
    else:
      # Receive tensor from previous rank
      if DEBUG_TURN_OFF_COMM:
        h = torch.randn((B, S, H), dtype=torch.float16, device='cuda')
      else:
        h = torch.empty((B, S, H), dtype=torch.float16, device='cuda')
        if USE_CUSTOM_COMM:
          teamh_c_helper.recv(h)
        else:
          handle = dist.irecv(h, local_rank - 1, tag = batch_idx)
          handle.wait()

    # Run transformer blocks
    if DEBUG_PERFORMANCE:
      torch.cuda.synchronize()
      perf_start = time.time()

    if batch.gen_cache:
      partial_cache_k_list = cache_k_list[:, offset_B : offset_B + B, :, :, :]
      partial_cache_v_list = cache_v_list[:, offset_B : offset_B + B, :, :, :]
    else:
      partial_cache_k_list = cache_k_list
      partial_cache_v_list = cache_v_list

    h, new_cache_k_list, new_cache_v_list = tb.forward(
      h, batch.cache_len, use_cache = batch.use_cache, gen_cache = batch.gen_cache,
      cache_k_list = partial_cache_k_list, cache_v_list = partial_cache_v_list, cont2ctx = cont2ctx_gpu,
      last_token_only = batch.gen_cache and local_rank == world_size - 1)

    if DEBUG_PERFORMANCE:
      torch.cuda.synchronize()
      elapsed = time.time() - perf_start
      FLOPs = 15 * (8 * B * S * H ** 2 + 4 * B * H * S ** 2 + 6 * B * S * H * 17920)
      TFLOPS = FLOPs / elapsed / 1e12
      BS_thr = B * S / elapsed
      print(f'Rank {local_rank} batch_idx={batch_idx} size={(B, S, H)} elapsed={elapsed} TFLOPS={TFLOPS} BS_thr={BS_thr}')

    # While GPU is running, we can run the grading process
    run_grade(grade_queue, grade_state, d2h_stream)
    
    # Epilog
    if local_rank < world_size - 1:
      # Send tensor to next rank
      if DEBUG_TURN_OFF_COMM:
        pass
      else:
        if USE_CUSTOM_COMM:
          teamh_c_helper.send(h, batch_idx == 0)
        else:
          dist.isend(h, local_rank + 1, tag = batch_idx)
      del h
    else:
      if batch.gen_cache:
        # Calculate log probabilities to be saved in cache
        h = h[:, -1, :] # [B, V]
        h = posttb.forward(h)
        h = torch.log_softmax(h, dim=-1)
        ctx_logprobs[offset_B : offset_B + B, :].copy_(h)
        del h
      if not batch.gen_cache:
        # Calculate log probabilities, send to cpu, and queue grading process
        h = posttb.forward(h)
        logprobs = torch.log_softmax(h, dim=-1)
        ctx_logprobs_copied = ctx_logprobs.clone()

        # You also should hold gpu buffers until the grading process is finished (preventing freeing of the memory)
        with torch.cuda.stream(d2h_stream):
          torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
          logprobs_cpu = torch.empty_like(logprobs, device='cpu', pin_memory=True).copy_(logprobs, non_blocking=True)
          ctx_logprobs_cpu = torch.empty_like(ctx_logprobs_copied, device='cpu', pin_memory=True).copy_(ctx_logprobs_copied, non_blocking=True)

          grade_queue.append((ctx_logprobs_cpu, logprobs_cpu, batch, logprobsum_db, docs, tokenized_docs, logprobs, ctx_logprobs_copied))
        del h, logprobs, ctx_logprobs_copied

    offset_B += B

    #torch.cuda.nvtx.range_pop()

  # Empty remaining grade queue
  run_grade(grade_queue, grade_state, d2h_stream)

  torch.cuda.synchronize()
  print(f'[Rank {local_rank}] Computation + Preprocess: {time.time() - time_round2["start"]:.2f} seconds')
  print(f'[Rank {local_rank}] Computation ONLY: {time.time() - time_round2["preprocess_done"]:.2f} seconds')

  if USE_CUSTOM_COMM:
    teamh_c_helper.finalize()

  # Time measurement complying to Round2 rule
  time_round2['end'] = time.time()

  if local_rank == world_size - 1:
    print_result(grade_state)
    print_time(time_round2)

def single_process(local_rank, world_size, tokenizer_path, ckpt_dir, cache_dir):
  if DEBUG_PROFILE:
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
      fire.Fire(main)
    prof.export_chrome_trace(f'trace_{local_rank}.json')
  elif DEBUG_PYTHON_PROFILE:
    with cProfile.Profile() as pr:
      fire.Fire(main)
    pr.dump_stats(f'profile.prof')
    #pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE).print_stats()
  else:
    main(local_rank, world_size, tokenizer_path, ckpt_dir, cache_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--nproc_per_node', type=int, required=True)
  parser.add_argument('--tokenizer_path', type=str, required=True)
  parser.add_argument('--ckpt_dir', type=str, required=True)
  parser.add_argument('--cache_dir', type=str, required=True)
  args = parser.parse_args()
  ps = []
  for local_rank in range(args.nproc_per_node):
    p = Process(target=single_process, args=(local_rank, args.nproc_per_node, args.tokenizer_path, args.ckpt_dir, args.cache_dir))
    p.start()
    print(f'[MAIN] process {local_rank} start: {time.time() - time_py_entered:.2f} seconds')
    ps.append(p)
  for local_rank in range(args.nproc_per_node):
    p.join()
    print(f'[MAIN] process {local_rank} joined: {time.time() - time_py_entered:.2f} seconds')
