# Scheduling related functions for hellaswag

import re
from dataclasses import dataclass
import teamh_c_helper

@dataclass
class Batch:
  data_idx: tuple[int] # [0, num_data * NUM_CHOICES)
  use_cache: bool
  gen_cache: bool
  seq_len: int # length excluding cache part

  # Used when gen_cache = True
  first_minibatch: bool
  
  # Used when use_cache = True
  cache_mapping: tuple[int]
  cache_len: int
  cache_dep: tuple[int]

  def __init__(self, data_idx, use_cache, gen_cache, seq_len, first_minibatch = None, cache_mapping = None, cache_len = 0, cache_dep = None):
    self.data_idx = data_idx
    self.use_cache = use_cache
    self.gen_cache = gen_cache
    self.seq_len = seq_len
    self.first_minibatch = first_minibatch
    self.cache_mapping = cache_mapping
    self.cache_len = cache_len
    self.cache_dep = cache_dep

# Taken from lm-evaluation-harness
def process_example(example, prefix_activity_label = True):
  def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text
  ctx = example['ctx_a'] + " " + example['ctx_b'].capitalize().rstrip()
  if prefix_activity_label:
    query = preprocess(example["activity_label"] + ": " + ctx)
  else:
    query = preprocess(ctx)
  endings = [preprocess(ending).lstrip() for ending in example["endings"]]

  out_example = {
    "query": query,
    "choices": endings,
    "gold": int(example["label"]),
  }
  return out_example

def encode_input(tokenizer, example):
  query_enc = tokenizer.encode(example['query'])
  new_reqs = [{
    'ctx': query_enc,
    'cont': tokenizer.encode(choice),
    } for choice in example['choices']]
  return new_reqs

# length is minimum of each block
def schedule_min(lengths, thr, debug=False):
  ctx_idx_c, ctx_blocks_c, total_ctx_wasted_c = teamh_c_helper.schedule_min_c(lengths, thr)

  if debug:
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
    if E[i] == -1:
      return None, None, None
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
    ctx_idx, ctx_blocks, total_ctx_wasted = idx, blocks, D[N - 1]
    assert ctx_idx == ctx_idx_c, f'{ctx_idx} != {ctx_idx_c}'
    assert ctx_blocks == ctx_blocks_c, f'{ctx_blocks} != {ctx_blocks_c}'
    assert total_ctx_wasted == total_ctx_wasted_c, f'{total_ctx_wasted} != {total_ctx_wasted_c}'

  assert ctx_blocks_c, "Failed to schedule"
  return ctx_idx_c, ctx_blocks_c, total_ctx_wasted_c

# length is maximum of each block
def schedule_max(lengths, thr, debug=False):
  ctx_idx_c, ctx_blocks_c, total_ctx_wasted_c = teamh_c_helper.schedule_max_c(lengths, thr)

  if debug:
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
    if E[i] == -1:
      return None, None, None
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
    ctx_idx, ctx_blocks, total_ctx_wasted = idx, blocks, D[N - 1]
    assert ctx_idx == ctx_idx_c, f'{ctx_idx} != {ctx_idx_c}'
    assert ctx_blocks == ctx_blocks_c, f'{ctx_blocks} != {ctx_blocks_c}'
    assert total_ctx_wasted == total_ctx_wasted_c, f'{total_ctx_wasted} != {total_ctx_wasted_c}'

  assert ctx_blocks_c, "Failed to schedule"
  return ctx_idx_c, ctx_blocks_c, total_ctx_wasted_c

def schedule_max_opt_32_128(lengths, thr):
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
    if rect_area >= thr and rect_area <= 4096:
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
      if rect_area >= thr and rect_area <= 4096 and D[i] > D[j] + penalty:
        D[i] = D[j] + penalty
        E[i] = j
  blocks = []
  i = N - 1
  if E[i] == -1:
    return None, None, None
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

  return idx, blocks, D[N - 1]

def preprocess_and_schedule_dataset(dataset, tokenizer, num_data, ctx_threshold, ctx_minibatch_threshold, cont_threshold, prefix_activity_label = True):
  NUM_CHOICES = 4
  # encode the whole dataset
  whole_pe = []
  whole_ei = []
  for i, data in enumerate(dataset):
    if i == num_data:
      break
    if i % 1000 == 0:
      print(f'Processing {i}...')
    pe = process_example(data, prefix_activity_label=prefix_activity_label)
    whole_pe.append(pe)
    whole_ei.extend(encode_input(tokenizer, pe))

  # whole_pe is list of {'query': str, 'choices': [str], 'gold': int} with length num_data
  # whole_ei is list of {'ctx': [int], 'cont': [int]} with length num_data * NUM_CHOICES

  ctx_lengths = [len(whole_ei[i]['ctx']) for i in range(0, len(whole_ei), NUM_CHOICES)]
  ctx_idx, ctx_blocks, total_ctx_wasted = schedule_min(ctx_lengths, ctx_threshold)

  # ctx_idx points to whole_pe [0, num_data)

  batches = []
  schedule_info = []
  s = 0
  sum_ctx_block_wasted = 0
  sum_ctx_block_effective = 0
  for i, ctx_block_size in enumerate(ctx_blocks):
    s += ctx_block_size
    ctx_block_start, ctx_block_end = s - ctx_block_size, s
    ctx_min_len = len(whole_ei[ctx_idx[ctx_block_start] * NUM_CHOICES]['ctx'])

    # Minibath ctx
    def evenDivide(num, div):
      groupSize, remainder = divmod(num, div)
      return [groupSize + (1 if x < remainder else 0) for x in range(div)]
    ctx_minibatch_blocks = evenDivide(ctx_block_size, ctx_block_size * ctx_min_len // ctx_minibatch_threshold)
    ctx_minibatch_block_start = ctx_block_start
    cache_dep = []
    for idx, ctx_minibatch in enumerate(ctx_minibatch_blocks):
      data_idx = tuple(ctx_idx[j] * NUM_CHOICES for j in range(ctx_minibatch_block_start, ctx_minibatch_block_start + ctx_minibatch))
      batches.append(Batch(data_idx, use_cache = False, gen_cache = True, first_minibatch = idx == 0, seq_len = ctx_min_len))
      cache_dep.append(len(batches)-1)
      ctx_minibatch_block_start += ctx_minibatch
    cache_dep = tuple(cache_dep)

    # ctx schedule validation
    data_idx = tuple(ctx_idx[j] * NUM_CHOICES for j in range(ctx_block_start, ctx_block_end))
    lengths = tuple(len(whole_ei[data_idx[j]]['ctx']) for j in range(ctx_block_size))
    min_S = min(lengths)
    ctx_block_wasted = sum(lengths[j] - min_S for j in range(ctx_block_size))
    ctx_block_effective = sum(lengths[j] for j in range(ctx_block_size))
    sum_ctx_block_wasted += ctx_block_wasted
    sum_ctx_block_effective += ctx_block_effective
    #print(f'ctx_block_wasted: {ctx_block_wasted}, ctx_block_effective: {ctx_block_effective}')

    cur_ctx_grp = []
    cur_grp = []
    for j in range(ctx_block_start, ctx_block_end):
      cur_ctx_grp.append(ctx_idx[j] * NUM_CHOICES)
      cur_grp.extend([ctx_idx[j] * NUM_CHOICES + k for k in range(NUM_CHOICES)])
    cont_lengths = tuple(len(whole_ei[j]['ctx']) + len(whole_ei[j]['cont']) - ctx_min_len - 1 for j in cur_grp)
    #cont_idx, cont_blocks, total_cont_wasted = schedule_max_opt_32_128(cont_lengths, cont_threshold)
    #print(f'schedule failed, fallback to schedule_max; ctx_block {ctx_block_start} ~ {ctx_block_end}')
    #if cont_idx is None:
    #  cont_idx, cont_blocks, total_cont_wasted = schedule_max(cont_lengths, cont_threshold)
    cont_idx, cont_blocks, total_cont_wasted = schedule_max(cont_lengths, cont_threshold)

    # cont_idx points to cur_grp; length is number of conts in current ctx group
    # cur_grp points to whole_ei; length is num_data * NUM_CHOICES

    sum_cont_block_wasted = 0
    sum_cont_block_effective = 0
    cont_block_end = 0
    for cont_block_size in cont_blocks:
      cont_block_end += cont_block_size
      cont_block_start = cont_block_end - cont_block_size
      data_idx = tuple(cur_grp[cont_idx[j]] for j in range(cont_block_start, cont_block_end))
      cache_mapping = tuple(cont_idx[j] // NUM_CHOICES for j in range(cont_block_start, cont_block_end))
      seq_len = max(len(whole_ei[data_idx[j]]['ctx']) + len(whole_ei[data_idx[j]]['cont']) - ctx_min_len - 1 for j in range(cont_block_size))
      batches.append(Batch(data_idx, use_cache = True, gen_cache = False, seq_len = seq_len, cache_mapping=cache_mapping, cache_len=ctx_min_len, cache_dep=cache_dep))
      # cont schedule validation
      lengths = tuple(len(whole_ei[data_idx[j]]['ctx']) + len(whole_ei[data_idx[j]]['cont']) for j in range(cont_block_size))
      max_S = max(lengths)
      cont_block_wasted = sum(max_S - lengths[j] for j in range(cont_block_size))
      cont_block_effective = sum(lengths[j] for j in range(cont_block_size))
      sum_cont_block_wasted += cont_block_wasted
      sum_cont_block_effective += cont_block_effective
      #print(f'cont_block_wasted: {cont_block_wasted}, cont_block_effective: {cont_block_effective}')
    assert sum_cont_block_wasted == total_cont_wasted
    # print(f'total_cont_wasted: {sum_cont_block_wasted}, total_cont_effective: {sum_cont_block_effective}')
    # print(f'total_cont_efficiency: {sum_cont_block_effective / (sum_cont_block_effective + sum_cont_block_wasted)}')
  assert sum_ctx_block_wasted == total_ctx_wasted
  print(f'total_ctx_wasted: {sum_ctx_block_wasted}, total_ctx_effective: {sum_ctx_block_effective}')
  print(f'total_ctx_efficiency: {sum_ctx_block_effective / (sum_ctx_block_effective + sum_ctx_block_wasted)}')

  
  return whole_pe, whole_ei, batches

def evaluate_schedule(whole_pe, whole_ei, batches):
  optimal = 0
  for i, ei in enumerate(whole_ei):
    if i % 4 == 0:
      optimal += len(ei['ctx'])
    optimal += len(ei['cont'])

  reality = 0
  for batch in batches:
    if not batch.use_cache:
      reality += min(len(whole_ei[idx]['ctx']) for idx in batch.data_idx) * len(batch.data_idx)
    if batch.use_cache:
      reality += max(len(whole_ei[idx]['ctx']) + len(whole_ei[idx]['cont']) - batch.cache_len - 1 for idx in batch.data_idx) * len(batch.data_idx)
  
  wasted = reality - optimal
  print(f'optimal: {optimal}, reality: {reality}, wasted: {wasted}, efficiency: {optimal / reality}')

def preprocess_and_schedule_dataset_typeB(dataset, tokenizer, num_data, ctx_threshold, cont_threshold):
  NUM_CHOICES = 4
  # encode the whole dataset
  whole_pe = []
  whole_ei = []
  for i, data in enumerate(dataset):
    if i == num_data:
      break
    if i % 1000 == 0:
      print(f'Processing {i}...')
    pe = process_example(data)
    whole_pe.append(pe)
    whole_ei.extend(encode_input(tokenizer, pe))

  # whole_pe is list of {'query': str, 'choices': [str], 'gold': int} with length num_data
  # whole_ei is list of {'ctx': [int], 'cont': [int]} with length num_data * NUM_CHOICES

  B = 32
  batches = []
  for start_idx in range(0, len(whole_ei), B):
    end_idx = min(start_idx + B, len(whole_ei))
    data_idx = tuple(range(start_idx, end_idx))
    seq_len = max(len(whole_ei[idx]['ctx']) + len(whole_ei[idx]['cont']) - 1 for idx in data_idx)
    batches.append(Batch(data_idx, use_cache = False, gen_cache = False, seq_len = seq_len))

  return whole_pe, whole_ei, batches

def preprocess_and_schedule_dataset_typeC(dataset, tokenizer, num_data, ctx_threshold, cont_threshold):
  NUM_CHOICES = 4
  # encode the whole dataset
  whole_pe = []
  whole_ei = []
  for i, data in enumerate(dataset):
    if i == num_data:
      break
    if i % 1000 == 0:
      print(f'Processing {i}...')
    pe = process_example(data)
    whole_pe.append(pe)
    whole_ei.extend(encode_input(tokenizer, pe))

  # whole_pe is list of {'query': str, 'choices': [str], 'gold': int} with length num_data
  # whole_ei is list of {'ctx': [int], 'cont': [int]} with length num_data * NUM_CHOICES

  idx = [i for i in range(len(whole_ei))]
  idx.sort(key=lambda x: len(whole_ei[x]['ctx']) + len(whole_ei[x]['cont']) - 1)

  B = 32
  batches = []
  for start_idx in range(0, len(whole_ei), B):
    end_idx = min(start_idx + B, len(whole_ei))
    data_idx = tuple(idx[i] for i in range(start_idx, end_idx))
    seq_len = max(len(whole_ei[idx]['ctx']) + len(whole_ei[idx]['cont']) - 1 for idx in data_idx)
    batches.append(Batch(data_idx, use_cache = False, gen_cache = False, seq_len = seq_len))

  return whole_pe, whole_ei, batches

def preprocess_and_schedule_dataset_typeD(dataset, tokenizer, num_data, ctx_threshold, cont_threshold):
  NUM_CHOICES = 4
  # encode the whole dataset
  whole_pe = []
  whole_ei = []
  for i, data in enumerate(dataset):
    if i == num_data:
      break
    if i % 1000 == 0:
      print(f'Processing {i}...')
    pe = process_example(data)
    whole_pe.append(pe)
    whole_ei.extend(encode_input(tokenizer, pe))

  # whole_pe is list of {'query': str, 'choices': [str], 'gold': int} with length num_data
  # whole_ei is list of {'ctx': [int], 'cont': [int]} with length num_data * NUM_CHOICES

  lengths = [len(whole_ei[i]['ctx']) + len(whole_ei[i]['cont']) - 1 for i in range(len(whole_ei))]
  idx, blocks, _ = schedule_max(lengths, cont_threshold)

  block_end = 0
  batches = []
  for block_size in blocks:
    block_end += block_size
    block_start = block_end - block_size
    data_idx = tuple(idx[i] for i in range(block_start, block_end))
    seq_len = max(len(whole_ei[idx]['ctx']) + len(whole_ei[idx]['cont']) - 1 for idx in data_idx)
    batches.append(Batch(data_idx, use_cache = False, gen_cache = False, seq_len = seq_len))

  return whole_pe, whole_ei, batches

def preprocess_and_schedule_dataset_typeE(dataset, tokenizer, num_data, ctx_threshold, cont_threshold, prefix_activity_label = True):
  NUM_CHOICES = 4
  # encode the whole dataset
  whole_pe = []
  whole_ei = []
  for i, data in enumerate(dataset):
    if i == num_data:
      break
    if i % 1000 == 0:
      print(f'Processing {i}...')
    pe = process_example(data, prefix_activity_label=prefix_activity_label)
    whole_pe.append(pe)
    whole_ei.extend(encode_input(tokenizer, pe))

  # whole_pe is list of {'query': str, 'choices': [str], 'gold': int} with length num_data
  # whole_ei is list of {'ctx': [int], 'cont': [int]} with length num_data * NUM_CHOICES

  ctx_lengths = [len(whole_ei[i]['ctx']) for i in range(0, len(whole_ei), NUM_CHOICES)]
  # TODO
  ctx_idx, ctx_blocks, total_ctx_wasted = schedule_min(ctx_lengths, ctx_threshold)

  # ctx_idx points to whole_pe [0, num_data)

  batches = []
  schedule_info = []
  s = 0
  sum_ctx_block_wasted = 0
  sum_ctx_block_effective = 0
  for i, ctx_block_size in enumerate(ctx_blocks):
    s += ctx_block_size
    ctx_block_start, ctx_block_end = s - ctx_block_size, s
    ctx_min_len = len(whole_ei[ctx_idx[ctx_block_start] * NUM_CHOICES]['ctx'])

    data_idx = tuple(ctx_idx[j] * NUM_CHOICES for j in range(ctx_block_start, ctx_block_end))
    batches.append(Batch(data_idx, use_cache = False, gen_cache = True, seq_len = ctx_min_len))
    cache_dep = len(batches) - 1

    # ctx schedule validation
    lengths = tuple(len(whole_ei[data_idx[j]]['ctx']) for j in range(ctx_block_size))
    min_S = min(lengths)
    ctx_block_wasted = sum(lengths[j] - min_S for j in range(ctx_block_size))
    ctx_block_effective = sum(lengths[j] for j in range(ctx_block_size))
    sum_ctx_block_wasted += ctx_block_wasted
    sum_ctx_block_effective += ctx_block_effective
    #print(f'ctx_block_wasted: {ctx_block_wasted}, ctx_block_effective: {ctx_block_effective}')

    cur_ctx_grp = []
    cur_grp = []
    for j in range(ctx_block_start, ctx_block_end):
      cur_ctx_grp.append(ctx_idx[j] * NUM_CHOICES)
      cur_grp.extend([ctx_idx[j] * NUM_CHOICES + k for k in range(NUM_CHOICES)])
    cont_lengths = tuple(len(whole_ei[j]['ctx']) + len(whole_ei[j]['cont']) - ctx_min_len - 1 for j in cur_grp)



    cont_idx = list(range(len(cont_lengths)))
    cont_idx.sort(key=lambda x: cont_lengths[x])
    B = 32
    cont_blocks = [B for _ in range(len(cont_lengths) // B)]
    if len(cont_lengths) % B != 0:
      cont_blocks[-1] += len(cont_lengths) % B

    # cont_idx points to cur_grp; length is number of conts in current ctx group
    # cur_grp points to whole_ei; length is num_data * NUM_CHOICES

    sum_cont_block_wasted = 0
    sum_cont_block_effective = 0
    cont_block_end = 0
    for cont_block_size in cont_blocks:
      cont_block_end += cont_block_size
      cont_block_start = cont_block_end - cont_block_size
      data_idx = tuple(cur_grp[cont_idx[j]] for j in range(cont_block_start, cont_block_end))
      cache_mapping = tuple(cont_idx[j] // NUM_CHOICES for j in range(cont_block_start, cont_block_end))
      seq_len = max(len(whole_ei[data_idx[j]]['ctx']) + len(whole_ei[data_idx[j]]['cont']) - ctx_min_len - 1 for j in range(cont_block_size))
      batches.append(Batch(data_idx, use_cache = True, gen_cache = False, seq_len = seq_len, cache_mapping=cache_mapping, cache_len=ctx_min_len, cache_dep=cache_dep))
      # cont schedule validation
      lengths = tuple(len(whole_ei[data_idx[j]]['ctx']) + len(whole_ei[data_idx[j]]['cont']) for j in range(cont_block_size))
      max_S = max(lengths)
      cont_block_wasted = sum(max_S - lengths[j] for j in range(cont_block_size))
      cont_block_effective = sum(lengths[j] for j in range(cont_block_size))
      sum_cont_block_wasted += cont_block_wasted
      sum_cont_block_effective += cont_block_effective
      #print(f'cont_block_wasted: {cont_block_wasted}, cont_block_effective: {cont_block_effective}')

  
  return whole_pe, whole_ei, batches

def preprocess_and_schedule_dataset_typeF(dataset, tokenizer, num_data, ctx_threshold, cont_threshold, prefix_activity_label = True):
  NUM_CHOICES = 4
  # encode the whole dataset
  whole_pe = []
  whole_ei = []
  for i, data in enumerate(dataset):
    if i == num_data:
      break
    if i % 1000 == 0:
      print(f'Processing {i}...')
    pe = process_example(data, prefix_activity_label=prefix_activity_label)
    whole_pe.append(pe)
    whole_ei.extend(encode_input(tokenizer, pe))

  # whole_pe is list of {'query': str, 'choices': [str], 'gold': int} with length num_data
  # whole_ei is list of {'ctx': [int], 'cont': [int]} with length num_data * NUM_CHOICES

  ctx_lengths = [len(whole_ei[i]['ctx']) for i in range(0, len(whole_ei), NUM_CHOICES)]
  # TODO
  #ctx_idx, ctx_blocks, total_ctx_wasted = schedule_min(ctx_lengths, ctx_threshold)
  ctx_idx = list(range(len(ctx_lengths)))
  ctx_idx.sort(key=lambda x: ctx_lengths[x])
  B = 128
  ctx_blocks = [B for _ in range(len(ctx_lengths) // B)]
  if len(ctx_lengths) % B != 0:
    ctx_blocks[-1] += len(ctx_lengths) % B

  # ctx_idx points to whole_pe [0, num_data)

  batches = []
  schedule_info = []
  s = 0
  sum_ctx_block_wasted = 0
  sum_ctx_block_effective = 0
  for i, ctx_block_size in enumerate(ctx_blocks):
    s += ctx_block_size
    ctx_block_start, ctx_block_end = s - ctx_block_size, s
    ctx_min_len = len(whole_ei[ctx_idx[ctx_block_start] * NUM_CHOICES]['ctx'])

    data_idx = tuple(ctx_idx[j] * NUM_CHOICES for j in range(ctx_block_start, ctx_block_end))
    batches.append(Batch(data_idx, use_cache = False, gen_cache = True, seq_len = ctx_min_len))
    cache_dep = len(batches) - 1

    # ctx schedule validation
    lengths = tuple(len(whole_ei[data_idx[j]]['ctx']) for j in range(ctx_block_size))
    min_S = min(lengths)
    ctx_block_wasted = sum(lengths[j] - min_S for j in range(ctx_block_size))
    ctx_block_effective = sum(lengths[j] for j in range(ctx_block_size))
    sum_ctx_block_wasted += ctx_block_wasted
    sum_ctx_block_effective += ctx_block_effective
    #print(f'ctx_block_wasted: {ctx_block_wasted}, ctx_block_effective: {ctx_block_effective}')

    cur_ctx_grp = []
    cur_grp = []
    for j in range(ctx_block_start, ctx_block_end):
      cur_ctx_grp.append(ctx_idx[j] * NUM_CHOICES)
      cur_grp.extend([ctx_idx[j] * NUM_CHOICES + k for k in range(NUM_CHOICES)])
    cont_lengths = tuple(len(whole_ei[j]['ctx']) + len(whole_ei[j]['cont']) - ctx_min_len - 1 for j in cur_grp)



    cont_idx = list(range(len(cont_lengths)))
    cont_idx.sort(key=lambda x: cont_lengths[x])
    B = 32
    cont_blocks = [B for _ in range(len(cont_lengths) // B)]
    if len(cont_lengths) % B != 0:
      cont_blocks[-1] += len(cont_lengths) % B

    # cont_idx points to cur_grp; length is number of conts in current ctx group
    # cur_grp points to whole_ei; length is num_data * NUM_CHOICES

    sum_cont_block_wasted = 0
    sum_cont_block_effective = 0
    cont_block_end = 0
    for cont_block_size in cont_blocks:
      cont_block_end += cont_block_size
      cont_block_start = cont_block_end - cont_block_size
      data_idx = tuple(cur_grp[cont_idx[j]] for j in range(cont_block_start, cont_block_end))
      cache_mapping = tuple(cont_idx[j] // NUM_CHOICES for j in range(cont_block_start, cont_block_end))
      seq_len = max(len(whole_ei[data_idx[j]]['ctx']) + len(whole_ei[data_idx[j]]['cont']) - ctx_min_len - 1 for j in range(cont_block_size))
      batches.append(Batch(data_idx, use_cache = True, gen_cache = False, seq_len = seq_len, cache_mapping=cache_mapping, cache_len=ctx_min_len, cache_dep=cache_dep))
      # cont schedule validation
      lengths = tuple(len(whole_ei[data_idx[j]]['ctx']) + len(whole_ei[data_idx[j]]['cont']) for j in range(cont_block_size))
      max_S = max(lengths)
      cont_block_wasted = sum(max_S - lengths[j] for j in range(cont_block_size))
      cont_block_effective = sum(lengths[j] for j in range(cont_block_size))
      sum_cont_block_wasted += cont_block_wasted
      sum_cont_block_effective += cont_block_effective
      #print(f'cont_block_wasted: {cont_block_wasted}, cont_block_effective: {cont_block_effective}')

  
  return whole_pe, whole_ei, batches

def preprocess_and_schedule_dataset_typeG(dataset, tokenizer, num_data, ctx_threshold, cont_threshold, prefix_activity_label = True):
  NUM_CHOICES = 4
  # encode the whole dataset
  whole_pe = []
  whole_ei = []
  for i, data in enumerate(dataset):
    if i == num_data:
      break
    if i % 1000 == 0:
      print(f'Processing {i}...')
    pe = process_example(data, prefix_activity_label=prefix_activity_label)
    whole_pe.append(pe)
    whole_ei.extend(encode_input(tokenizer, pe))

  idx = [i for i in range(len(whole_pe))]
  idx.sort(key=lambda x: whole_ei[x * NUM_CHOICES]['ctx'])
  idx.sort(key=lambda x: len(whole_ei[x * NUM_CHOICES]['ctx']))
  count = 0
  for i in range(1, len(idx)):
    if whole_ei[idx[i - 1] * NUM_CHOICES]['ctx'] == whole_ei[idx[i] * NUM_CHOICES]['ctx']:
      count += 1


  #encs = [whole_ei[i]['ctx'] + whole_ei[i]['cont'] for i in range(0, len(whole_ei))]
  #encs.sort()
  #print(len(encs))
  #for enc in encs:
  #  print(' '.join([str(i) for i in [len(enc)] + enc]))
  return

  #def get_score(start_b, end_b, s):
  #  prev = []
  #  score = 0
  #  for b in range(start_b, end_b):
  #    if len(encs[b]) < s:
  #      return -1
  #    if prev == encs[b][:s]:
  #      score += s
  #    else:
  #      prev = encs[b][:s]
  #  return score

  #N = len(encs)
  #for i in range(N):
  #  B = i + 1
  #  S = 1
  #  max_score = get_score(0, B, S)
  #  while True:
  #    S += 1
  #    new_score = get_score(0, B, S)
  #    print(f'new_score = {new_score}')
  #    if max_score > new_score:
  #      max_S = S - 1
  #      break
  #    max_score = new_score
  #  print(f'for B={B}, max_S={max_S}, max_score={max_score}')
  #return



  # whole_pe is list of {'query': str, 'choices': [str], 'gold': int} with length num_data
  # whole_ei is list of {'ctx': [int], 'cont': [int]} with length num_data * NUM_CHOICES

  ctx_lengths = [len(whole_ei[i]['ctx']) for i in range(0, len(whole_ei), NUM_CHOICES)]
  # TODO
  #ctx_idx, ctx_blocks, total_ctx_wasted = schedule_min(ctx_lengths, ctx_threshold)
  ctx_idx = list(range(len(ctx_lengths)))
  ctx_idx.sort(key=lambda x: ctx_lengths[x])
  B = 128
  ctx_blocks = [B for _ in range(len(ctx_lengths) // B)]
  if len(ctx_lengths) % B != 0:
    ctx_blocks[-1] += len(ctx_lengths) % B

  # ctx_idx points to whole_pe [0, num_data)

  batches = []
  schedule_info = []
  s = 0
  sum_ctx_block_wasted = 0
  sum_ctx_block_effective = 0
  for i, ctx_block_size in enumerate(ctx_blocks):
    s += ctx_block_size
    ctx_block_start, ctx_block_end = s - ctx_block_size, s
    ctx_min_len = len(whole_ei[ctx_idx[ctx_block_start] * NUM_CHOICES]['ctx'])

    data_idx = tuple(ctx_idx[j] * NUM_CHOICES for j in range(ctx_block_start, ctx_block_end))
    batches.append(Batch(data_idx, use_cache = False, gen_cache = True, seq_len = ctx_min_len))
    cache_dep = len(batches) - 1

    # ctx schedule validation
    lengths = tuple(len(whole_ei[data_idx[j]]['ctx']) for j in range(ctx_block_size))
    min_S = min(lengths)
    ctx_block_wasted = sum(lengths[j] - min_S for j in range(ctx_block_size))
    ctx_block_effective = sum(lengths[j] for j in range(ctx_block_size))
    sum_ctx_block_wasted += ctx_block_wasted
    sum_ctx_block_effective += ctx_block_effective
    #print(f'ctx_block_wasted: {ctx_block_wasted}, ctx_block_effective: {ctx_block_effective}')

    cur_ctx_grp = []
    cur_grp = []
    for j in range(ctx_block_start, ctx_block_end):
      cur_ctx_grp.append(ctx_idx[j] * NUM_CHOICES)
      cur_grp.extend([ctx_idx[j] * NUM_CHOICES + k for k in range(NUM_CHOICES)])
    cont_lengths = tuple(len(whole_ei[j]['ctx']) + len(whole_ei[j]['cont']) - ctx_min_len - 1 for j in cur_grp)



    cont_idx = list(range(len(cont_lengths)))
    cont_idx.sort(key=lambda x: cont_lengths[x])
    B = 32
    cont_blocks = [B for _ in range(len(cont_lengths) // B)]
    if len(cont_lengths) % B != 0:
      cont_blocks[-1] += len(cont_lengths) % B

    # cont_idx points to cur_grp; length is number of conts in current ctx group
    # cur_grp points to whole_ei; length is num_data * NUM_CHOICES

    sum_cont_block_wasted = 0
    sum_cont_block_effective = 0
    cont_block_end = 0
    for cont_block_size in cont_blocks:
      cont_block_end += cont_block_size
      cont_block_start = cont_block_end - cont_block_size
      data_idx = tuple(cur_grp[cont_idx[j]] for j in range(cont_block_start, cont_block_end))
      cache_mapping = tuple(cont_idx[j] // NUM_CHOICES for j in range(cont_block_start, cont_block_end))
      seq_len = max(len(whole_ei[data_idx[j]]['ctx']) + len(whole_ei[data_idx[j]]['cont']) - ctx_min_len - 1 for j in range(cont_block_size))
      batches.append(Batch(data_idx, use_cache = True, gen_cache = False, seq_len = seq_len, cache_mapping=cache_mapping, cache_len=ctx_min_len, cache_dep=cache_dep))
      # cont schedule validation
      lengths = tuple(len(whole_ei[data_idx[j]]['ctx']) + len(whole_ei[data_idx[j]]['cont']) for j in range(cont_block_size))
      max_S = max(lengths)
      cont_block_wasted = sum(max_S - lengths[j] for j in range(cont_block_size))
      cont_block_effective = sum(lengths[j] for j in range(cont_block_size))
      sum_cont_block_wasted += cont_block_wasted
      sum_cont_block_effective += cont_block_effective
      #print(f'cont_block_wasted: {cont_block_wasted}, cont_block_effective: {cont_block_effective}')

  
  return whole_pe, whole_ei, batches