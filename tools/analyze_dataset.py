from datasets import load_dataset
from llama import Tokenizer
import re
import torch
import numpy as np
import sys

CACHE_DIR='../local_disk/datasets'
TOKENIZER_PATH='../local_disk/llama_preprocessed/tokenizer.model'

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
  #idx = out_example['query'].index(': ') + 2
  #query_inst = out_example['query'][:idx]
  #query_ctx = out_example['query'][idx:]
  #assert query_inst == example['activity_label'] + ': '
  #out_example['query_inst'] = query_inst
  #out_example['query_ctx'] = query_ctx
  return out_example

# triplet
#def encode_input(tokenizer, example):
#  def encode_pair(inst, context, continuation):
#    n_spaces = len(context) - len(context.rstrip())
#    if n_spaces > 0:
#        continuation = context[-n_spaces:] + continuation
#        context = context[:-n_spaces]
#    bos = False
#    whole_enc = tokenizer.encode(inst + context + continuation, bos=bos, eos=False)
#    inst_ctx_enc = tokenizer.encode(inst + context, bos=bos, eos=False)
#    inst_enc = tokenizer.encode(inst, bos=bos, eos=False)
#    inst_enc_len = len(inst_enc)
#    inst_ctx_enc_len = len(inst_ctx_enc)
#    ctx_enc = whole_enc[inst_enc_len:inst_ctx_enc_len]
#    cont_enc = whole_enc[inst_ctx_enc_len:]
#    return inst_enc, ctx_enc, cont_enc 
#  reqs = [(example['query_inst'], example['query_ctx'], ' {}'.format(choice)) for choice in example['choices']]
#  new_reqs = []
#  for inst, ctx, cont in reqs:
#    inst_enc, ctx_enc, cont_enc = encode_pair(inst, ctx, cont)
#    new_reqs.append((inst_enc, ctx_enc, cont_enc))
#  return new_reqs

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

def save_encoded():
  dataset = load_dataset("hellaswag", cache_dir=CACHE_DIR, split='validation')
  tokenizer = Tokenizer(model_path=TOKENIZER_PATH)

  eis = []
  for i, data in enumerate(dataset):
    if i % 100 == 0:
      print(f'Processing {i}...')
    pe = process_example(data)
    eis.extend(encode_input(tokenizer, pe))

  full_batch = [ei[2] + ei[3] for ei in eis]
  max_len = max([len(enc) for enc in full_batch])
  print(f'Max length: {max_len}')
  for i in range(len(full_batch)):
    full_batch[i].extend([-1] * (max_len - len(full_batch[i])))
  np.save('full_batch.npy', full_batch)

def pop_padding(items):
  x = items.copy()
  while x[-1] == -1:
    x.pop()
  return x

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
  print(f'total penalty: {D[N - 1]}')
  s = 0
  for i, block in enumerate(blocks):
    s += block
    cur_s = lengths[idx[s - block]]
    print(f'block {i}: {block} x {cur_s} = {block * cur_s}')
  print(blocks)
  print(sum(blocks))

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
  print(f'total penalty: {D[N - 1]}')
  s = 0
  for i, block in enumerate(blocks):
    s += block
    cur_s = lengths[idx[s - 1]]
    print(f'block {i}: {block} x {cur_s} = {block * cur_s}')
  print(blocks)
  print(sum(blocks))

  return idx, blocks

dataset = load_dataset("hellaswag", cache_dir=CACHE_DIR, split='validation')
tokenizer = Tokenizer(model_path=TOKENIZER_PATH)

eis = []
for i, data in enumerate(dataset):
  if i == 1000:
    break
  if i % 100 == 0:
    print(f'Processing {i}...')
  pe = process_example(data)
  eis.extend(encode_input(tokenizer, pe))

NC = 4
THR = 1024
ctx_lengths = [len(eis[i][2]) for i in range(0, len(eis), NC)]
ctx_idx, ctx_blocks = schedule_min(ctx_lengths, THR)

s = 0
for i, ctx_block in enumerate(ctx_blocks):
  s += ctx_block
  ctx_min_len = len(eis[ctx_idx[s - ctx_block] * NC][2])
  cur_grp = []
  for j in range(s - ctx_block, s):
    cur_grp.extend([ctx_idx[j] * NC + k for k in range(NC)])
  cont_lengths = [len(eis[j][2]) + len(eis[j][3]) - ctx_min_len for j in cur_grp]
  cont_idx, cont_blocks = schedule_max(cont_lengths, THR)









#save_encoded()

#full_batch = np.load('full_batch.npy').tolist()
#full_batch = [pop_padding(x) for x in full_batch]
##full_batch = sorted(full_batch)
#full_batch = sorted(full_batch, key=lambda x : len(x))
#for i in range(len(full_batch)):
#  for j in range(len(full_batch[i])):
#    if full_batch[i][j] == -1:
#      break
#    print(f'{full_batch[i][j]:5d} ', end='')
#  print()

# sort list of list by lexicographical order
