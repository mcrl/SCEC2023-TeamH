import torch

LOAD_PATH = 'local_disk/llama_pretrained/30B'
SAVE_PATH = 'local_disk/llama_preprocessed/30B_cpu.pth'
MP = 4
V = 32000
H = 6656
H2 = 17920
L = 60

ckpt = []
for i in range(MP):
  print(f'Loading ckpt {i}...')
  ckpt.append(torch.load(f'{LOAD_PATH}/consolidated.{i:02d}.pth', map_location='cpu'))

merged = {}

def col_merge(key, expected_shape, expected_dtype):
  ts = [ckpt[i][key] for i in range(MP)]
  t = torch.cat(ts, dim=1)
  assert t.shape == expected_shape and t.dtype == expected_dtype
  merged[key] = t.clone().detach()

def row_merge(key, expected_shape, expected_dtype):
  ts = [ckpt[i][key] for i in range(MP)]
  t = torch.cat(ts, dim=0)
  assert t.shape == expected_shape and t.dtype == expected_dtype
  merged[key] = t.clone().detach()

def no_merge(key, expected_shape, expected_dtype):
  ts = [ckpt[i][key] for i in range(MP)]
  for i in range(1, MP):
    assert torch.equal(ts[0], ts[i])
  t = ts[0]
  assert t.shape == expected_shape and t.dtype == expected_dtype
  merged[key] = t.clone().detach()

print(f'Merging misc. tensors...')
col_merge('tok_embeddings.weight', (V, H), torch.float16)
no_merge('norm.weight', (H,), torch.float16)
row_merge('output.weight', (V, H), torch.float16)

for i in range(L):
  print(f'Merging layer {i}...')
  row_merge(f'layers.{i}.attention.wq.weight', (H, H), torch.float16)
  row_merge(f'layers.{i}.attention.wk.weight', (H, H), torch.float16)
  row_merge(f'layers.{i}.attention.wv.weight', (H, H), torch.float16)
  col_merge(f'layers.{i}.attention.wo.weight', (H, H), torch.float16)
  row_merge(f'layers.{i}.feed_forward.w1.weight', (H2, H), torch.float16)
  col_merge(f'layers.{i}.feed_forward.w2.weight', (H, H2), torch.float16)
  row_merge(f'layers.{i}.feed_forward.w3.weight', (H2, H), torch.float16)
  no_merge(f'layers.{i}.attention_norm.weight', (H,), torch.float16)
  no_merge(f'layers.{i}.ffn_norm.weight', (H,), torch.float16)

total_size = 0
for key in merged:
  t = merged[key]
  expected_size = t.numel() * 2
  total_size += expected_size 

print(f'Expected total_size: {total_size} bytes')

print(f'Final save to {SAVE_PATH}...')
torch.save(merged, SAVE_PATH)