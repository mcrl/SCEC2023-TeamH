import torch

LOAD_PATH = '../local_disk/llama_preprocessed/30B_cpu.pth'
SAVE_PATH = '../local_disk/llama_preprocessed'

ckpt = torch.load(LOAD_PATH, map_location='cpu')

L = 60
SPLIT = 4

for i in range(SPLIT):
  local_ckpt = {}
  if i == 0:
    local_ckpt['tok_embeddings.weight'] = ckpt['tok_embeddings.weight'].clone().detach()
  if i == 3:
    local_ckpt['norm.weight'] = ckpt['norm.weight'].clone().detach()
    local_ckpt['output.weight'] = ckpt['output.weight'].clone().detach()
  for j in range(i * (L // SPLIT), (i + 1) * (L // SPLIT)):
    local_ckpt[f'layers.{j}.attention.wq.weight'] = ckpt[f'layers.{j}.attention.wq.weight'].clone().detach()
    local_ckpt[f'layers.{j}.attention.wk.weight'] = ckpt[f'layers.{j}.attention.wk.weight'].clone().detach()
    local_ckpt[f'layers.{j}.attention.wv.weight'] = ckpt[f'layers.{j}.attention.wv.weight'].clone().detach()
    local_ckpt[f'layers.{j}.attention.wo.weight'] = ckpt[f'layers.{j}.attention.wo.weight'].clone().detach()
    local_ckpt[f'layers.{j}.feed_forward.w1.weight'] = ckpt[f'layers.{j}.feed_forward.w1.weight'].clone().detach()
    local_ckpt[f'layers.{j}.feed_forward.w2.weight'] = ckpt[f'layers.{j}.feed_forward.w2.weight'].clone().detach()
    local_ckpt[f'layers.{j}.feed_forward.w3.weight'] = ckpt[f'layers.{j}.feed_forward.w3.weight'].clone().detach()
    local_ckpt[f'layers.{j}.attention_norm.weight'] = ckpt[f'layers.{j}.attention_norm.weight'].clone().detach()
    local_ckpt[f'layers.{j}.ffn_norm.weight'] = ckpt[f'layers.{j}.ffn_norm.weight'].clone().detach()
  size = 0
  for key in local_ckpt:
    numel = local_ckpt[key].numel()
    size += numel * 2
    print(key, numel)
  print(f'Expected size {size} bytes')
  torch.save(local_ckpt, f'{SAVE_PATH}/30B_cpu_{i}.pth')

