import torch
import argparse
import os

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory that contains consolidated.{00~03}.pth')
  args = parser.parse_args()

  MP = 4
  PP = 4
  V = 32000
  H = 6656
  H2 = 17920
  L = 60

  ckpt = []
  for i in range(MP):
    fn = os.path.join(args.data_dir, f'consolidated.{i:02d}.pth')
    print(f'Loading ckpt {fn}...')
    ckpt.append(torch.load(fn, map_location='cpu'))

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

  # Free memory by delete original tensors
  ckpt = merged
  del merged

  for i in range(PP):
    local_ckpt = {}
    if i == 0:
      local_ckpt['tok_embeddings.weight'] = ckpt['tok_embeddings.weight'].clone().detach()
    if i == 3:
      local_ckpt['norm.weight'] = ckpt['norm.weight'].clone().detach()
      local_ckpt['output.weight'] = ckpt['output.weight'].clone().detach()
    for j in range(i * (L // PP), (i + 1) * (L // PP)):
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
    fn = os.path.join(args.data_dir, f'30B_cpu_{i}.pth')
    print(f'Saving {fn} with tensors of total size {size} bytes')
    torch.save(local_ckpt, fn)

if __name__ == '__main__':
  main()