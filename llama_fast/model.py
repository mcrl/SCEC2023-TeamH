# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import importlib
import numbers

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    freqs_cis = torch.view_as_real(freqs_cis).flatten(1)
    return freqs_cis

class Attention(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()

    self.n_heads = args.n_heads
    self.head_dim = args.dim // args.n_heads

    self.wq = nn.Linear(
        args.dim,
        args.n_heads * self.head_dim,
        bias=False,
    )
    self.wk = nn.Linear(
        args.dim,
        args.n_heads * self.head_dim,
        bias=False,
    )
    self.wv = nn.Linear(
        args.dim,
        args.n_heads * self.head_dim,
        bias=False,
    )
    self.wo = nn.Linear(
        args.n_heads * self.head_dim,
        args.dim,
        bias=False,
    )

    self.emb = RotaryEmbed()

  def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], use_cache: bool, gen_cache: bool, cache_k = None, cache_v = None, cont2ctx = None, last_token_only = False):
    #torch.cuda.nvtx.range_push(f'Attention')

    bsz, seqlen, _ = x.shape
    xk, xv = self.wk(x), self.wv(x)
    xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

    if last_token_only:
      xq = self.wq(x[:, -1:, :])
      xq = xq.view(bsz, 1, self.n_heads, self.head_dim)
    else:
      xq = self.wq(x)
      xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)

    if last_token_only:
      xq = self.emb(xq, freqs_cis[-1:])
    else:
      xq = self.emb(xq, freqs_cis)
    xk = self.emb(xk, freqs_cis)

    if use_cache:
      keys = torch.cat((cache_k[cont2ctx], xk), dim=1) # (bsz, start_pos + seqlen, n_heads, head_dim)
      values = torch.cat((cache_v[cont2ctx], xv), dim=1)
    else:
      keys = xk
      values = xv

    new_cache_k = None
    new_cache_v = None
    if gen_cache:
      cache_k.copy_(keys)
      cache_v.copy_(values)

    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    #torch.cuda.nvtx.range_push(f'scaled_dot_Attention')

    if last_token_only:
      output = F.scaled_dot_product_attention(xq, keys, values)
      output = output.transpose(1, 2).contiguous().view(bsz, 1, -1)
    else:
      if not use_cache:
        output = F.scaled_dot_product_attention(xq, keys, values, is_causal=True) # (bsz, n_heads, seqlen, head_dim)
      if use_cache:
        output = F.scaled_dot_product_attention(xq, keys, values, attn_mask = mask) # (bsz, n_heads, seqlen, head_dim)
      output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

    #torch.cuda.nvtx.range_pop()

    #torch.cuda.nvtx.range_pop()

    return self.wo(output), new_cache_k, new_cache_v

class FeedForward(nn.Module):
  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      multiple_of: int,
  ):
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    self.w1 = nn.Linear(
        dim, hidden_dim, bias=False
    )
    self.w2 = nn.Linear(
        hidden_dim, dim, bias=False
    )
    self.w3 = nn.Linear(
        dim, hidden_dim, bias=False
    )

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))

global fused_layer_norm_cuda
fused_layer_norm_cuda = None
    
class FusedRMSNorm(torch.nn.Module):
  def __init__(self, normalized_shape, eps=1e-6):
    super().__init__()

    global fused_layer_norm_cuda
    fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)
    self.normalized_shape = torch.Size(normalized_shape)
    self.eps = eps
    self.weight = Parameter(torch.empty(*normalized_shape))
    self.reset_parameters()

  def reset_parameters(self):
    init.ones_(self.weight)

  def forward(self, input):
    input_ = input.contiguous()
    weight_ = self.weight.contiguous()
    output, invvar = fused_layer_norm_cuda.rms_forward_affine(
        input_, self.normalized_shape, weight_, self.eps)
    return output

class RotaryEmbed(torch.nn.Module):
  def __init__(self):
    super().__init__()

    global fused_layer_norm_cuda
    fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

  def forward(self, input, freqs):
    output = fused_layer_norm_cuda.rotary_emb(input, freqs)
    return output

class TransformerBlock(nn.Module):
  def __init__(self, layer_id: int, args: ModelArgs):
    super().__init__()
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads
    self.attention = Attention(args)
    self.feed_forward = FeedForward(
        dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
    )
    self.layer_id = layer_id
    self.attention_norm = FusedRMSNorm(args.dim, eps=args.norm_eps)
    self.ffn_norm = FusedRMSNorm(args.dim, eps=args.norm_eps)

  def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], use_cache: bool, gen_cache: bool, cache_k = None, cache_v = None, cont2ctx = None, last_token_only = False):
    #torch.cuda.nvtx.range_push(f'TfBlock {self.layer_id}')

    h, new_cache_k, new_cache_v = self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, use_cache, gen_cache, cache_k, cache_v, cont2ctx, last_token_only)

    if last_token_only:
      x = x[:, -1:, :]
    h = x + h
    out = h + self.feed_forward.forward(self.ffn_norm(h))

    #torch.cuda.nvtx.range_pop()

    return out, new_cache_k, new_cache_v

class TransformerBlocks(nn.Module):
  def __init__(self, params: ModelArgs, layer_idx, num_layers):
    super().__init__()
    self.params = params
    self.layer_idx = layer_idx
    self.num_layers = num_layers

    self.layers = torch.nn.ModuleList()
    for i in range(layer_idx, layer_idx + num_layers):
      self.layers.append(TransformerBlock(i, params))

    self.freqs_cis = precompute_freqs_cis(
        self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
    )
  
  @torch.inference_mode()
  def forward(self, h: torch.Tensor, start_pos: int, use_cache: bool, gen_cache: bool, cache_k_list = None, cache_v_list = None, cont2ctx = None, last_token_only = False):
    #torch.cuda.nvtx.range_push(f'TfBlocks {self.layer_idx} {self.num_layers}')
    _bsz, seqlen, _ = h.shape
    self.freqs_cis = self.freqs_cis.to(h.device)
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

    mask = None
    if use_cache:
      mask = torch.ones(seqlen, start_pos + seqlen, dtype=torch.bool, device=h.device).tril(diagonal=start_pos)

    new_cache_k_list = []
    new_cache_v_list = []

    for i, layer in enumerate(self.layers):
      h, new_cache_k, new_cache_v = layer(h, start_pos, freqs_cis, mask, use_cache, gen_cache, cache_k_list[i], cache_v_list[i], cont2ctx,
                                          last_token_only = last_token_only and i == self.num_layers - 1)
      new_cache_k_list.append(new_cache_k)
      new_cache_v_list.append(new_cache_v)

    #torch.cuda.nvtx.range_pop()

    return h, new_cache_k_list, new_cache_v_list
  
  def custom_load(self, full_dict):
    local_dict = {}
    for i in range(self.num_layers):
      local_dict[f'layers.{i}.attention.wq.weight'] = full_dict[f'layers.{self.layer_idx + i}.attention.wq.weight']
      local_dict[f'layers.{i}.attention.wk.weight'] = full_dict[f'layers.{self.layer_idx + i}.attention.wk.weight']
      local_dict[f'layers.{i}.attention.wv.weight'] = full_dict[f'layers.{self.layer_idx + i}.attention.wv.weight']
      local_dict[f'layers.{i}.attention.wo.weight'] = full_dict[f'layers.{self.layer_idx + i}.attention.wo.weight']
      local_dict[f'layers.{i}.feed_forward.w1.weight'] = full_dict[f'layers.{self.layer_idx + i}.feed_forward.w1.weight']
      local_dict[f'layers.{i}.feed_forward.w2.weight'] = full_dict[f'layers.{self.layer_idx + i}.feed_forward.w2.weight']
      local_dict[f'layers.{i}.feed_forward.w3.weight'] = full_dict[f'layers.{self.layer_idx + i}.feed_forward.w3.weight']
      local_dict[f'layers.{i}.attention_norm.weight'] = full_dict[f'layers.{self.layer_idx + i}.attention_norm.weight']
      local_dict[f'layers.{i}.ffn_norm.weight'] = full_dict[f'layers.{self.layer_idx + i}.ffn_norm.weight']
    self.load_state_dict(local_dict)

class PreTransformer(nn.Module):
  def __init__(self, params: ModelArgs):
    super().__init__()
    self.tok_embeddings = nn.Embedding(
        params.vocab_size, params.dim
    )

  @torch.inference_mode()
  def forward(self, tokens: torch.Tensor):
    h = self.tok_embeddings(tokens)
    return h

  def custom_load(self, full_dict):
    local_dict = {}
    local_dict[f'tok_embeddings.weight'] = full_dict[f'tok_embeddings.weight']
    self.load_state_dict(local_dict)

class PostTransformer(nn.Module):
  def __init__(self, params: ModelArgs):
    super().__init__()
    self.norm = FusedRMSNorm(params.dim, eps=params.norm_eps)
    self.output = nn.Linear(
      params.dim, params.vocab_size, bias=False
    )

  @torch.inference_mode()
  def forward(self, h: torch.Tensor):
    h = self.norm(h)
    output = self.output(h)
    return output.float()

  def custom_load(self, full_dict):
    local_dict = {}
    local_dict[f'norm.weight'] = full_dict[f'norm.weight']
    local_dict[f'output.weight'] = full_dict[f'output.weight']
    self.load_state_dict(local_dict)
