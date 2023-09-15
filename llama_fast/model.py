# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


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
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq = bsz seqlen n_heads(52) head_dim(128)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # xq = bsz seqlen n_heads, head_dim//2 (64)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # freq_cis = seqlen, head_dim//2
    #freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[3])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

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

  def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], phase: int, cache_k = None, cache_v = None, cont2ctx = None):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    if phase == 0:
      cache_k = xk # (bsz, seqlen, n_heads, head_dim)
      cache_v = xv
      keys = xk
      values = xv
    elif phase == 1:
      keys = torch.cat((cache_k[cont2ctx], xk), dim=1) # (bsz, start_pos + seqlen, n_heads, head_dim)
      values = torch.cat((cache_v[cont2ctx], xv), dim=1)

    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    if phase == 0:
      output = F.scaled_dot_product_attention(xq, keys, values, is_causal=True) # (bsz, n_heads, seqlen, head_dim)
    if phase == 1:
      output = F.scaled_dot_product_attention(xq, keys, values, attn_mask = mask) # (bsz, n_heads, seqlen, head_dim)

    output = output.transpose(
        1, 2
    ).contiguous().view(bsz, seqlen, -1)

    if phase == 0:
      return self.wo(output), cache_k, cache_v
    elif phase == 1:
      return self.wo(output)
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

class RMSNorm(torch.nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

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
    self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
    self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

  def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], phase: int, cache_k = None, cache_v = None, cont2ctx = None):
    if phase == 0:
      h, cache_k, cache_v = self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, phase)
    if phase == 1:
      h = self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, phase, cache_k, cache_v, cont2ctx)

    h = x + h
    out = h + self.feed_forward.forward(self.ffn_norm(h))

    if phase == 0:
      return out, cache_k, cache_v
    if phase == 1:
      return out

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
  def forward(self, h: torch.Tensor, start_pos: int, phase: int, cache_k_list = None, cache_v_list = None, cont2ctx = None):
    _bsz, seqlen, _ = h.shape
    self.freqs_cis = self.freqs_cis.to(h.device)
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

    mask = None
    if phase == 1:
      mask = torch.ones(seqlen, start_pos + seqlen, dtype=torch.bool, device=h.device).tril(diagonal=start_pos)

    if phase == 0:
      cache_k_list = []
      cache_v_list = []
      for i, layer in enumerate(self.layers):
        h, cache_k, cache_v = layer(h, start_pos, freqs_cis, mask, phase)
        cache_k_list.append(cache_k)
        cache_v_list.append(cache_v)
      return h, cache_k_list, cache_v_list
    if phase == 1:
      for i, layer in enumerate(self.layers):
        h = layer(h, start_pos, freqs_cis, mask, phase, cache_k_list[i], cache_v_list[i], cont2ctx)
      return h, None, None
  
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
    self.norm = RMSNorm(params.dim, eps=params.norm_eps)
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
