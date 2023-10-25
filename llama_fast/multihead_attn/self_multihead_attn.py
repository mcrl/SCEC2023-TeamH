from typing import Optional
import importlib

import torch
from torch import nn

from ..model import ModelArgs, RotaryEmbed
from .fast_self_multihead_attn_func import fast_self_attn_func

global fast_multihead_attn
fast_multihead_attn = None

# https://github.com/NVIDIA/apex/blob/master/apex/contrib/multihead_attn/self_multihead_attn.py
class FastMultiheadAttention(nn.Module):
  def __init__(
      self,
      args: ModelArgs,):
    
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

    global fast_multihead_attn
    fast_multihead_attn = importlib.import_module("fast_multihead_attn")

  def forward(self, x: torch.Tensor, residual_x, start_pos: int, freqs_cis: torch.Tensor, 
              mask: Optional[torch.Tensor], use_cache: bool, gen_cache: bool, 
              cache_k = None, cache_v = None, cont2ctx = None, last_token_only = False):
    
    return fast_multihead_attn.mha(self.n_heads, self.head_dim, x, residual_x, start_pos, 
                                   self.wq, self.wk, self.wv, freqs_cis, 
                                   mask, use_cache, gen_cache, 
                                   cache_k, cache_v, cont2ctx, last_token_only)
  