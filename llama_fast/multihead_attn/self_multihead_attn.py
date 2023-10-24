from typing import Optional

import torch
from torch import nn

from ..model import ModelArgs, RotaryEmbed
from .self_multihead_attn_func import self_attn_func
from .fast_self_multihead_attn_func import fast_self_attn_func


# https://github.com/NVIDIA/apex/blob/master/apex/contrib/multihead_attn/self_multihead_attn.py
class FastMultiheadAttention(nn.Module):
  def __init__(
      self,
      args: ModelArgs,
      impl="fast",):
    
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

    if impl == "fast":
      self.attn_func = fast_self_attn_func
    elif impl == "default":
      self.attn_func = self_attn_func
    else:
      assert False, "Unsupported impl: {} !".format(impl)

  def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
    input_weights = (
      torch.cat(
        [
          self.wq.weight.view(self.n_heads, 1, self.head_dim, self.n_heads * self.head_dim),
          self.wk.weight.view(self.n_heads, 1, self.head_dim, self.n_heads * self.head_dim),
          self.wv.weight.view(self.n_heads, 1, self.head_dim, self.n_heads * self.head_dim),
        ],
        dim=1,
      )
      .reshape(3 * self.n_heads * self.head_dim, self.n_heads * self.head_dim)
      .contiguous()
    )

    if self.impl == "fast":
      outputs = self.attn_func(
        mask is not None, 
        False, 
        self.num_heads,
        x,
        input_weights,
        self.wo.weight,
        None,
        None,
        mask,
        self.mask_additive,
        0.0,
      )
  # else:
  #     outputs = self.attn_func(
  #         attn_mask is not None,
  #         is_training,
  #         self.num_heads,
  #         self.scaling,
  #         query,
  #         input_weights,
  #         self.out_proj_weight,
  #         input_bias,
  #         self.out_proj_bias,
  #         mask,
  #         self.mask_additive,
  #         self.dropout,
  #     )
  