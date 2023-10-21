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

  def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], use_cache: bool, gen_cache: bool, cache_k = None, cache_v = None, cont2ctx = None):
    #torch.cuda.nvtx.range_push(f'Attention')

    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

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

    if not use_cache:
      output = F.scaled_dot_product_attention(xq, keys, values, is_causal=True) # (bsz, n_heads, seqlen, head_dim)
    if use_cache:
      output = F.scaled_dot_product_attention(xq, keys, values, attn_mask = mask) # (bsz, n_heads, seqlen, head_dim)

    #torch.cuda.nvtx.range_pop()

    output = output.transpose(
        1, 2
    ).contiguous().view(bsz, seqlen, -1)

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

# class RMSNorm(torch.nn.Module):
#   def __init__(self, dim: int, eps: float = 1e-6):
#     super().__init__()
#     self.eps = eps
#     self.weight = nn.Parameter(torch.ones(dim))

#   def _norm(self, x):
#     return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

#   def forward(self, x):
#     output = self._norm(x.float()).type_as(x)
#     return output * self.weight

global fused_layer_norm_cuda
fused_layer_norm_cuda = None

def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())

# Reference implementation from Huggingface
def manual_rms_norm(input, normalized_shape, weight, eps):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape)-1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)

    if weight is None:
        return input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)

    return weight * input

def fused_rms_norm_affine(input, weight, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, weight, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return FusedRMSNormAffineFunction.apply(*args)


def fused_rms_norm(input, normalized_shape, eps=1e-6):
    args = _cast_if_autocast_enabled(input, normalized_shape, eps)
    with torch.cuda.amp.autocast(enabled=False):
        return FusedRMSNormFunction.apply(*args)

class FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_forward(input_, ctx.normalized_shape, ctx.eps)
        ctx.save_for_backward(input_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, invvar = ctx.saved_tensors
        grad_input = None
        grad_input = fused_layer_norm_cuda.rms_backward(
            grad_output.contiguous(), invvar, input_, ctx.normalized_shape, ctx.eps
        )
        return grad_input, None, None

class FusedRMSNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_forward_affine(
            input_, ctx.normalized_shape, weight_, ctx.eps)
        ctx.save_for_backward(input_, weight_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, invvar = ctx.saved_tensors
        grad_input = grad_weight = None
        grad_input, grad_weight = fused_layer_norm_cuda.rms_backward_affine(
           grad_output.contiguous(), invvar, input_, ctx.normalized_shape, weight_, ctx.eps
        )
        return grad_input, grad_weight, None, None
    
class FusedRMSNorm(torch.nn.Module):
    r"""Applies RMS Normalization over a mini-batch of inputs

    Currently only runs on cuda() tensors.

    .. math::
        y = \frac{x}{\mathrm{RMS}[x]} * \gamma

    The root-mean-square is calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` is a learnable affine transform parameter of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    `epsilon` is added to the mean-square, then the root of the sum is taken.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, RMS Normalization applies per-element scale
        with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized}\_\text{shape}[0] \times \text{normalized}\_\text{shape}[1]
                    \times \ldots \times \text{normalized}\_\text{shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = apex.normalization.FusedRMSNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = apex.normalization.FusedRMSNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = apex.normalization.FusedRMSNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = apex.normalization.FusedRMSNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Root Mean Square Layer Normalization`: https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(*normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input):
        #torch.cuda.nvtx.range_push(f'RMSNorm')
        if torch.jit.is_tracing() or torch.jit.is_scripting() or not input.is_cuda:
            #torch.cuda.nvtx.range_pop()
            return manual_rms_norm(input, self.normalized_shape, self.weight, self.eps)

        if self.elementwise_affine:
            #torch.cuda.nvtx.range_pop()
            return fused_rms_norm_affine(input, self.weight, self.normalized_shape, self.eps)
        else:
            #torch.cuda.nvtx.range_pop()
            return fused_rms_norm(input, self.normalized_shape, self.eps)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)


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

  def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], use_cache: bool, gen_cache: bool, cache_k = None, cache_v = None, cont2ctx = None):
    #torch.cuda.nvtx.range_push(f'TfBlock {self.layer_id}')

    h, new_cache_k, new_cache_v = self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, use_cache, gen_cache, cache_k, cache_v, cont2ctx)

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
  def forward(self, h: torch.Tensor, start_pos: int, use_cache: bool, gen_cache: bool, cache_k_list = None, cache_v_list = None, cont2ctx = None):
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
      h, new_cache_k, new_cache_v = layer(h, start_pos, freqs_cis, mask, use_cache, gen_cache, cache_k_list[i], cache_v_list[i], cont2ctx)
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
