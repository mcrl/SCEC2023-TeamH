#include <iostream>
#include <math.h>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dropout.cuh"
#include "softmax.cuh"
#include "strided_batched_gemm.cuh"

namespace multihead_attn {
namespace self_bias_additive_mask {
namespace cublas_gemmex {

std::vector<torch::Tensor>
fwd_cuda_teamh(
  const int n_heads,
  const int head_dim,
  torch::Tensor const &x,
  torch::Tensor const &residual_x,
  torch::Tensor const &wq,
  torch::Tensor const &wk,
  torch::Tensor const &wv,
  const int start_pos,
  torch::Tensor const &freqs_cis,
  torch::Tensor const &mask,
  bool use_cache,
  bool gen_cache,
  torch::Tensor const &cache_k,
  torch::Tensor const &cache_v,
  auto const &cont2ctx,
  bool last_token_only, ) {

  const int bsz = x.size(0);
  const int seqlen = x.size(1);

  // There is no reason to use more than one stream as every kernel is
  // sequentially dependent
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  // Set cublas math mode
  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  // Inference mode
  auto act_options = inputs.options().requires_grad(false);

  torch::Tensor xq;
  torch::Tensor xk = torch::empty({ bsz, seq_len, n_heads * head_dim }, act_options);
  torch::Tensor xv = torch::empty({ bsz, seq_len, n_heads * head_dim }, act_options);

  // xk = wk(x)
  cublas_nn(x, wk.transpose(0, 1).contiguous(), xk, x.shape(0) * x.shape(1), wk.shape(1), wk.shape(0), 1.0f, 0.0f);
  // xv = wv(x)
  cublas_nn(x, wv.transpose(0, 1).contiguous(), xv, x.shape(0) * x.shape(1), wv.shape(1), wv.shape(0), 1.0f, 0.0f);
  // xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
  xk = xk.view({ bsz, seqlen, n_heads, head_dim });
  // xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
  xv = xv.view({ bsz, seqlen, n_heads, head_dim });

  if (last_token_only) {
    // xq = self.wq(x[:, -1 : , : ])
    // xq = xq.view(bsz, 1, self.n_heads, self.head_dim)
    xq = torch::empty({ bsz, 1, n_heads * head_dim }, act_options);
    cublas_nn(x.index({ "...", Slice(-1, None, None).contiguous(), "..." }), wq.transpose(0, 1).contiguous(), xq, wq.shape(1), wq.shape(0), 1.0f, 0.0f);
    xq = xq.view({ bsz, 1, n_heads, head_dim });
  } else {
    // xq = self.wq(x)
    // xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
    xq = torch::empty({ bsz, seqlen, n_heads * head_dim }, act_options);
    cublas_nn(x, wq.transpose(0, 1).contiguous(), xq, x.shape(0) * x.shape(1), wq.shape(1), wq.shape(0), 1.0f, 0.0f);
    xq = xq.view({ bsz, seqlen, n_heads, head_dim });
  }

  if (last_token_only) {
    // xq = self.emb(xq, freqs_cis[-1:])
    xq = rotary_emb(xq, freqs_cis.index({ Slice(-1, None, None).contiguous() }));
  } else {
    // xq = self.emb(xq, freqs_cis)
    xq = rotary_emb(xq, freqs_cis);
  }
  // xk = self.emb(xk, freqs_cis)
  xk = rotary_emb(xk, freqs_cis);

  if (use_cache) {
    // xk = torch.cat([cache_k, xk], dim = 1)
    xk = torch::cat({ cache_k[cont2ctx], xk }, 1); // keys
    // xv = torch.cat([cache_v, xv], dim = 1)
    xv = torch::cat({ cache_v[cont2ctx], xv }, 1); // values
  }

  torch::Tensor bmm1 = torch::empty({ bsz, seq_len, xk.shape(1), head_dim }, act_options);
  torch::Tensor softmax1 = torch::empty({ bsz, seq_len, xk.shape(1), head_dim }, act_options);
  torch::Tensor bmm2 =
  torch::Tensor output;

  // Scaled dot product attention
  const half scale = 1.0 / sqrt(static_cast<half>(head_dim));
  cublas_nn(xq, xk.transpose(1, 2).contiguous(), bmm1, xq.shape(0) * xq.shape(1), xk.shape(1), xk.shape(0), scale, 0.0f); // ??

  // Masked Softmax
  [[maybe_unused]] bool softmax_success = false;
  softmax_success = dispatch_additive_masked_softmax<half, half, half>(
    reinterpret_cast<half *>(softmax1),
    reinterpret_cast<const half *>(bmm1), 
    mask, xk.shape(1),
    pad_batch_stride, 
    bsz * seqlen,
    pad_batch_stride);
    
  // MM2
  
  // Output Linear
  if (residual_x == None) {
    output = torch::empty({ bsz, seq_len, n_heads * head_dim }, act_options);
  } else {
    if (last_token_only) {
      output = residual_x.index({ "...", Slice(-1, None, None).contiguous(), "..."});
    } else {
      output = residual_x;
    }
  }
  cublas_nn(bmm2, wo.transpose(0, 1).contiguous(), output, bmm2.shape(0) * bmm2.shape(1), wo.shape(1), wo.shape(0), 1.0f, 0.0f);

  return { output, xk, xv };
}

void cublas_nn(torch::Tensor& A, torch::Tensor& B, torch::Tensor& C, int M, int N, int K, float _alpha, float _beta) {
  __half alpha = __float2half(_alpha), beta = __float2half(_beta);
  int lda = K, ldb = N, ldc = N;
  // A = M by K
  // B = N by K
  // C = M by N

  // should do C^T = B^T (transposed) * A^T (normal)

  CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           (const __half*)B.data_ptr(), ldb,
                           (const __half*)A.data_ptr(), lda, &beta,
                           (__half*)C.data_ptr(), ldc)); 
}
} // end namespace cublas_gemmex
} // namespace self_bias_additive_mask
} // end namespace multihead_attn