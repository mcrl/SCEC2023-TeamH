#include <vector>

#include <cuda_fp16.h>
#include <torch/extension.h>


#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

namespace multihead_attn {
namespace self_bias_additive_mask {
namespace cublas_gemmex {

std::vector<torch::Tensor> fwd_cuda(bool use_time_mask, bool is_training,
                                    int heads, torch::Tensor const &inputs,
                                    torch::Tensor const &input_weights,
                                    torch::Tensor const &output_weights,
                                    torch::Tensor const &input_biases,
                                    torch::Tensor const &output_biases,
                                    const half *pad_mask, float dropout_prob);

std::vector<torch::Tensor>
fwd(bool use_mask, bool use_time_mask, bool is_training, int heads,
    torch::Tensor const &inputs, torch::Tensor const &input_weights,
    torch::Tensor const &output_weights, torch::Tensor const &input_biases,
    torch::Tensor const &output_biases, torch::Tensor const &pad_mask,
    float dropout_prob) {
  TORCH_CHECK(inputs.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");

  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(input_weights.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(use_mask, "no mask is not supported");

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Half,
               "Only Half is supported");
  }

  return fwd_cuda(use_time_mask, is_training, heads, inputs, input_weights,
                  output_weights, input_biases, output_biases,
                  use_mask ? static_cast<const half *>(pad_mask.data_ptr())
                           : nullptr,
                  dropout_prob);
}

} // end namespace cublas_gemmex
} // namespace self_bias_additive_mask
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("self_attn_bias_additive_mask_forward", &multihead_attn::self_bias_additive_mask::cublas_gemmex::fwd,
        "Self Multihead Attention with Bias -- Forward.");
}

#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT