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
namespace fused_softmax {
namespace additive_mask_softmax_dropout {

std::vector<torch::Tensor> fwd_cuda(bool is_training, int heads,
                                    torch::Tensor const &input,
                                    const half *pad_mask, float dropout_prob);

std::vector<torch::Tensor> fwd(bool use_mask, bool is_training, int heads,
                               torch::Tensor const &input,
                               torch::Tensor const &pad_mask,
                               float dropout_prob) {
  TORCH_CHECK(input.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Half,
               "Only BYTE is supported");
  }

  return fwd_cuda(is_training, heads, input,
                  use_mask ? static_cast<const half *>(pad_mask.data_ptr())
                           : nullptr,
                  dropout_prob);
}

} // namespace additive_mask_softmax_dropout

namespace mask_softmax_dropout {

std::vector<torch::Tensor> fwd_cuda(bool is_training, int heads,
                                    torch::Tensor const &input,
                                    const uint8_t *pad_mask,
                                    float dropout_prob);

std::vector<torch::Tensor> fwd(bool use_mask, bool is_training, int heads,
                               torch::Tensor const &input,
                               torch::Tensor const &pad_mask,
                               float dropout_prob) {
  TORCH_CHECK(input.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Byte,
               "Only BYTE is supported");
  }

  return fwd_cuda(is_training, heads, input,
                  use_mask ? static_cast<const uint8_t *>(pad_mask.data_ptr())
                           : nullptr,
                  dropout_prob);
}

} // end namespace mask_softmax_dropout
} // end namespace fused_softmax

namespace encdec {
namespace cublas_gemmex {

std::vector<torch::Tensor> fwd_cuda(bool use_time_mask, bool is_training,
                                    int heads, torch::Tensor const &inputs_q,
                                    torch::Tensor const &inputs_kv,
                                    torch::Tensor const &input_weights_q,
                                    torch::Tensor const &input_weights_kv,
                                    torch::Tensor const &output_weights,
                                    const uint8_t *pad_mask,
                                    float dropout_prob);

std::vector<torch::Tensor>
fwd(bool use_mask, bool use_time_mask, bool is_training, int heads,
    torch::Tensor const &inputs_q, torch::Tensor const &inputs_kv,
    torch::Tensor const &input_weights_q, torch::Tensor const &input_weights_kv,
    torch::Tensor const &output_weights, torch::Tensor const &pad_mask,
    float dropout_prob) {
  TORCH_CHECK(inputs_q.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(inputs_kv.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_weights_q.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(input_weights_kv.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");

  TORCH_CHECK(inputs_q.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(inputs_kv.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(input_weights_q.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(input_weights_kv.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Byte,
               "Only BYTE is supported");
  }

  return fwd_cuda(use_time_mask, is_training, heads, inputs_q, inputs_kv,
                  input_weights_q, input_weights_kv, output_weights,
                  use_mask ? static_cast<const uint8_t *>(pad_mask.data_ptr())
                           : nullptr,
                  dropout_prob);
}

} // end namespace cublas_gemmex
} // end namespace encdec

namespace encdec_norm_add {
namespace cublas_gemmex {

std::vector<torch::Tensor> fwd_cuda(bool use_time_mask, bool is_training,
                                    int heads, torch::Tensor const &inputs_q,
                                    torch::Tensor const &inputs_kv,
                                    torch::Tensor const &lyr_nrm_gamma_weights,
                                    torch::Tensor const &lyr_nrm_beta_weights,
                                    torch::Tensor const &input_weights_q,
                                    torch::Tensor const &input_weights_kv,
                                    torch::Tensor const &output_weights,
                                    const uint8_t *pad_mask,
                                    float dropout_prob);

std::vector<torch::Tensor>
fwd(bool use_mask, bool use_time_mask, bool is_training, int heads,
    torch::Tensor const &inputs_q, torch::Tensor const &inputs_kv,
    torch::Tensor const &lyr_nrm_gamma_weights,
    torch::Tensor const &lyr_nrm_beta_weights,
    torch::Tensor const &input_weights_q, torch::Tensor const &input_weights_kv,
    torch::Tensor const &output_weights, torch::Tensor const &pad_mask,
    float dropout_prob) {
  TORCH_CHECK(inputs_q.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(inputs_kv.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(lyr_nrm_gamma_weights.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(lyr_nrm_beta_weights.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(input_weights_q.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(input_weights_kv.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");

  TORCH_CHECK(inputs_q.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(inputs_kv.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_gamma_weights.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_beta_weights.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(input_weights_q.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(input_weights_kv.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Byte,
               "Only BYTE is supported");
  }

  return fwd_cuda(use_time_mask, is_training, heads, inputs_q, inputs_kv,
                  lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q,
                  input_weights_kv, output_weights,
                  use_mask ? static_cast<const uint8_t *>(pad_mask.data_ptr())
                           : nullptr,
                  dropout_prob);
}

} // end namespace cublas_gemmex
} // end namespace encdec_norm_add

namespace self {
namespace cublas_gemmex {

std::vector<torch::Tensor> fwd_cuda(bool use_time_mask, bool is_training,
                                    int heads, torch::Tensor const &inputs,
                                    torch::Tensor const &input_weights,
                                    torch::Tensor const &output_weights,
                                    const uint8_t *pad_mask,
                                    float dropout_prob);

std::vector<torch::Tensor>
fwd(bool use_mask, bool use_time_mask, bool is_training, int heads,
    torch::Tensor const &inputs, torch::Tensor const &input_weights,
    torch::Tensor const &output_weights, torch::Tensor const &pad_mask,
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

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Byte,
               "Only BYTE is supported");
  }

  return fwd_cuda(
      use_time_mask, is_training, heads, inputs, input_weights, output_weights,
      use_mask ? static_cast<const uint8_t *>(pad_mask.data_ptr()) : nullptr,
      dropout_prob);
}

} // end namespace cublas_gemmex
} // end namespace self
namespace self_bias {
namespace cublas_gemmex {

std::vector<torch::Tensor>
fwd_cuda(bool use_time_mask, bool is_training, int heads,
         torch::Tensor const &inputs, torch::Tensor const &input_weights,
         torch::Tensor const &output_weights, torch::Tensor const &input_biases,
         torch::Tensor const &output_biases, const uint8_t *pad_mask,
         float dropout_prob);

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

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Byte,
               "Only BYTE is supported");
  }

  return fwd_cuda(use_time_mask, is_training, heads, inputs, input_weights,
                  output_weights, input_biases, output_biases,
                  use_mask ? static_cast<const uint8_t *>(pad_mask.data_ptr())
                           : nullptr,
                  dropout_prob);
}

} // end namespace cublas_gemmex
} // namespace self_bias
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

namespace self_norm_add {
namespace cublas_gemmex {

std::vector<torch::Tensor> fwd_cuda(bool use_time_mask, bool is_training,
                                    int heads, torch::Tensor const &inputs,
                                    torch::Tensor const &lyr_nrm_gamma_weights,
                                    torch::Tensor const &lyr_nrm_beta_weights,
                                    torch::Tensor const &input_weights,
                                    torch::Tensor const &output_weights,
                                    const uint8_t *pad_mask,
                                    float dropout_prob);

std::vector<torch::Tensor>
fwd(bool use_mask, bool use_time_mask, bool is_training, int heads,
    torch::Tensor const &inputs, torch::Tensor const &lyr_nrm_gamma_weights,
    torch::Tensor const &lyr_nrm_beta_weights,
    torch::Tensor const &input_weights, torch::Tensor const &output_weights,
    torch::Tensor const &pad_mask, float dropout_prob) {
  TORCH_CHECK(inputs.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(lyr_nrm_gamma_weights.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(lyr_nrm_beta_weights.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(input_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");

  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_gamma_weights.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_beta_weights.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(input_weights.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half,
             "Only HALF is supported");

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Byte,
               "Only BYTE is supported");
  }

  return fwd_cuda(
      use_time_mask, is_training, heads, inputs, lyr_nrm_gamma_weights,
      lyr_nrm_beta_weights, input_weights, output_weights,
      use_mask ? static_cast<const uint8_t *>(pad_mask.data_ptr()) : nullptr,
      dropout_prob);
}

} // end namespace cublas_gemmex
} // end namespace self_norm_add
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("additive_mask_softmax_dropout_forward",
        &multihead_attn::fused_softmax::additive_mask_softmax_dropout::fwd,
        "Self Multihead Attention masked softmax dropout -- Forward.");
  m.def("mask_softmax_dropout_forward", &multihead_attn::fused_softmax::mask_softmax_dropout::fwd,
        "Self Multihead Attention masked softmax dropout -- Forward.");
  m.def("encdec_multihead_attn_forward", &multihead_attn::encdec::cublas_gemmex::fwd,
        "Encdec Multihead Attention Forward.");
  m.def("encdec_multihead_attn_norm_add_forward", &multihead_attn::encdec_norm_add::cublas_gemmex::fwd,
        "Encdec Multihead Attention Plus Layer Norm and Residual Add Forward.");
  m.def("self_attn_forward", &multihead_attn::self::cublas_gemmex::fwd,
        "Self Multihead Attention Forward.");
  m.def("self_attn_bias_forward", &multihead_attn::self_bias::cublas_gemmex::fwd,
        "Self Multihead Attention with Bias -- Forward.");
  m.def("self_attn_bias_additive_mask_forward", &multihead_attn::self_bias_additive_mask::cublas_gemmex::fwd,
        "Self Multihead Attention with Bias -- Forward.");
  m.def("self_attn_norm_add_forward", &multihead_attn::self_norm_add::cublas_gemmex::fwd,
        "Self Multihead Attention Plus Layer Norm and Residual Add Forward.");
}

#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT