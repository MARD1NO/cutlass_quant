#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#include "cutlass/numeric_types.h"
// #include "cutlass/cutlass::half_t.h"
#include "cutlass/integer_subbyte.h"

#include "fpA_intB_gemm.h"

#include <stdio.h>

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

// // https://github.com/NVIDIA/apex/blob/master/csrc/type_shim.h
// // #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// #define DISPATCH_HALF_AND_BF16(TYPE, NAME, ...)                                \
//   switch (TYPE) {                                                              \
//   case at::ScalarType::Half: {                                                 \
//     using scalar_t = at::Half;                                                 \
//     __VA_ARGS__();                                                             \
//     break;                                                                     \
//   }                                                                            \
//   case at::ScalarType::BFloat16: {                                             \
//     using scalar_t = at::BFloat16;                                             \
//     __VA_ARGS__();                                                             \
//     break;                                                                     \
//   }                                                                            \
//   default:                                                                     \
//     AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");            \
//   }

// template
// void fastertransformer::gemm_fp16_int_bias_act<cutlass::uint4b_t>(const cutlass::half_t *A, const cutlass::uint4b_t *B,
// 			    const cutlass::half_t *weight_scales, const cutlass::half_t *bias,
// 			    cutlass::half_t *C, std::optional<std::string> activation, int m,
// 			    int n, int k, int bias_stride, char *workspace_ptr,
// 			    size_t workspace_bytes, cudaStream_t stream);

at::Tensor cutlass_quant_fwd(at::Tensor input, at::Tensor weight, at::Tensor weight_scales, int bits) {

  TORCH_CHECK(bits == 4 || bits == 8);
  int m = input.size(0);
  int k = input.size(1);
  int n = weight.size(1) * (32 / bits);

  TORCH_CHECK(input.dtype() == torch::kFloat16);
  TORCH_CHECK(weight.dtype() == torch::kInt32);
  TORCH_CHECK(weight_scales.dtype() == torch::kFloat16);
  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(weight.is_cuda());
  TORCH_CHECK(weight_scales.is_cuda());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(weight_scales.is_contiguous());
  CHECK_SHAPE(input, m, k);
  CHECK_SHAPE(weight, k, n / (32 / bits));
  CHECK_SHAPE(weight_scales, n);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};

  // create output/workspace tensor
  auto opts = input.options();
  auto out = at::empty({m, n}, opts);
  auto workspace = at::empty({1 << 22}, opts.dtype(torch::kInt8));

  if (bits == 4) {
    fastertransformer::gemm_fp16_int_bias_act(
        reinterpret_cast<cutlass::half_t *>(input.data_ptr()),
        reinterpret_cast<cutlass::uint4b_t *>(weight.data_ptr()),
        reinterpret_cast<cutlass::half_t *>(weight_scales.data_ptr()),
        nullptr,
        reinterpret_cast<cutlass::half_t *>(out.data_ptr()),
        std::nullopt,
        m,
        n,
        k,
        0,
        reinterpret_cast<char *>(workspace.data_ptr()),
        1 << 22,
        at::cuda::getCurrentCUDAStream());
  } else {
    fastertransformer::gemm_fp16_int_bias_act(
        reinterpret_cast<cutlass::half_t *>(input.data_ptr()),
        reinterpret_cast<uint8_t *>(weight.data_ptr()),
        reinterpret_cast<cutlass::half_t *>(weight_scales.data_ptr()),
        nullptr,
        reinterpret_cast<cutlass::half_t *>(out.data_ptr()),
        std::nullopt,
        m,
        n,
        k,
        0,
        reinterpret_cast<char *>(workspace.data_ptr()),
        1 << 22,
        at::cuda::getCurrentCUDAStream());
  }

  return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &cutlass_quant_fwd, "Quant linear forward");
}
