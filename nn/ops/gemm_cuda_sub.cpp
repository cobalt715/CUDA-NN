#ifndef COBALT_715_USE_CUDA

#include <cstdint>
#include "gemm.cuh"
#include "nn/dtype.hpp"
#include "nn/cuda/util.cuh"

namespace cobalt_715::nn::ops::cuda{

template<nn::mutable_dtype T>
void gemm_impl(const T alpha,

               const T *a,
               const int64_t a_row_stride,
               const int64_t a_col_stride,

               const T *b,
               const int64_t b_row_stride,
               const int64_t b_col_stride,

               const T beta,

               T *out,
               const int64_t out_row_stride,
               const int64_t out_col_stride,

               const int64_t I,
               const int64_t J,
               const int64_t K){

  nn::cuda::throw_not_enabled();
}

#define INSTANTIATE_GEMM_IMPL(T) \
  template void gemm_impl<T>(const T alpha,const T *a,const int64_t a_row_stride,const int64_t a_col_stride, \
  const T *b,const int64_t b_row_stride,const int64_t b_col_stride, \
  const T beta,T *out,const int64_t out_row_stride,const int64_t out_col_stride, \
  const int64_t I,const int64_t J,const int64_t K);

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_GEMM_IMPL)

}//namespace cobalt_715::nn::ops::cuda

#endif