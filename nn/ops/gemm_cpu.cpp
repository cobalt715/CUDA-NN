#include <cstdint>
#include "gemm.cuh"
#include "nn/dtype.hpp"

namespace cobalt_715::nn::ops::cpu{

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

  for(int64_t i = 0;i < I;i++){
    for(int64_t j = 0;j < J;j++){
      T sum = 0;
      for(int64_t k = 0;k < K;k++){
        sum += a[i * a_row_stride + k * a_col_stride] * b[k * b_row_stride + j * b_col_stride];
      }

      T *od = &out[i * out_row_stride + j * out_col_stride];

      *od = (alpha * sum) + (beta * (*od));
    }
  }
}

#define INSTANTIATE_GEMM_IMPL(T) \
  template void gemm_impl<T>(const T alpha,const T *a,const int64_t a_row_stride,const int64_t a_col_stride, \
  const T *b,const int64_t b_row_stride,const int64_t b_col_stride, \
  const T beta,T *out,const int64_t out_row_stride,const int64_t out_col_stride, \
  const int64_t I,const int64_t J,const int64_t K);

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_GEMM_IMPL)

}//namespace cobalt_715::nn::ops::cpu