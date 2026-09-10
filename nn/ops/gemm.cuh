#pragma once

#include <cstdint>
#include "nn/dtype.hpp"

namespace cobalt_715::nn::ops{

namespace cpu{

//out = alpha * ab + beta * out
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
               const int64_t K);

}//namespace cpu

namespace cuda{

//out = alpha * ab + beta * out
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
               const int64_t K);

}//namespace cuda

}//namespace cobalt_715::nn::ops