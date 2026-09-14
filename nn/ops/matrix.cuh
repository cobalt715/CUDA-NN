#pragma once

#include <cstdint>
#include <nn/dtype.hpp>

namespace cobalt_715::nn::ops::matrix{

namespace cpu{

template<nn::mutable_dtype T>
void add(const T *a,
         const int64_t a_row_stride,
         const int64_t a_col_stride,

         const T *b,
         const int64_t b_row_stride,
         const int64_t b_col_stride,

         T *out,
         const int64_t out_row_stride,
         const int64_t out_col_stride,

         const int64_t rows,
         const int64_t cols) noexcept;

template<nn::mutable_dtype T>
void sub(const T *a,
         const int64_t a_row_stride,
         const int64_t a_col_stride,

         const T *b,
         const int64_t b_row_stride,
         const int64_t b_col_stride,

         T *out,
         const int64_t out_row_stride,
         const int64_t out_col_stride,

         const int64_t rows,
         const int64_t cols) noexcept;

template<nn::mutable_dtype T>
void mul(const T *a,
         const int64_t a_row_stride,
         const int64_t a_col_stride,

         const T *b,
         const int64_t b_row_stride,
         const int64_t b_col_stride,

         T *out,
         const int64_t out_row_stride,
         const int64_t out_col_stride,

         const int64_t rows,
         const int64_t cols) noexcept;

template<nn::mutable_dtype T>
void div(const T *a,
         const int64_t a_row_stride,
         const int64_t a_col_stride,

         const T *b,
         const int64_t b_row_stride,
         const int64_t b_col_stride,

         T *out,
         const int64_t out_row_stride,
         const int64_t out_col_stride,

         const int64_t rows,
         const int64_t cols) noexcept;

}//namespace cpu

namespace cuda{

template<nn::mutable_dtype T>
void add(const T *a,
         const int64_t a_row_stride,
         const int64_t a_col_stride,

         const T *b,
         const int64_t b_row_stride,
         const int64_t b_col_stride,

         T *out,
         const int64_t out_row_stride,
         const int64_t out_col_stride,

         const int64_t rows,
         const int64_t cols);

template<nn::mutable_dtype T>
void sub(const T *a,
         const int64_t a_row_stride,
         const int64_t a_col_stride,

         const T *b,
         const int64_t b_row_stride,
         const int64_t b_col_stride,

         T *out,
         const int64_t out_row_stride,
         const int64_t out_col_stride,

         const int64_t rows,
         const int64_t cols);

template<nn::mutable_dtype T>
void mul(const T *a,
         const int64_t a_row_stride,
         const int64_t a_col_stride,

         const T *b,
         const int64_t b_row_stride,
         const int64_t b_col_stride,

         T *out,
         const int64_t out_row_stride,
         const int64_t out_col_stride,

         const int64_t rows,
         const int64_t cols);

template<nn::mutable_dtype T>
void div(const T *a,
         const int64_t a_row_stride,
         const int64_t a_col_stride,

         const T *b,
         const int64_t b_row_stride,
         const int64_t b_col_stride,

         T *out,
         const int64_t out_row_stride,
         const int64_t out_col_stride,

         const int64_t rows,
         const int64_t cols);

}//namespace cuda

}//namespace cobalt_715::nn::ops::matrix