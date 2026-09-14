#include <cassert>
#include "vec.cuh"
#include "op.cuh"
#include "nn/dtype.hpp"
#include "nn/cuda/util.cuh"

namespace cobalt_715::nn::ops::matrix::cpu{

template<nn::mutable_dtype T,class Op>
void binary_op_2d(const T *a,
                  const int64_t a_row_stride,
                  const int64_t a_col_stride,

                  const T *b,
                  const int64_t b_row_stride,
                  const int64_t b_col_stride,

                  T *out,
                  const int64_t out_row_stride,
                  const int64_t out_col_stride,

                  const int64_t rows,
                  const int64_t cols) noexcept{

  for(int64_t row = 0;row < rows;row++){
    for(int64_t col = 0;col < cols;col++){
      const T a_val = a[row * a_row_stride + col * a_col_stride];
      const T b_val = b[row * b_row_stride + col * b_col_stride];

      out[row * out_row_stride + col * out_col_stride] = Op::f(a_val,b_val);
    }
  }
}

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
         const int64_t cols) noexcept{

  binary_op_2d<T,op::add<T>>(
    a,a_row_stride,a_col_stride,
    b,b_row_stride,b_col_stride,
    out,out_row_stride,out_col_stride,
    rows,cols);
}

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
         const int64_t cols) noexcept{

  binary_op_2d<T,op::sub<T>>(
    a,a_row_stride,a_col_stride,
    b,b_row_stride,b_col_stride,
    out,out_row_stride,out_col_stride,
    rows,cols);
}

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
         const int64_t cols) noexcept{

  binary_op_2d<T,op::mul<T>>(
    a,a_row_stride,a_col_stride,
    b,b_row_stride,b_col_stride,
    out,out_row_stride,out_col_stride,
    rows,cols);
}

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
         const int64_t cols) noexcept{

  binary_op_2d<T,op::div<T>>(
    a,a_row_stride,a_col_stride,
    b,b_row_stride,b_col_stride,
    out,out_row_stride,out_col_stride,
    rows,cols);
}

#define INSTANTIATE_2D_ADD_SUB_MUL_DIV(T) \
  template void add( \
  const T *a,const int64_t a_row_stride,const int64_t a_col_stride, \
  const T *b,const int64_t b_row_stride,const int64_t b_col_stride, \
  T *out,const int64_t out_row_stride,const int64_t out_col_stride, \
  const int64_t rows,const int64_t cols) noexcept; \
  template void sub( \
  const T *a,const int64_t a_row_stride,const int64_t a_col_stride, \
  const T *b,const int64_t b_row_stride,const int64_t b_col_stride, \
  T *out,const int64_t out_row_stride,const int64_t out_col_stride, \
  const int64_t rows,const int64_t cols) noexcept; \
  template void mul( \
  const T *a,const int64_t a_row_stride,const int64_t a_col_stride, \
  const T *b,const int64_t b_row_stride,const int64_t b_col_stride, \
  T *out,const int64_t out_row_stride,const int64_t out_col_stride, \
  const int64_t rows,const int64_t cols) noexcept; \
  template void div( \
  const T *a,const int64_t a_row_stride,const int64_t a_col_stride, \
  const T *b,const int64_t b_row_stride,const int64_t b_col_stride, \
  T *out,const int64_t out_row_stride,const int64_t out_col_stride, \
  const int64_t rows,const int64_t cols) noexcept;

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_2D_ADD_SUB_MUL_DIV)

}//namespace cobalt_715::nn::ops::matrix::cpu