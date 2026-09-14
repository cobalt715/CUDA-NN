#ifdef COBALT_715_USE_CUDA

#include <cassert>
#include "vec.cuh"
#include "op.cuh"
#include "nn/dtype.hpp"
#include "nn/cuda/util.cuh"
#include "nn/cuda/config.cuh"
#include "nn/cuda/LaunchConfig.cuh"

namespace cobalt_715::nn::ops::matrix::cuda{

template<nn::mutable_dtype T,class Op>
__global__ void binary_op_2d(const T *a,
                             const int64_t a_row_stride,
                             const int64_t a_col_stride,

                             const T *b,
                             const int64_t b_row_stride,
                             const int64_t b_col_stride,

                             T *out,
                             const int64_t out_row_stride,
                             const int64_t out_col_stride,

                             const int64_t rows,
                             const int64_t cols){

  const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if(row >= rows || col >= cols) return;

  const T a_val = a[row * a_row_stride + col * a_col_stride];
  const T b_val = b[row * b_row_stride + col * b_col_stride];

  out[row * out_row_stride + col * out_col_stride] = Op::f(a_val,b_val);
}

inline nn::cuda::LaunchConfig binary_op_2d_config(const int64_t row,const int64_t col) noexcept{
  dim3 grid((row + 15) / 16,(col + 15) / 16);
  dim3 block(16,16);

  return nn::cuda::LaunchConfig(grid,block);
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
         const int64_t cols){

  const nn::cuda::LaunchConfig lc = binary_op_2d_config(rows,cols);

  binary_op_2d<T,op::add<T>><<<lc.grid_,lc.block_>>>(
    a,a_row_stride,a_col_stride,
    b,b_row_stride,b_col_stride,
    out,out_row_stride,out_col_stride,
    rows,cols);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
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
         const int64_t cols){

  const nn::cuda::LaunchConfig lc = binary_op_2d_config(rows,cols);

  binary_op_2d<T,op::sub<T>><<<lc.grid_,lc.block_>>>(
    a,a_row_stride,a_col_stride,
    b,b_row_stride,b_col_stride,
    out,out_row_stride,out_col_stride,
    rows,cols);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
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
         const int64_t cols){

  const nn::cuda::LaunchConfig lc = binary_op_2d_config(rows,cols);

  binary_op_2d<T,op::mul<T>><<<lc.grid_,lc.block_>>>(
    a,a_row_stride,a_col_stride,
    b,b_row_stride,b_col_stride,
    out,out_row_stride,out_col_stride,
    rows,cols);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
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
         const int64_t cols){

  const nn::cuda::LaunchConfig lc = binary_op_2d_config(rows,cols);

  binary_op_2d<T,op::div<T>><<<lc.grid_,lc.block_>>>(
    a,a_row_stride,a_col_stride,
    b,b_row_stride,b_col_stride,
    out,out_row_stride,out_col_stride,
    rows,cols);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
}

#define INSTANTIATE_2D_ADD_SUB_MUL_DIV(T) \
  template void add( \
  const T *a,const int64_t a_row_stride,const int64_t a_col_stride, \
  const T *b,const int64_t b_row_stride,const int64_t b_col_stride, \
  T *out,const int64_t out_row_stride,const int64_t out_col_stride, \
  const int64_t rows,const int64_t cols); \
  template void sub( \
  const T *a,const int64_t a_row_stride,const int64_t a_col_stride, \
  const T *b,const int64_t b_row_stride,const int64_t b_col_stride, \
  T *out,const int64_t out_row_stride,const int64_t out_col_stride, \
  const int64_t rows,const int64_t cols); \
  template void mul( \
  const T *a,const int64_t a_row_stride,const int64_t a_col_stride, \
  const T *b,const int64_t b_row_stride,const int64_t b_col_stride, \
  T *out,const int64_t out_row_stride,const int64_t out_col_stride, \
  const int64_t rows,const int64_t cols); \
  template void div( \
  const T *a,const int64_t a_row_stride,const int64_t a_col_stride, \
  const T *b,const int64_t b_row_stride,const int64_t b_col_stride, \
  T *out,const int64_t out_row_stride,const int64_t out_col_stride, \
  const int64_t rows,const int64_t cols);

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_2D_ADD_SUB_MUL_DIV)

}//namespace cobalt_715::nn::ops::matrix::cuda

#endif