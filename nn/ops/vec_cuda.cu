#ifdef COBALT_715_USE_CUDA

#include "vec.cuh"
#include "op.cuh"
#include "nn/cuda/util.cuh"
#include "nn/cuda/config.cuh"
#include "nn/cuda/LaunchConfig.cuh"

#include <cstdio>
#include <iostream>

namespace cobalt_715::nn::ops::vec::cuda{

template<typename Op>
__global__ void binary_op(const float *a,const float *b,float *out,const int64_t n){
  const int64_t x = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if(x >= n) return;

  out[x] = Op::f(a[x],b[x]);
}

inline nn::cuda::LaunchConfig binary_op_config(const int64_t n){
  dim3 grid((n + 255) / 256);
  dim3 block(256);

  return nn::cuda::LaunchConfig(grid,block);
}

void add(const float *a,const float *b,float *out,const int64_t n){
  const nn::cuda::LaunchConfig lc = binary_op_config(n);

  binary_op<op::add><<<lc.grid_,lc.block_>>>(a,b,out,n);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
}

void sub(const float *a,const float *b,float *out,const int64_t n){
  const nn::cuda::LaunchConfig lc = binary_op_config(n);

  binary_op<op::sub><<<lc.grid_,lc.block_>>>(a,b,out,n);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
}

void mul(const float *a,const float *b,float *out,const int64_t n){
  const nn::cuda::LaunchConfig lc = binary_op_config(n);

  binary_op<op::mul><<<lc.grid_,lc.block_>>>(a,b,out,n);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
}

void div(const float *a,const float *b,float *out,const int64_t n){
  const nn::cuda::LaunchConfig lc = binary_op_config(n);

  binary_op<op::div><<<lc.grid_,lc.block_>>>(a,b,out,n);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
}

}//namespace cobalt_715::nn::ops::vec::cuda

#endif //COBALT_715_USE_CUDA