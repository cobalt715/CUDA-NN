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

inline nn::cuda::LaunchConfig binary_op_config(const int64_t n) noexcept{
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

__global__ void dot_kernel(const float *a,const float *b,float *sum,const int64_t n){
  const int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tix = threadIdx.x;

  __shared__ float out[256];

  if(x < n){
    out[threadIdx.x] = a[x] * b[x];
  }else{
    out[threadIdx.x] = 0.0f;
  }

  __syncthreads();

  for(int m = 128;m > 0;m /= 2){
    if(tix < m){
      out[tix] += out[tix + m];
    }
    __syncthreads();
  }

  if(tix == 0){
    atomicAdd(sum,out[0]);
  }
}

inline nn::cuda::LaunchConfig dot_kernel_config(const int64_t n) noexcept{
  return binary_op_config(n);
}

void dot(const float *a,const float *b,float *out,const int64_t n){
  const nn::cuda::LaunchConfig lc = dot_kernel_config(n);

  cudaMemset(out,0,sizeof(float));
  dot_kernel<<<lc.grid_,lc.block_>>>(a,b,out,n);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
}

}//namespace cobalt_715::nn::ops::vec::cuda

#endif //COBALT_715_USE_CUDA