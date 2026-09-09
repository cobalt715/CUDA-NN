#ifdef COBALT_715_USE_CUDA

#include <cassert>
#include "vec.cuh"
#include "op.cuh"
#include "nn/Dtype.hpp"
#include "nn/cuda/util.cuh"
#include "nn/cuda/config.cuh"
#include "nn/cuda/LaunchConfig.cuh"

#include <cstdio>
#include <iostream>

namespace cobalt_715::nn::ops::vec::cuda{

template<nn::mutable_dtype T,typename Op>
__global__ void binary_op(const T *a,const T *b,T *out,const int64_t n){
  const int64_t x = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if(x >= n) return;

  out[x] = Op::f(a[x],b[x]);
}

inline nn::cuda::LaunchConfig binary_op_config(const int64_t n) noexcept{
  dim3 grid((n + 255) / 256);
  dim3 block(256);

  return nn::cuda::LaunchConfig(grid,block);
}

template<nn::mutable_dtype T>
void add(const T *a,const T *b,T *out,const int64_t n){
  const nn::cuda::LaunchConfig lc = binary_op_config(n);

  binary_op<T,op::add<T>><<<lc.grid_,lc.block_>>>(a,b,out,n);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
}

template<nn::mutable_dtype T>
void sub(const T *a,const T *b,T *out,const int64_t n){
  const nn::cuda::LaunchConfig lc = binary_op_config(n);

  binary_op<T,op::sub<T>><<<lc.grid_,lc.block_>>>(a,b,out,n);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
}

template<nn::mutable_dtype T>
void mul(const T *a,const T *b,T *out,const int64_t n){
  const nn::cuda::LaunchConfig lc = binary_op_config(n);

  binary_op<T,op::mul<T>><<<lc.grid_,lc.block_>>>(a,b,out,n);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
}

template<nn::mutable_dtype T>
void div(const T *a,const T *b,T *out,const int64_t n){
  const nn::cuda::LaunchConfig lc = binary_op_config(n);

  binary_op<T,op::div<T>><<<lc.grid_,lc.block_>>>(a,b,out,n);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
}

#define INSTANTIATE_ADD_SUB_MUL_DIV(T) \
  template void add(const T *a,const T *b,T *out,const int64_t n); \
  template void sub(const T *a,const T *b,T *out,const int64_t n); \
  template void mul(const T *a,const T *b,T *out,const int64_t n); \
  template void div(const T *a,const T *b,T *out,const int64_t n);

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_ADD_SUB_MUL_DIV)

template<nn::mutable_dtype T>
__global__ void dot_kernel(const T *a,const T *b,T *sum,const int64_t n){
  const int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tix = threadIdx.x;

  __shared__ T out[256];

  if(x < n){
    out[threadIdx.x] = a[x] * b[x];
  }else{
    out[threadIdx.x] = 0;
  }

  __syncthreads();

  for(int m = 128;m > 0;m /= 2){
    if(tix < m){
      out[tix] += out[tix + m];
    }
    __syncthreads();
  }

  if(tix == 0){
    if constexpr (std::is_same_v<T,int64_t>){
    static_assert(sizeof(int64_t) == sizeof(unsigned long long));
    atomicAdd(
        reinterpret_cast<unsigned long long*>(sum),
        static_cast<unsigned long long>(out[0])
    );
    }else{
      atomicAdd(sum,out[0]);
    }
  }
}

inline nn::cuda::LaunchConfig dot_kernel_config(const int64_t n) noexcept{
  return binary_op_config(n);
}

template<nn::mutable_dtype T>
void dot(const T *a,const T *b,T *out,const int64_t n){
  const nn::cuda::LaunchConfig lc = dot_kernel_config(n);

  cudaMemset(out,0,sizeof(float));
  dot_kernel<T><<<lc.grid_,lc.block_>>>(a,b,out,n);

  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
}

#define INSTANTIATE_DOT(T) \
  template void dot(const T *a,const T *b,T *out,const int64_t n);

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_DOT)

}//namespace cobalt_715::nn::ops::vec::cuda

#endif //COBALT_715_USE_CUDA