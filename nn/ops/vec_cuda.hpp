#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include "vec_fwd.hpp"
#include "nn/Backend.hpp"

namespace cobalt_715::nn::ops::vec{

namespace op{

struct add{
  __forceinline__ __host__ __device__ float operator()(const float a,const float b) const{
    return a + b;
  }
};

struct sub{
  __forceinline__ __host__ __device__ float operator()(const float a,const float b) const{
    return a - b;
  }
};

struct mul{
  __forceinline__ __host__ __device__ float operator()(const float a,const float b) const{
    return a * b;
  }
};

struct div{
  __forceinline__ __host__ __device__ float operator()(const float a,const float b) const{
    return a / b;
  }
};

}//namespace op

template<class T>
__global__ void cuda_binary_alias_safe(const float *a,const float *b,float *out,const size_t n){
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x >= n) return;

  T t;

  out[x] = t(a[x],b[x]);
}

//out[i] = a[i] + b[i]
template<>
inline void add_alias_safe<Backend::CUDA>(const float *a,const float *b,float *out,const size_t n){
  cuda_binary_alias_safe<op::add><<<(n + 255) / 256,256>>>(a,b,out,n);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

template<>
inline void add<Backend::CUDA>(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n){
  add_alias_safe<Backend::CUDA>(a,b,out,n);
}

//out[i] = a[i] - b[i]
template<>
inline void sub_alias_safe<Backend::CUDA>(const float *a,const float *b,float *out,const size_t n){
  cuda_binary_alias_safe<op::sub><<<(n + 255) / 256,256>>>(a,b,out,n);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

template<>
inline void sub<Backend::CUDA>(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n){
  sub_alias_safe<Backend::CUDA>(a,b,out,n);
}

//out[i] = a[i] * b[i]
template<>
inline void mul_alias_safe<Backend::CUDA>(const float *a,const float *b,float *out,const size_t n){
  cuda_binary_alias_safe<op::mul><<<(n + 255) / 256,256>>>(a,b,out,n);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

template<>
inline void mul<Backend::CUDA>(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n){
  mul_alias_safe<Backend::CUDA>(a,b,out,n);
}

//out[i] = a[i] / b[i]
template<>
inline void div_alias_safe<Backend::CUDA>(const float *a,const float *b,float *out,const size_t n){
  cuda_binary_alias_safe<op::div><<<(n + 255) / 256,256>>>(a,b,out,n);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

template<>
inline void div<Backend::CUDA>(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n){
  div_alias_safe<Backend::CUDA>(a,b,out,n);
}

}//namespace cobalt_715::nn::ops::vec