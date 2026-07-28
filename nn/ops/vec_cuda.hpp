#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include "vec_fwd.hpp"
#include "nn/Backend.hpp"

namespace cobalt_715::nn::ops::vec{

__global__ void cuda_add_alias_safe(const float *a,const float *b,float *out,const size_t n){
  const int64_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x >= n) return;

  out[x] = a[x] + b[x];
}

//out[i] = a[i] + b[i]
template<>
inline void add_alias_safe<Backend::CUDA>(const float *a,const float *b,float *out,const size_t n){
  cuda_add_alias_safe<<<(n + 255) / 256,256>>>(a,b,out,n);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

template<>
inline void add<Backend::CUDA>(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n){
  add_alias_safe<Backend::CUDA>(a,b,out,n);
}

}//namespace cobalt_715::nn::ops::vec