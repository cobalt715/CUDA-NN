#include "Storage.cuh"
#include <cuda_runtime.h>
#include "nn/ops/vec.hpp"
#include "nn/Backend.hpp"

namespace cobalt_715::nn::tensor{

__global__ void cuda_fill(float *__restrict__ a,const int64_t n,const float f){
  const int64_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x >= n) return;

  a[x] = f;
}

void Storage::fill(const float f){
  if(backend_ == Backend::CPU){
    std::fill(data_,data_ + size_,f);
  }else if(backend_ == Backend::CUDA){
    cuda_fill<<<(size_ + 255) / 256,256>>>(data_,size_,f);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
      throw std::runtime_error(cudaGetErrorString(err));
    }
  }
 }

void Storage::add(const Storage &a,const Storage &b,Storage &out){
  #ifndef NDEBUG
    if(!same_backend(a,b,out)) throw std::runtime_error("Storage::add Backend mismatch");
    if(!Storage::same_size(a,b,out)) throw std::runtime_error("Storage::add size mismatch");
  #endif

  if(a.backend() == Backend::CPU){
    ops::vec::add<Backend::CPU>(a.data(),b.data(),out.data(),a.size());
  }else if(a.backend() == Backend::CUDA){
    ops::vec::add<Backend::CUDA>(a.data(),b.data(),out.data(),a.size());
  }
}

}//namespace cobalt_715::nn::tensor