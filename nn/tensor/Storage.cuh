#pragma once

#include <cuda_runtime.h>
#include <string>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <ostream>
#include "nn/Backend.hpp"

namespace cobalt_715::nn::tensor{

static __global__ void fill_kernel(float *__restrict__ a,const int64_t n,const float f){
  const int64_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x >= n) return;

  a[x] = f;
}

struct Storage{
private:
  float *data_;
  int64_t size_;
  Backend backend_;

public:
  Storage(const int64_t size,const Backend backend)
    : size_(size),
      backend_(backend){

    if(size_ < 0){
      throw std::invalid_argument("tensor::Storage::constructor negative size");
    }

    if(backend_ == Backend::CPU){
      data_ = new float[size_];
    }else if(backend_ == Backend::CUDA){
      const cudaError_t err = cudaMalloc((void**)&data_,size_ * sizeof(float));

      if(err != cudaSuccess){
        throw std::runtime_error(std::string("tensor::Storage::constructor") + cudaGetErrorString(err));
      }
    }
  }

  ~Storage(){
    if(backend_ == Backend::CPU){
      delete[] data_;
    }else if(backend_ == Backend::CUDA){
      cudaFree(data_);
    }
  }

  __host__ __device__ float* data(){
    return data_;
  }

  __host__ __device__ const float* data() const{
    return data_;
  }

  __host__ __device__ float& at(const int64_t i){
    return data_[i];
  }

  __host__ __device__ const float& at(const int64_t i) const{
    return data_[i];
  }

  __host__ __device__ int64_t size() const{
    return size_;
  }

  Backend backend() const{
    return backend_;
  }

  void fill(const float f){
    if(backend_ == Backend::CPU){
      std::fill(data_,data_ + size_,f);
    }else if(backend_ == Backend::CUDA){
      fill_kernel<<<(size_ + 255) / 256,256>>>(data_,size_,f);

      cudaError_t err = cudaGetLastError();
      if(err != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(err));
      }
    }
  }

  Storage toCPU() const{
    Storage s(size_,Backend::CPU);
    
    if(backend_ == Backend::CPU){
      std::memcpy(s.data(),data_,size_ * sizeof(float));
    }else if(backend_ == Backend::CUDA){
      cudaError_t err = cudaMemcpy(s.data(),data_,size_ * sizeof(float),cudaMemcpyDeviceToHost);

      if(err != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(err));
      }
    }

    return s;
  }

  Storage toCUDA() const{
    Storage s(size_,Backend::CUDA);

    if(backend_ == Backend::CPU){
      cudaMemcpy(s.data(),data_,size_ * sizeof(float),cudaMemcpyHostToDevice);
    }else if(backend_ == Backend::CUDA){
      cudaError_t err = cudaMemcpy(s.data(),data_,size_ * sizeof(float),cudaMemcpyDeviceToDevice);

      if(err != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(err));
      }
    }

    return s;
  }

  std::string to_string() const{
    std::string s = "Storage(" + std::to_string(size_) + "," + std::string(nn::to_string(backend_)) + ")\n{";

    Storage st = toCPU();

    for(int64_t i = 0;i < st.size() - 1;i++){
      s += std::to_string(st.at(i)) + " , ";
    }

    s += std::to_string(st.at(st.size() - 1)) + "}";

    return s;
  }
};

inline std::ostream& operator<<(std::ostream &o,const Storage &s){
  return o << s.to_string();
}

}//namespace cobalt_715::nn::tensor