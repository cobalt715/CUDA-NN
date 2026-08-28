#pragma once

#include <string>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <ostream>
#include <algorithm>
#include <cuda_runtime.h>
#include "nn/Backend.hpp"
#include "nn/CUDAUtil.cuh"

namespace cobalt_715::nn::tensor{

struct Storage{
private:
  float *data_ = nullptr;
  int64_t size_;
  Backend backend_;

  void release(){
    if(backend_ == Backend::CPU){
      delete[] data_;
    }else if(backend_ == Backend::CUDA){
      cudaFree(data_);
    }

    data_ = nullptr;
    size_ = 0;
  }

public:
  Storage(const int64_t size,const Backend backend)
    : size_(size),
      backend_(backend){

    if(size_ < 0){
      throw std::invalid_argument("tensor::Storage::constructor negative size");
    }else if(size_ == 0) return;//要素が無いならnullptrになる

    if(backend_ == Backend::CPU){
      data_ = new float[size_];
    }else if(backend_ == Backend::CUDA){
      const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&data_),size_ * sizeof(float));

      nn::check_cuda(err);
    }
  }

  ~Storage() noexcept{
    release();
  }

  //コピー禁止
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  //moveのみ
  Storage(Storage &&s) noexcept
    : data_(s.data()),
      size_(s.size()),
      backend_(s.backend()){

      s.data_ = nullptr;
      s.size_ = 0;
  }

  Storage& operator=(Storage &&s) noexcept{
    if(this == &s) return *this;

    release();

    data_ = s.data();
    size_ = s.size();
    backend_ = s.backend();

    s.data_ = nullptr;
    s.size_ = 0;

    return *this;
  }

  __host__ __device__ float* data() noexcept{
    return data_;
  }

  __host__ __device__ const float* data() const noexcept{
    return data_;
  }

  __host__ __device__ float& at(const int64_t i) noexcept{
    return data_[i];
  }

  __host__ __device__ const float& at(const int64_t i) const noexcept{
    return data_[i];
  }

  __host__ __device__ int64_t size() const noexcept{
    return size_;
  }

  Backend backend() const noexcept{
    return backend_;
  }

  bool empty() const noexcept{
    return size_ == 0;
  }

  //clone
  Storage clone() const{
    if(backend_ == Backend::CPU){
      return toCPU();
    }else if(backend_ == Backend::CUDA){
      return toCUDA();
    }

    return toCPU();
  }

  //to cpu
  Storage toCPU() const{
    Storage s(size_,Backend::CPU);
    
    if(backend_ == Backend::CPU){
      std::memcpy(s.data(),data_,size_ * sizeof(float));
    }else if(backend_ == Backend::CUDA){
      const cudaError_t err = cudaMemcpy(s.data(),data_,size_ * sizeof(float),cudaMemcpyDeviceToHost);

      nn::check_cuda(err);
    }

    return s;
  }

  //to cuda
  Storage toCUDA() const{
    Storage s(size_,Backend::CUDA);

    if(backend_ == Backend::CPU){
      const cudaError_t err = cudaMemcpy(s.data(),data_,size_ * sizeof(float),cudaMemcpyHostToDevice);

      nn::check_cuda(err);
    }else if(backend_ == Backend::CUDA){
      const cudaError_t err = cudaMemcpy(s.data(),data_,size_ * sizeof(float),cudaMemcpyDeviceToDevice);

      nn::check_cuda(err);
    }

    return s;
  }

  //to string
  std::string to_string() const{
    return to_string(size_);
  }

  std::string to_string(const int64_t limit) const{
    std::string text = "tensor::Storage(size_="
                       + std::to_string(size_)
                       + ", backend_="
                       + nn::to_string(backend_)
                       + ", data_={";

    const int64_t count = std::clamp<int64_t>(limit,0,size_);

    float *data = nullptr;

    if(backend_ == Backend::CPU){
      data = data_;
    }else if(backend_ == Backend::CUDA){
      data = new float[count];

      const cudaError_t err = cudaMemcpy(data,data_,count * sizeof(float),cudaMemcpyDeviceToHost);

      if(err != cudaSuccess) delete[] data;

      nn::check_cuda(err);
    }

    for(int64_t i = 0; i < count; ++i){
      if(i != 0) text += ", ";
      text += std::to_string(data[i]);
    }

    if(count < size_){
      text += ", ...";
    }

    text += "})";

    if(backend_ == Backend::CUDA){
      delete[] data;
    }

    return text;
  }
};


inline std::ostream& operator<<(std::ostream &o,const Storage &s){
  return o << s.to_string();
}

}//namespace cobalt_715::nn::tensor