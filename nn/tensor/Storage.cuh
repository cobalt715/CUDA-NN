#pragma once

#include <string>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <ostream>
#include <memory>
#include <algorithm>
#include "nn/Backend.hpp"
#include "nn/Dtype.hpp"
#include "nn/cuda/config.cuh"
#include "nn/cuda/util.cuh"

namespace cobalt_715::nn::tensor{

template<nn::dtype T=float>
struct Storage{
  //templateがconstだとエラーを投げる
  static_assert(!std::is_const_v<T>,"tensor::Storage T must not be const");

private:
  T *data_ = nullptr;
  int64_t size_;
  Backend backend_;

  inline void release(){
    if(backend_ == Backend::CPU){
      delete[] data_;
    }else if(backend_ == Backend::CUDA){
      #ifdef COBALT_715_USE_CUDA
        cudaFree(data_);
      #endif
    }

    data_ = nullptr;
    size_ = 0;
  }

public:
  Storage(const int64_t size,const Backend backend = Backend::CPU)
    : size_(size),
      backend_(backend){

    if(size_ < 0){
      throw std::invalid_argument("tensor::Storage::constructor negative size");
    }else if(size_ == 0) return;//要素が無いならnullptrになる

    if(backend_ == Backend::CPU){
      data_ = new T[size_];
    }else if(backend_ == Backend::CUDA){
      #ifdef COBALT_715_USE_CUDA
        const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&data_),size_ * sizeof(T));

        cuda::check(err);
      #else
        cuda::throw_not_enabled();
      #endif
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

  inline T* data() noexcept{
    return data_;
  }

  inline const T* data() const noexcept{
    return data_;
  }

  inline T& at(const int64_t i){
    #ifndef NDEBUG
      if(i < 0 || size_ <= i) throw std::out_of_range("tensor::Storage::at");
    #endif

    return data_[i];
  }

  inline const T& at(const int64_t i) const{
    #ifndef NDEBUG
      if(i < 0 || size_ <= i) throw std::out_of_range("tensor::Storage::at");
    #endif

    return data_[i];
  }

  inline int64_t size() const noexcept{
    return size_;
  }

  inline Backend backend() const noexcept{
    return backend_;
  }

  inline bool empty() const noexcept{
    return size_ == 0;
  }

  inline std::string dtype() const noexcept{
    return nn::dtype_name<T>();
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

  //to(Backend)
  Storage to(const Backend backend) const{
    if(backend == Backend::CPU){
      return toCPU();
    }else if(backend == Backend::CUDA){
      return toCUDA();
    }

    return toCPU();
  }

  //to cpu
  Storage toCPU() const{
    Storage s(size_,Backend::CPU);
    
    if(backend_ == Backend::CPU){
      std::memcpy(s.data(),data_,size_ * sizeof(T));
    }else if(backend_ == Backend::CUDA){
      #ifdef COBALT_715_USE_CUDA
        const cudaError_t err = cudaMemcpy(s.data(),data_,size_ * sizeof(T),cudaMemcpyDeviceToHost);

        cuda::check(err);
      #else
        cuda::throw_not_enabled();
      #endif
    }

    return s;
  }

  //to cuda
  Storage toCUDA() const{
    #ifdef COBALT_715_USE_CUDA
      Storage s(size_,Backend::CUDA);

      if(backend_ == Backend::CPU){
        const cudaError_t err = cudaMemcpy(s.data(),data_,size_ * sizeof(T),cudaMemcpyHostToDevice);

        cuda::check(err);
      }else if(backend_ == Backend::CUDA){
        const cudaError_t err = cudaMemcpy(s.data(),data_,size_ * sizeof(T),cudaMemcpyDeviceToDevice);

        cuda::check(err);
      }

      return s;
    #else
      cuda::throw_not_enabled();
    #endif
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
                       + ", dtype="
                       + nn::dtype_name<T>()
                       + ", data_={";

    const int64_t count = std::clamp<int64_t>(limit,0,size_);

    const T *data = nullptr;
    std::unique_ptr<T[]> host_data = std::make_unique<T[]>(count);

    if(backend_ == Backend::CPU){
      data = data_;
    }else if(backend_ == Backend::CUDA){
      #ifdef COBALT_715_USE_CUDA
        const cudaError_t err = cudaMemcpy(host_data.get(),data_,count * sizeof(T),cudaMemcpyDeviceToHost);

        data = host_data.get();

        cuda::check(err);
      #else
        cuda::throw_not_enabled();
      #endif
    }

    for(int64_t i = 0; i < count; ++i){
      if(i != 0) text += ", ";
      text += std::to_string(data[i]);
    }

    if(count < size_){
      if(count == 0){
        text += "...";
      }else{
        text += ", ...";
      }
    }

    text += "})";

    return text;
  }
};

template<class T>
inline std::ostream& operator<<(std::ostream &o,const Storage<T> &s){
  return o << s.to_string(8);
}

}//namespace cobalt_715::nn::tensor