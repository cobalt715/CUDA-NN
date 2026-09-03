#pragma once

#include <cstdint>
#include <type_traits>
#include "MatrixView.hpp"
#include "Storage.cuh"
#include "nn/Backend.hpp"
#include "nn/Dtype.hpp"
#include "nn/cuda/config.cuh"
#include "nn/cuda/util.cuh"

namespace cobalt_715::nn::tensor{

template<nn::dtype T>
Storage<std::remove_const_t<T>> MatrixView<T>::to_string_cpu_copy(const int64_t ro,const int64_t co) const{
  Storage<std::remove_const_t<T>> data(ro * co,Backend::CPU);

  for(int64_t row = 0;row < ro;row++){
    for(int64_t col = 0;col < co;col++){
      data.at(row * co + col) = at(row,col);
    }
  }

  return data;
}

#ifdef __CUDACC__
namespace{

template<cobalt_715::nn::dtype T>
__global__ void copy(const T *a,
                     T *data,
                     const int64_t ro,
                     const int64_t co,
                     const int64_t row_stride,
                     const int64_t col_stride,
                     const int64_t offset){

  //templateがconstだとエラーを投げる
  static_assert(!std::is_const_v<T>,"nn/tensor/MatrixView.cu namespace::__global__ copy() T must not be const");

  const int64_t x = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t y = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;

  if(co <= x) return;
  if(ro <= y) return;

  data[y * co + x] = a[offset + y * row_stride + x * col_stride];
}

}//namespace
#endif

template<dtype T>
Storage<std::remove_const_t<T>> MatrixView<T>::to_string_cuda_copy(const int64_t ro,const int64_t co) const{
  #ifdef COBALT_715_USE_CUDA
  Storage<std::remove_const_t<T>> data(ro * co,Backend::CUDA);

  const dim3 grid((ro + 15) / 16,(co + 15) / 16);
  const dim3 block(16,16);

  #ifdef __CUDACC__
  copy<std::remove_const_t<T>><<<grid,block>>>(data_,data.data(),ro,co,row_stride_,col_stride_,offset_);
  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());
  #endif

  return data.toCPU();
  #else
  nn::cuda::throw_not_enabled();
  #endif
}

}//namespace cobalt_715::nn::tensor