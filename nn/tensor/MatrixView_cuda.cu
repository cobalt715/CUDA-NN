#ifdef COBALT_715_USE_CUDA

#include <cstdint>
#include <type_traits>
#include "MatrixView.hpp"
#include "Storage.cuh"
#include "nn/Backend.hpp"
#include "nn/Dtype.hpp"
#include "nn/cuda/config.cuh"
#include "nn/cuda/util.cuh"

namespace cobalt_715::nn::tensor{

template<cobalt_715::nn::dtype T>
__global__ void matrix_to_string_cuda_element_copy(T *data,
                                                   const T *arr,
                                                   const int64_t ro,
                                                   const int64_t co,
                                                   const int64_t row_stride,
                                                   const int64_t col_stride){

  //templateがconstだとエラーを投げる
  static_assert(!std::is_const_v<T>,"nn/tensor/MatrixView.cu namespace::__global__ copy() T must not be const");

  const int64_t x = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t y = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;

  if(co <= x) return;
  if(ro <= y) return;

  data[y * co + x] = arr[y * row_stride + x * col_stride];
}

#define INSTANTIATE_MATRIX_TO_STRING_CUDA_ELEMENT_COPY(T) \
  template __global__ void \
  matrix_to_string_cuda_element_copy( \
  T *data, \
  const T *arr, \
  const int64_t ro, \
  const int64_t co, \
  const int64_t row_stride, \
  const int64_t col_stride);

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_MATRIX_TO_STRING_CUDA_ELEMENT_COPY)

template<dtype T>
Storage<std::remove_const_t<T>> MatrixView<T>::to_string_cuda_copy(const int64_t ro,const int64_t co) const{
  Storage<std::remove_const_t<T>> arr(ro * co,Backend::CUDA);

  const dim3 grid((co + 15) / 16,(ro + 15) / 16);
  const dim3 block(16,16);

  matrix_to_string_cuda_element_copy<std::remove_const_t<T>><<<grid,block>>>(arr.data(),data_.data() + offset_,ro,co,row_stride_,col_stride_);
  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());

  return arr.toCPU();
}

#define INSTANTIATE_TO_STRING_CUDA_COPY(T) \
  template Storage<std::remove_const_t<T>> MatrixView<T>::to_string_cuda_copy(const int64_t ro,const int64_t co) const;

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_TO_STRING_CUDA_COPY)
COBALT_715_FOR_EACH_CONST_DTYPE(INSTANTIATE_TO_STRING_CUDA_COPY)

}//namespace cobalt_715::nn::tensor

#endif