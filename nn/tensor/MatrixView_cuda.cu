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
__global__ void to_string_cuda_element_copy(const T *a,
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

template<dtype T>
Storage<std::remove_const_t<T>> MatrixView<T>::to_string_cuda_copy(const int64_t ro,const int64_t co) const{
  Storage<std::remove_const_t<T>> data(ro * co,Backend::CUDA);

  const dim3 grid((co + 15) / 16,(ro + 15) / 16);
  const dim3 block(16,16);

  to_string_cuda_element_copy<std::remove_const_t<T>><<<grid,block>>>(data_.data(),data.data(),ro,co,row_stride_,col_stride_,offset_);
  nn::cuda::check(cudaGetLastError());
  nn::cuda::check(cudaDeviceSynchronize());

  return data.toCPU();
}

#define INSTANTIATE_1(T) \
  template Storage<std::remove_const_t<T>> MatrixView<T>::to_string_cuda_copy(const int64_t ro,const int64_t co) const;

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_1)
COBALT_715_FOR_EACH_CONST_DTYPE(INSTANTIATE_1)

#define INSTANTIATE_2(T) \
  template __global__ void \
  to_string_cuda_element_copy( \
  const T *a, \
  T *data, \
  const int64_t ro, \
  const int64_t co, \
  const int64_t row_stride, \
  const int64_t col_stride, \
  const int64_t offset);

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_2)

}//namespace cobalt_715::nn::tensor

#endif