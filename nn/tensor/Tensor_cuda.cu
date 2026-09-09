#ifdef COBALT_715_USE_CUDA

#include <cstdint>
#include <type_traits>
#include "Tensor.hpp"
#include "Storage.cuh"
#include "nn/Backend.hpp"
#include "nn/Dtype.hpp"
#include "nn/cuda/config.cuh"
#include "nn/cuda/util.cuh"

namespace cobalt_715::nn::tensor{

template<cobalt_715::nn::mutable_dtype T>
__global__ void tensor_to_string_cuda_element_copy(T *data,
                                                   const T *arr,
                                                   const int64_t *data_stride,
                                                   const int64_t *arr_stride,
                                                   const int64_t N,
                                                   const int64_t data_numel){

  const int64_t x = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if(data_numel <= x) return;

  int64_t offset = 0;
  int64_t some = x;

  for(int64_t i = 0;i < N;i++){
    const int64_t ds =data_stride[i];
    offset += some / ds * arr_stride[i];
    some %= ds;
  }

  data[x] = arr[offset];
}

template<cobalt_715::nn::mutable_dtype T>
Storage<T> Tensor<T>::to_string_cuda_copy(const int64_t copy_len,
                                          const Storage<int64_t> &stride) const{

  Storage<T> data(copy_len,Backend::CUDA);

  const dim3 grid((copy_len + 255) / 256);
  const dim3 block(256);

  tensor_to_string_cuda_element_copy<T><<<grid,block>>>(data.data(),data_.data(),stride.toCUDA().data(),device_stride_.data(),rank(),copy_len);

  return data.toCPU();
}

#define INSTANTIATE_1(T) \
  template Storage<T> \
  Tensor<T>::to_string_cuda_copy( \
  const int64_t copy_len, \
  const Storage<int64_t> &stride) const;

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_1)

#define INSTANTIATE_2(T) \
  template __global__ void \
  tensor_to_string_cuda_element_copy( \
  T *data, \
  const T *arr, \
  const int64_t *data_stride, \
  const int64_t *arr_stride, \
  const int64_t N, \
  const int64_t data_numel);

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_2)

}//namespace cobalt_715::nn::tensor

#endif