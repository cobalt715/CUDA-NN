#ifndef COBALT_715_USE_CUDA

#include <cstdint>
#include <type_traits>
#include "Tensor.hpp"
#include "Storage.cuh"
#include "nn/Backend.hpp"
#include "nn/dtype.hpp"
#include "nn/cuda/config.cuh"
#include "nn/cuda/util.cuh"

namespace cobalt_715::nn::tensor{

template<cobalt_715::nn::mutable_dtype T>
Storage<T> Tensor<T>::to_string_cuda_copy(const int64_t copy_len,
                                          const Storage<int64_t> &stride) const{
  nn::cuda::throw_not_enabled();
}

#define INSTANTIATE_1(T) \
  template Storage<T> \
  Tensor<T>::to_string_cuda_copy( \
  const int64_t copy_len, \
  const Storage<int64_t> &stride) const;

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_1)

}//namespace cobalt_715::nn::tensor

#endif