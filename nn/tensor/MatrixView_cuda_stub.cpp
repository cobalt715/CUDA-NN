#ifndef COBALT_715_USE_CUDA

#include <cstdint>
#include <type_traits>
#include "MatrixView.hpp"
#include "Storage.cuh"
#include "nn/Backend.hpp"
#include "nn/Dtype.hpp"
#include "nn/cuda/config.cuh"
#include "nn/cuda/util.cuh"

namespace cobalt_715::nn::tensor{

template<dtype T>
Storage<std::remove_const_t<T>> MatrixView<T>::to_string_cuda_copy(const int64_t ro,const int64_t co) const{
  nn::cuda::throw_not_enabled();
}

#define INSTANTIATE_1(T) \
  template Storage<std::remove_const_t<T>> MatrixView<T>::to_string_cuda_copy(const int64_t ro,const int64_t co) const;

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_1)
COBALT_715_FOR_EACH_CONST_DTYPE(INSTANTIATE_1)

}//namespace cobalt_715::nn::tensor

#endif