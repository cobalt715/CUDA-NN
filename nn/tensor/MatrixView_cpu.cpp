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

#define INSTANTIATE(T) \
  template Storage<std::remove_const_t<T>> MatrixView<T>::to_string_cpu_copy(const int64_t ro,const int64_t co) const;

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE)
COBALT_715_FOR_EACH_CONST_DTYPE(INSTANTIATE)

}//namespace cobalt_715::nn::tensor