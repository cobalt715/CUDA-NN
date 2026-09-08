#include <cstdint>
#include "Tensor.hpp"
#include "Storage.cuh"
#include "nn/Dtype.hpp"

namespace cobalt_715::nn::tensor{

template<nn::dtype T>
Storage<T> Tensor<T>::to_string_cpu_copy(const int64_t copy_len,
                                      const Storage<int64_t> &stride) const{

  Storage<T> data(copy_len,Backend::CPU);

  for(int64_t i = 0;i < copy_len;i++){
    int64_t offset = 0;
    int64_t some = i;
    for(int64_t j = 0;j < stride.size();j++){
      offset += some / stride.at(j) * stride_.at(j);
      some %= stride.at(j);
    }

    data.at(i) = data_.at(offset);
  }

  return data;
}

#define INSTANTIATE(T) \
  template Storage<T> Tensor<T>::to_string_cpu_copy(const int64_t copy_len, \
                                                    const Storage<int64_t> &stride) const;

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE)

}//namespace cobalt_715::nn::tensor