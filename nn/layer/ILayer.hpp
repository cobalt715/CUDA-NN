#pragma once

#include <vector>
#include <cstdint>
#include <random>
#include "nn/tensor/Tensor.hpp"
#include "nn/dtype.hpp"

namespace cobalt_715::nn::layer{

template<nn::mutable_dtype T>
struct ILayer{
  virtual ~ILayer() = default;

  virtual const tensor::Tensor<T>& forward(const tensor::Tensor<T> &input,bool training=true) = 0;

  virtual const tensor::Tensor<T>& backward(const tensor::Tensor<T> &grad_output) = 0;

  virtual void step(const T lr,const int64_t batch_size) = 0;

  virtual void zero_grad() = 0;

  virtual void random_init(std::mt19937 &gen) = 0;

  virtual int64_t data_bytes() const = 0;
};

}//namespace cobalt_715::nn::layer