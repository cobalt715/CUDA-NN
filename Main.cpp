#include <iostream>
#include <vector>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/MatrixView.hpp"
#include "nn/tensor/ops.hpp"
#include "nn/ops/vec.cuh"
#include "nn/ops/gemm.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage<float> as(64,Backend::CPU);
  tensor::Storage<float> bs(64,Backend::CPU);
  tensor::Storage<float> cs(64,Backend::CPU);

  for(int64_t i = 0;i < as.size();i++){
    as.at(i) = i;
  }
  for(int64_t i = 0;i < bs.size();i++){
    bs.at(i) = i * 0.1f;
  }
  for(int64_t i = 0;i < cs.size();i++){
    cs.at(i) = -i;
  }

  tensor::Tensor<float> at({2,4,8},as,Backend::CUDA);
  tensor::Tensor<float> bt({2,4,8},bs,Backend::CUDA);
  tensor::Tensor<float> ct({2,4,8},cs,Backend::CUDA);

  std::cout << at.to_string() << std::endl;
  std::cout << bt.to_string() << std::endl;
  std::cout << ct.to_string() << std::endl;

  tensor::add(at,bt,ct);

  std::cout << at.to_string() << std::endl;
  std::cout << bt.to_string() << std::endl;
  std::cout << ct.to_string() << std::endl;

  std::cout << cs << std::endl;

  return 0;
}