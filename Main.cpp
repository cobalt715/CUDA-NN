#include <iostream>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/MatrixView.hpp"
#include "nn/ops/vec.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage<int64_t> s({1,2,4,7,9},Backend::CPU);
  std::cout << s << std::endl;

  tensor::Tensor t0(s);

  std::cout << t0.to_string() << std::endl;

  tensor::Tensor t1({1,4,2,6});

  return 0;
}