#include <iostream>
#include <vector>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/MatrixView.hpp"
#include "nn/ops/vec.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage<int64_t> s({1,2,4,7,9},Backend::CUDA);
  std::cout << s << std::endl;

  tensor::Tensor t0(s);

  t0.at({0,1,3,4,5});

  std::cout << t0.to_string() << std::endl;

  tensor::Tensor t1({1,4,2,6});

  return 0;
}