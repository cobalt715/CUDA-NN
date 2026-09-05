#include <iostream>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/tensor/MatrixView.hpp"
#include "nn/ops/vec.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage<int32_t> s({1,2,4,7,9},Backend::CUDA);

  std::cout << s << std::endl;

  return 0;
}