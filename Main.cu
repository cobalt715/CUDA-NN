#include <iostream>
#include <cstdint>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage s(10,Backend::CUDA);

  std::cout << s << std::endl;

  s.fill(10.0f);

  std::cout << s << std::endl;

  return 0;
}