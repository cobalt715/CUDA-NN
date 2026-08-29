//nvcc -I. -std=c++20 *.cpp nn/ops/*.cpp nn/ops/*.cu -o main
//nvcc -DCOBALT_715_USE_CUDA -I. -std=c++20 *.cpp nn/ops/*.cpp nn/ops/*.cu -o main
//g++ -I. -std=c++20 *.cpp nn/ops/*.cpp nn/ops/*.cu -o main

#include <iostream>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/ops/vec.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage s0(20,Backend::CPU);
  tensor::Storage s1(20,Backend::CPU);

  for(int i = 0;i < s0.size();i++){
    s0.at(i) = i;
    s1.at(i) = i / 10.0f;
  }

  std::cout << s0 << std::endl;
  std::cout << s1 << std::endl;

  tensor::Storage out(20,Backend::CPU);

  //ops::vec::cpu::add(s0.data(),s1.data(),out.data(),s0.size());

  std::cout << out << std::endl;

  s0 = s0.toCUDA();
  s1 = s1.toCUDA();
  out = out.toCUDA();

  ops::vec::cuda::add(s0.data(),s1.data(),out.data(),s0.size());

  std::cout << out << std::endl;

  return 0;
}