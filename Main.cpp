//nvcc -I. -std=c++20 *.cpp nn/ops/*.cpp nn/ops/*.cu -o main
//nvcc -DCOBALT_715_USE_CUDA -I. -std=c++20 *.cpp nn/ops/*.cpp nn/ops/*.cu -o main
//g++ -I. -std=c++20 *.cpp nn/ops/*.cpp -o main

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

  std::cout << "0" << s0 << std::endl;
  std::cout << "1" << s1 << std::endl;

  tensor::Storage out(20,Backend::CPU);

  ops::vec::cpu::add(s0.data(),s1.data(),out.data(),s0.size());

  std::cout << "2" << out << std::endl;

  tensor::Storage s01 = s0.toCUDA();
  tensor::Storage s11 = s1.toCUDA();
  tensor::Storage out1(20,Backend::CUDA);

  std::cout << "3" << s01 << std::endl;
  std::cout << "4" << s11 << std::endl;

  ops::vec::cuda::sub(s01.data(),s11.data(),out1.data(),s0.size());

  std::cout << "5" << out1 << std::endl;

  return 0;
}