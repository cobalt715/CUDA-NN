#include <iostream>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage s(10,Backend::CPU);

  for(int i = 0;i < s.size();i++){
    s.at(i) = i + 1;
  }

  std::cout << s << std::endl;
  std::cout << s.to_string(7) << std::endl;

  tensor::Storage s0(0,Backend::CPU);

  for(int i = 0;i < s0.size();i++){
    s0.at(i) = i + 1;
  }

  std::cout << s0 << std::endl;
  std::cout << s0.to_string(7) << std::endl;

  tensor::Storage s1 = s.toCUDA();
  tensor::Storage s01 = s0.toCUDA();

  std::cout << s1 << std::endl;
  std::cout << s1.to_string(7) << std::endl;
  std::cout << s01 << std::endl;
  std::cout << s01.to_string(7) << std::endl;

  s.at(2) = 100;
  std::cout << s << std::endl;
  std::cout << s.toCUDA() << std::endl;

  return 0;
}