#include <iostream>
#include <cstdint>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage a(10,Backend::CPU);

  for(int i = 0;i < a.size();i++){
    a.at(i) = i;
  }

  std::cout << a << std::endl;

  tensor::Storage b(10,Backend::CPU);

  for(int i = 0;i < b.size();i++){
    b.at(i) = i / 10.0f;
  }

  std::cout << b << std::endl;

  tensor::Storage out(10,Backend::CUDA);

  tensor::Storage::add(a.toCUDA(),b.toCUDA(),out);
  //tensor::Storage::add(a,b,out);

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << out << std::endl;

  return 0;
}