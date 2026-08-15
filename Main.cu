#include <iostream>
#include <cstdint>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/tensor/Matrix.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage a(12,Backend::CPU);

  for(int i = 0;i < a.size();i++){
    a.at(i) = i;
  }

  tensor::Storage b(16,Backend::CPU);

  for(int i = 0;i < b.size();i++){
    b.at(i) = i / 10.0f;
  }

  tensor::Matrix m1(3,4,std::move(a.toCUDA()));
  tensor::Matrix m2(4,4,std::move(b.toCUDA()));
  tensor::Matrix m3(3,4,Backend::CUDA);

  std::cout << m1 << "\n" << m2 << "\n" << m3 << std::endl;

  tensor::Matrix::matmul(m1,m2,m3);

  std::cout << m3 << std::endl;

  return 0;
}