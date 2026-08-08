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

  std::cout << a << std::endl;

  tensor::Matrix m(3,4,std::move(a));

  std::cout << m << std::endl;

  std::cout << m.toCUDA().t() << std::endl;

  m = m.toCUDA();

  tensor::Matrix::add(m,m,m);

  std::cout << m << std::endl;

  return 0;
}