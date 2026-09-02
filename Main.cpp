#include <iostream>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/tensor/MatrixView.hpp"
#include "nn/ops/vec.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage s(10);

  tensor::MatrixView mv(2,5,s);

  for(int64_t i = 0;i < mv.rows();i++){
    for(int64_t j = 0;j < mv.cols();j++){
      mv.at(i,j) = i + j * 0.1f;
    }
  }

  std::cout << s << std::endl;

  const tensor::Storage<double> cs(7,Backend::CUDA);

  tensor::MatrixView<const double> cmv(2,5,cs);

  std::cout << cs << std::endl;

  std::cout << cmv.dtype() << std::endl;

  return 0;
}