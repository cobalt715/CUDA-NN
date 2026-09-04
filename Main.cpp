#include <iostream>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/tensor/MatrixView.hpp"
#include "nn/ops/vec.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage arr(64);

  for(int64_t i = 0;i < arr.size();i++){
    arr.at(i) = i;
  }

  std::cout << arr.to_string(0) << std::endl;

  tensor::MatrixView arrv(8,8,arr);

  std::cout << arrv << std::endl;
  std::cout << "88" << arrv.to_string(8,8) << std::endl;
  std::cout << "80" << arrv.to_string(8,0) << std::endl;
  std::cout << "08" << arrv.to_string(0,8) << std::endl;
  std::cout << "00" << arrv.to_string(0,0) << std::endl;

  tensor::Storage no(0);

  std::cout << no << std::endl;

  tensor::MatrixView nov(0,0,no);

  std::cout << nov << std::endl;

  auto nov1 = tensor::MatrixView(0,8,no);

  std::cout << nov1 << std::endl;

  auto nov2 = tensor::MatrixView(8,0,no);

  std::cout << nov2 << std::endl;

  tensor::Storage s(10);

  tensor::MatrixView mv(2,5,s);

  for(int64_t i = 0;i < mv.rows();i++){
    for(int64_t j = 0;j < mv.cols();j++){
      mv.at(i,j) = i + j * 0.1f;
    }
  }

  std::cout << s << std::endl;

  std::cout << mv << std::endl;

  const tensor::Storage<double> cs(7,Backend::CUDA);

  tensor::MatrixView<const double> cmv(2,5,cs);

  std::cout << cs << std::endl;

  std::cout << cmv << std::endl;

  std::cout << cmv.dtype() << std::endl;

  tensor::Storage cu88 = arr.to(Backend::CPU);

  std::cout << cu88 << std::endl;

  tensor::MatrixView cu88v(8,8,cu88);

  std::cout << cu88v << std::endl;

  return 0;
}