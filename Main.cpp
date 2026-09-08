#include <iostream>
#include <vector>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/MatrixView.hpp"
#include "nn/ops/vec.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage<double> data(3 * 2 * 4,Backend::CPU);
  for(int64_t i = 0;i < data.size();i++) data.at(i) = i;

  //data = data.toCUDA();

  std::cout << data << std::endl;

  tensor::MatrixView<double> mv(6,4,data);

  std::cout << mv << std::endl;

  tensor::Storage<int64_t> s({3,2,4},Backend::CPU);
  std::cout << s << std::endl;

  tensor::Tensor t0(s,data);

  std::cout << t0.to_string({0}) << std::endl;

  std::cout << t0 << std::endl;

  std::cout << t0.to_string({2,1,3}) << std::endl;

  return 0;
}