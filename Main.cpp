#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/MatrixView.hpp"
#include "nn/tensor/ops.hpp"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage<float> as(64,Backend::CPU);
  tensor::Storage<float> bs(64,Backend::CPU);
  tensor::Storage<float> cs(64,Backend::CPU);

  for(int64_t i = 0;i < as.size();i++){
    as.at(i) = i;
  }
  for(int64_t i = 0;i < bs.size();i++){
    bs.at(i) = i * 0.1f;
  }
  for(int64_t i = 0;i < cs.size();i++){
    cs.at(i) = -i;
  }

  as = as.toCUDA();
  bs = bs.toCUDA();
  cs = cs.toCUDA();

  tensor::MatrixView<float> a(2,8,16,2,as);
  tensor::MatrixView<float> b(4,8,16,2,bs);
  tensor::MatrixView<float> c(4,8,16,2,cs);

  tensor::add(a,b,c);

  std::cout << a.to_string() << std::endl;
  std::cout << b.to_string() << std::endl;
  std::cout << c.to_string() << std::endl;

  std::cout << cs.to_string() << std::endl;

  return 0;
}