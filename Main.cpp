#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/MatrixView.hpp"
#include "nn/tensor/ops.hpp"
#include "nn/layer/ILayer.hpp"
#include "nn/layer/Dense.hpp"

using namespace cobalt_715::nn;

int main(){
  std::mt19937 gen(0);

  tensor::Tensor<double> input({4,2},{0,0,0,1000,1000,0,1000,1000},Backend::CPU);
  tensor::Tensor<double> output({4,1},{0,1000,1000,0},Backend::CPU);
  tensor::Tensor<double> some({4,1},Backend::CPU);


  std::cout << input.to_string() << std::endl;
  std::cout << output.to_string() << std::endl;

  layer::Dense<double> d0(2,3);
  layer::Dense<double> d1(3,1);

  d0.random_init(gen);
  d1.random_init(gen);

  for(int64_t i = 0;i < 100000;i++){
    const tensor::Tensor<double> &out = d1.forward(d0.forward(input));

    std::cout << out.to_string() << std::endl;

    tensor::sub(out,output,some);

    d0.backward(d1.backward(some));

    d0.step(0.000001,1);
    d1.step(0.000001,1);

    d0.zero_grad();
    d1.zero_grad();
  }

  return 0;
}