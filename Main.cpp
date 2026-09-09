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
  {
  tensor::Storage<int64_t> a(10,Backend::CPU);
  tensor::Storage<int64_t> b(10,Backend::CPU);
  tensor::Storage<int64_t> out(1,Backend::CPU);

  for(int i = 0;i < a.size();i++){
    a.at(i) = i;
    b.at(i) = i + 3;
  }

  std::cout << "a:" << a << std::endl;
  std::cout << "b:" << b << std::endl;

  ops::vec::cpu::dot(a.data(),b.data(),out.data(),a.size());

  std::cout << out << std::endl;

  a = a.toCUDA();
  b = b.toCUDA();
  tensor::Storage<int64_t> outc(1,Backend::CUDA);

  ops::vec::cuda::dot(a.data(),b.data(),outc.data(),a.size());

  std::cout << outc << std::endl;
  }

  tensor::Storage<int32_t> a(10,Backend::CPU);
  tensor::Storage<int32_t> b(10,Backend::CPU);
  tensor::Storage<int32_t> out(10,Backend::CPU);

  for(int i = 0;i < a.size();i++){
    a.at(i) = i;
    b.at(i) = (i + 1) * 1.1f;
  }

  std::cout << "a:" << a << std::endl;
  std::cout << "b:" << b << std::endl;

  ops::vec::cpu::add(a.data(),b.data(),out.data(),a.size());
  std::cout << out << std::endl;
  ops::vec::cpu::sub(a.data(),b.data(),out.data(),a.size());
  std::cout << out << std::endl;
  ops::vec::cpu::mul(a.data(),b.data(),out.data(),a.size());
  std::cout << out << std::endl;
  ops::vec::cpu::div(a.data(),b.data(),out.data(),a.size());
  std::cout << out << std::endl;

  tensor::Storage<double> ac(10,Backend::CPU);
  tensor::Storage<double> bc(10,Backend::CPU);
  tensor::Storage<double> outc(10,Backend::CUDA);

  for(int i = 0;i < ac.size();i++){
    ac.at(i) = i;
    bc.at(i) = i * 11;
  }

  ac = ac.toCUDA();
  bc = bc.toCUDA();

  std::cout << "\nac:" << ac << std::endl;
  std::cout << "bc:" << bc << std::endl;

  ops::vec::cuda::add(ac.data(),bc.data(),outc.data(),ac.size());
  std::cout << outc << std::endl;
  ops::vec::cuda::sub(ac.data(),bc.data(),outc.data(),ac.size());
  std::cout << outc << std::endl;
  ops::vec::cuda::mul(ac.data(),bc.data(),outc.data(),ac.size());
  std::cout << outc << std::endl;
  ops::vec::cuda::div(ac.data(),bc.data(),outc.data(),ac.size());
  std::cout << outc << std::endl;

  return 0;
}