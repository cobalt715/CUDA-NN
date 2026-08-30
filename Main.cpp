#include <iostream>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/ops/vec.cuh"

using namespace cobalt_715::nn;

int main(){
  const int64_t size = 1024 * 1024 * 256;

  std::cout << "size:" << size << std::endl;

  tensor::Storage ac(size,Backend::CPU);
  tensor::Storage bc(size,Backend::CPU);
  tensor::Storage outc(size,Backend::CPU);

  outc.at(0) = 10.0f;

  for(int64_t i = 0;i < size;i++){
    ac.at(i) = (i - size / 2.0f) / size;
    bc.at(i) = i * 10000 / size;
  }

  std::cout << ac.to_string(5) << std::endl;
  std::cout << bc.to_string(5) << std::endl;
  std::cout << outc.to_string(5) << std::endl;

  auto t0 = std::chrono::high_resolution_clock::now();
  ops::vec::cpu::dot(ac.data(),bc.data(),outc.data(),ac.size());
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout << outc.to_string(5) << std::endl;

  outc.at(0) = 100.0f;

  tensor::Storage ag = ac.toCUDA();
  tensor::Storage bg = bc.toCUDA();
  tensor::Storage outg = outc.toCUDA();

  std::cout << ag.to_string(5) << std::endl;
  std::cout << bg.to_string(5) << std::endl;
  std::cout << outg.to_string(5) << std::endl;

  auto t2 = std::chrono::high_resolution_clock::now();
  ops::vec::cuda::dot(ag.data(),bg.data(),outg.data(),ag.size());
  cudaDeviceSynchronize();
  auto t3 = std::chrono::high_resolution_clock::now();

  std::cout << outg.to_string(5) << std::endl;

  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::cout << time << std::endl;
  time = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  std::cout << time << std::endl;

  return 0;
}