#include <iostream>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/ops/vec.cuh"

using namespace cobalt_715::nn;

int main(){
  const int64_t size = 100000000;

  std::cout << "size:" << size << std::endl;

  auto t0 = std::chrono::high_resolution_clock::now();

  tensor::Storage s0(size,Backend::CPU);
  tensor::Storage s1(size,Backend::CPU);
  tensor::Storage out0(size,Backend::CPU);

  auto t1 = std::chrono::high_resolution_clock::now();

  ops::vec::cpu::add(s0.data(),s1.data(),out0.data(),s0.size());

  auto t2 = std::chrono::high_resolution_clock::now();

  tensor::Storage s2(size,Backend::CUDA);
  tensor::Storage s3(size,Backend::CUDA);
  tensor::Storage out1(size,Backend::CUDA);

  auto t3 = std::chrono::high_resolution_clock::now();

  ops::vec::cuda::add(s2.data(),s3.data(),out1.data(),s0.size());
  //cudaDeviceSynchronize();

  auto t4 = std::chrono::high_resolution_clock::now();

  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::cout << time << std::endl;
  time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cout << time << std::endl;
  time = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  std::cout << time << std::endl;
  time = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
  std::cout << time << std::endl;

  return 0;
}