#include <iostream>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/ops/vec.cuh"

using namespace cobalt_715::nn;

int main(){
  const int64_t size = 1024 * 1024 * 1;

  std::cout << "size:" << size << std::endl;

  auto t0 = std::chrono::high_resolution_clock::now();
  tensor::Storage ac(size,Backend::CPU);
  tensor::Storage bc(size,Backend::CPU);
  tensor::Storage outc(size,Backend::CPU);
  auto t1 = std::chrono::high_resolution_clock::now();

  outc.at(0) = 10.0f;

  for(int64_t i = 0;i < size;i++){
    ac.at(i) = (i - size / 2.0f) / size;
    bc.at(i) = i * 10000 / size;
  }

  std::cout << ac.to_string(5) << std::endl;
  std::cout << bc.to_string(5) << std::endl;
  std::cout << outc.to_string(5) << std::endl;

  auto t2 = std::chrono::high_resolution_clock::now();
  ops::vec::cpu::add(ac.data(),bc.data(),outc.data(),ac.size());
  auto t3 = std::chrono::high_resolution_clock::now();

  std::cout << outc.to_string(5) << std::endl;

  outc.at(0) = 100.0f;

  auto t4 = std::chrono::high_resolution_clock::now();
  tensor::Storage ag(size,Backend::CUDA);
  tensor::Storage bg(size,Backend::CUDA);
  tensor::Storage outg(size,Backend::CUDA);
  auto t5 = std::chrono::high_resolution_clock::now();

  ag = ac.toCUDA();
  bg = bc.toCUDA();
  outg = outc.toCUDA();

  std::cout << ag.to_string(5) << std::endl;
  std::cout << bg.to_string(5) << std::endl;
  std::cout << outg.to_string(5) << std::endl;

  auto t6 = std::chrono::high_resolution_clock::now();
  ops::vec::cuda::add(ag.data(),bg.data(),outg.data(),ag.size());
  cudaDeviceSynchronize();
  auto t7 = std::chrono::high_resolution_clock::now();

  std::cout << outg.to_string(5) << std::endl;

  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::cout << "cpu malloc:" << time << "ms" << std::endl;
  time = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  std::cout << "cpu compute:" << time << "ms" << std::endl;
  time = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();
  std::cout << "cuda malloc:" << time << "ms" << std::endl;
  time = std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count();
  std::cout << "cuda compute:" << time << "ms" << std::endl;

  return 0;
}