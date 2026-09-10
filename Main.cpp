#include <iostream>
#include <vector>
#include <chrono>
#include "nn/Backend.hpp"
#include "nn/tensor/Storage.cuh"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/MatrixView.hpp"
#include "nn/ops/vec.cuh"
#include "nn/ops/gemm.cuh"

using namespace cobalt_715::nn;

int main(){
  tensor::Storage<float> a(1024 * 1024,Backend::CPU);
  tensor::Storage<float> b(1024 * 1024,Backend::CPU);
  tensor::Storage<float> out(1024 * 1024,Backend::CPU);

  for(int64_t i = 0;i < a.size();i++){
    a.at(i) = i;
  }

  for(int64_t i = 0;i < b.size();i++){
    b.at(i) = i * 0.1;
  }

  for(int64_t i = 0;i < out.size();i++){
    out.at(i) = -i * 0.5;
  }

  tensor::MatrixView<float> amv(1024,1024,a);
  tensor::MatrixView<float> bmv(1024,1024,b);
  tensor::MatrixView<float> omv(1024,1024,out);

  auto t0 = std::chrono::high_resolution_clock::now();
  tensor::MatrixView<float>::matmul_impl<float,float,float>(1,amv,bmv,0,omv);
  auto t1 = std::chrono::high_resolution_clock::now();

  auto t2 = std::chrono::high_resolution_clock::now();
  tensor::MatrixView<float>::matmul_impl<float,float,float>(1,amv,bmv,0,omv);
  auto t3 = std::chrono::high_resolution_clock::now();

  a = a.toCUDA();
  b = b.toCUDA();
  out = out.toCUDA();

  auto t4 = std::chrono::high_resolution_clock::now();
  tensor::MatrixView<float>::matmul_impl<float,float,float>(1,amv,bmv,0,omv);
  //cudaDeviceSynchronize();
  auto t5 = std::chrono::high_resolution_clock::now();

  auto t6 = std::chrono::high_resolution_clock::now();
  tensor::MatrixView<float>::matmul_impl<float,float,float>(1,amv,bmv,0,omv);
  //cudaDeviceSynchronize();
  auto t7 = std::chrono::high_resolution_clock::now();

  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::cout << "time: " << time << "ms\n";
  time = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  std::cout << "time: " << time << "ms\n";
  time = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();
  std::cout << "time: " << time << "ms\n";
  time = std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count();
  std::cout << "time: " << time << "ms\n";
  std::cout.flush();

  return 0;
}