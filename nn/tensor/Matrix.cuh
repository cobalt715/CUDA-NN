#pragma once

#include <cuda_runtime.h>
#include <string>
#include <cstdint>
#include "Storage.cuh"
#include "nn/Backend.hpp"
#include "nn/ops/vec_fwd.hpp"

namespace cobalt_715::nn::tensor{

namespace op{

struct add{
  __forceinline__ __host__ __device__ float operator()(const float a,const float b) const{
    return a + b;
  }
};

struct sub{
  __forceinline__ __host__ __device__ float operator()(const float a,const float b) const{
    return a - b;
  }
};

struct mul{
  __forceinline__ __host__ __device__ float operator()(const float a,const float b) const{
    return a * b;
  }
};

struct div{
  __forceinline__ __host__ __device__ float operator()(const float a,const float b) const{
    return a / b;
  }
};

}//namespace op

template<class T>
__global__ void cuda_matrix_binary_alias_safe(const int64_t rows,
                                              const int64_t cols,
                                              const float *a,
                                              const int64_t a_stride,
                                              const bool a_bool,
                                              const float *b,
                                              const int64_t b_stride,
                                              const bool b_bool,
                                              float *out,
                                              const int64_t out_stride,
                                              const bool out_bool){

  const int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= cols || y >= rows) return;

  T t;

  out[out_bool ? y * out_stride + x:x * out_stride + y] = t(a[a_bool ? y * a_stride + x:x * a_stride + y],b[b_bool ? y * b_stride + x:x * b_stride + y]);
}

struct Matrix{
  Storage data_;
  bool row_con_;
  int64_t rows_;
  int64_t cols_;
  int64_t stride_;

  Matrix(int64_t rows,int64_t cols,Backend backend,bool row_con = true)
    : data_(rows * cols,backend),
      row_con_(row_con),
      rows_(rows),
      cols_(cols),
      stride_(row_con ? cols:rows){}

  Matrix(int64_t rows,int64_t cols,Storage &&data,bool row_con = true)
    : data_(std::move(data)),
      row_con_(row_con),
      rows_(rows),
      cols_(cols),
      stride_(row_con ? cols:rows){}

  __host__ __device__ Storage& data(){
    return data_;
  }

  __host__ __device__ const Storage& data() const{
    return data_;
  }

  Backend backend() const{
    return data_.backend();
  }

  __host__ __device__ bool row_con() const{
    return row_con_;
  }

  __host__ __device__ int64_t rows() const{
    return rows_;
  }

  __host__ __device__ int64_t cols() const{
    return cols_;
  }

  __host__ __device__ int64_t stride() const{
    return stride_;
  }

  __host__ __device__ float& at(const int64_t row,const int64_t col){
    return row_con_ ? data_.at(row * stride_ + col):data_.at(col * stride_ + row);
  }

  __host__ __device__ const float& at(const int64_t row,const int64_t col) const{
    return row_con_ ? data_.at(row * stride_ + col):data_.at(col * stride_ + row);
  }

  template<typename M,typename... Ms>
  inline static bool same_shape(const M &m,const Ms&... ms){
    return ((m.rows() == ms.rows() && m.cols() == ms.cols()) && ...);
  }

  static void add(const Matrix &a,const Matrix &b,Matrix &out){
    #ifndef NDEBUG
      if(!nn::same_backend(a,b,out)) throw std::runtime_error("Matrix::add Backend mismatch");

      if(!same_shape(a,b,out)) throw std::runtime_error("Matrix::add shape mismatch");
    #endif

    if(a.backend() == Backend::CPU){
      for(int64_t i = 0;i < a.rows();i++){
        for(int64_t j = 0;j < a.cols();j++){
          out.at(i,j) = a.at(i,j) + b.at(i,j);
        }
      }
    }else if(a.backend() == Backend::CUDA){
      cuda_matrix_binary_alias_safe<op::add><<<dim3((a.cols() + 15) / 16,(a.rows() + 15) / 16),dim3(16,16)>>>(a.rows(),a.cols(),
                                                                                                                            a.data().data(),a.stride(),a.row_con(),
                                                                                                                            b.data().data(),b.stride(),b.row_con(),
                                                                                                                            out.data().data(),out.stride(),out.row_con());

      cudaError_t err = cudaGetLastError();
      if(err != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(err));
      }
    }
  }

  Matrix& t(){
    row_con_ = !row_con_;

    std::swap(rows_,cols_);

    return *this;
  }

  Matrix toCPU() const{
    return Matrix(rows_,cols_,data_.toCPU(),row_con_);
  }

  Matrix toCUDA() const{
    return Matrix(rows_,cols_,data_.toCUDA(),row_con_);
  }

  std::string to_string() const{
    std::string s = "Matrix " + std::to_string(rows_) + " , " + std::to_string(cols_)
                    + "\n" + std::to_string(row_con_) + " , " + std::to_string(stride_)
                    + "\n" + std::string(nn::to_string(backend()))
                    + "\n{\n";

    Storage st  = data_.toCPU();

    for(int64_t i = 0;i < rows_;i++){
      s += "  { ";
      for(int64_t j = 0;j < cols_;j++){
        s += std::to_string(row_con_ ? st.at(i * stride_ + j):st.at(j * stride_ + i)) + " , ";
      }
      s += "}\n";
    }

    s += "}";

    return s;
  }
};

inline std::ostream& operator<<(std::ostream &o,const Matrix &m){
  return o << m.to_string();
}

}//namespace cobalt_715::nn::tensor