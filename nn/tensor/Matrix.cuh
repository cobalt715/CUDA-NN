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

//OUT = AB
//16*16*K
__global__ void gemm(const float *a,
                     const int64_t a_row_stride,
                     const int64_t a_col_stride,
                     const float *b,
                     const int64_t b_row_stride,
                     const int64_t b_col_stride,
                     float *out,
                     const int64_t out_row_stride,
                     const int64_t out_col_stride,
                     const int64_t I,
                     const int64_t J,
                     const int64_t K){

  const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;

  const int64_t tx = threadIdx.x;
  const int64_t ty = threadIdx.y;

  __shared__ float a_pack[16][16];
  __shared__ float bt_pack[16][16];

  float sum = 0.0f;

  for(int64_t kk = 0;kk < K;kk += 16){
    if(row < I && kk + tx < K){
      a_pack[ty][tx] =
        a[row * a_row_stride + (kk + tx) * a_col_stride];
    }else{
      a_pack[ty][tx] = 0.0f;
    }

    if(col < J && kk + ty < K){
      bt_pack[tx][ty] =
        b[(kk + ty) * b_row_stride + col * b_col_stride];
    }else{
      bt_pack[tx][ty] = 0.0f;
    }

    __syncthreads();

    for(int64_t k = 0;k < 16;k++){
      sum += a_pack[ty][k] * bt_pack[tx][k];
    }
    __syncthreads();
  }

  if(row < I && col < J){
    out[row * out_row_stride + col * out_col_stride] = sum;
  }

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

  static void matmul(const Matrix &a,const Matrix &b,Matrix &out){
    if(!nn::same_backend(a,b,out)) throw std::runtime_error("Matrix::matmul Backend mismatch");

    if(a.backend() != Backend::CUDA) throw std::runtime_error("qawsedrftgyhujikolp");

    if(out.rows() != a.rows() && out.cols() != b.cols() && a.cols() != b.rows()) throw std::runtime_error("123456789;");

    gemm<<<dim3((out.cols() + 15) / 16,(out.rows() + 15) / 16),dim3(16,16)>>>(a.data().data(),
                                                                              (a.row_con()) ? a.stride():1,
                                                                              (a.row_con()) ? 1:a.stride(),
                                                                              b.data().data(),
                                                                              (b.row_con()) ? b.stride():1,
                                                                              (b.row_con()) ? 1:b.stride(),
                                                                              out.data().data(),
                                                                              (out.row_con()) ? out.stride():1,
                                                                              (out.row_con()) ? 1:out.stride(),
                                                                              out.rows(),
                                                                              out.cols(),
                                                                              a.cols());
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