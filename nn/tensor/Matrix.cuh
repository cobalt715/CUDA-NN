#pragma once

#include <string>
#include <cstdint>
#include "Storage.cuh"
#include "nn/Backend.hpp"

namespace cobalt_715::nn::tensor{

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

  void t(){
    row_con_ = !row_con_;
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