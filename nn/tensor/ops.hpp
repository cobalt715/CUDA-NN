//Tensor,MatrixViewの演算をまとめる
#pragma once

#include <cstdint>
#include <type_traits>
#include "Storage.cuh"
#include "Tensor.hpp"
#include "MatrixView.hpp"
#include "nn/Backend.hpp"
#include "nn/dtype.hpp"
#include "nn/ops/vec.cuh"
#include "nn/ops/matrix.cuh"
#include "nn/ops/gemm.cuh"

namespace cobalt_715::nn::tensor{

//const Storage<int64_t>& shape()があるかどうか
template<class T>
concept HasShape =
requires(const T &x){
  { x.shape() } -> std::same_as<const Storage<int64_t>&>;
};

template<HasShape T,HasShape... Ts>
bool same_shape(const T& first,const Ts&... rest){
  bool (*same)(const Storage<int64_t> &a,const Storage<int64_t> &b) =
    [](const Storage<int64_t> &a,const Storage<int64_t> &b){
      if(a.size() != b.size()) return false;

      for(int64_t i = 0;i < a.size();i++){
        if(a.at(i) != b.at(i)) return false;
      }

      return true;
    };

  return ((same(rest.shape(),first.shape())) && ...);
}

template<class T>
concept HasRowsCols =
requires(const T &x){
  { x.rows() } -> std::same_as<int64_t>;
  { x.cols() } -> std::same_as<int64_t>;
};

template<HasRowsCols T,HasRowsCols... Ts>
bool same_shape(const T& first,const Ts&... rest){
  return ((rest.rows() == first.rows() && rest.cols() == first.cols()) && ...);
}

template<nn::mutable_dtype T>
void add(const Tensor<T> &a,const Tensor<T> &b,Tensor<T> &out){
  if(!nn::same_backend(a,b,out)) throw std::invalid_argument("tensor::ops::add backend mismatch");
  if(!same_shape(a,b,out)) throw std::invalid_argument("tensor::ops::add shape mismatch");

  if(a.backend() == Backend::CPU){
    nn::ops::vec::cpu::add(a.data(),b.data(),out.data(),a.numel());
  }else if(a.backend() == Backend::CUDA){
    nn::ops::vec::cuda::add(a.data(),b.data(),out.data(),a.numel());
  }
}

template<nn::mutable_dtype T>
void sub(const Tensor<T> &a,const Tensor<T> &b,Tensor<T> &out){
  if(!nn::same_backend(a,b,out)) throw std::invalid_argument("tensor::ops::sub backend mismatch");
  if(!same_shape(a,b,out)) throw std::invalid_argument("tensor::ops::sub shape mismatch");

  if(a.backend() == Backend::CPU){
    nn::ops::vec::cpu::sub(a.data(),b.data(),out.data(),a.numel());
  }else if(a.backend() == Backend::CUDA){
    nn::ops::vec::cuda::sub(a.data(),b.data(),out.data(),a.numel());
  }
}

template<nn::mutable_dtype T>
void mul(const Tensor<T> &a,const Tensor<T> &b,Tensor<T> &out){
  if(!nn::same_backend(a,b,out)) throw std::invalid_argument("tensor::ops::mul backend mismatch");
  if(!same_shape(a,b,out)) throw std::invalid_argument("tensor::ops::mul shape mismatch");

  if(a.backend() == Backend::CPU){
    nn::ops::vec::cpu::mul(a.data(),b.data(),out.data(),a.numel());
  }else if(a.backend() == Backend::CUDA){
    nn::ops::vec::cuda::mul(a.data(),b.data(),out.data(),a.numel());
  }
}

template<nn::mutable_dtype T>
void div(const Tensor<T> &a,const Tensor<T> &b,Tensor<T> &out){
  if(!nn::same_backend(a,b,out)) throw std::invalid_argument("tensor::ops::div backend mismatch");
  if(!same_shape(a,b,out)) throw std::invalid_argument("tensor::ops::div shape mismatch");

  if(a.backend() == Backend::CPU){
    nn::ops::vec::cpu::div(a.data(),b.data(),out.data(),a.numel());
  }else if(a.backend() == Backend::CUDA){
    nn::ops::vec::cuda::div(a.data(),b.data(),out.data(),a.numel());
  }
}

template<nn::dtype U,nn::dtype V,nn::mutable_dtype W>
void add(const MatrixView<U> &a,const MatrixView<V> &b,MatrixView<W> &out){
  //U,V,Wがすべて同じ型
  static_assert(
    std::is_same_v<std::remove_const_t<U>,std::remove_const_t<V>>
    &&
    std::is_same_v<std::remove_const_t<V>,std::remove_const_t<W>>
  );

  if(!nn::same_backend(a,b,out)) throw std::invalid_argument("tensor::ops::add backend mismatch");
  if(!same_shape(a,b,out)) throw std::invalid_argument("tensor::ops::add shape mismatch");

  if(a.backend() == Backend::CPU){
    nn::ops::matrix::cpu::add(a.data().data() + a.offset(),a.row_stride(),a.col_stride(),
                              b.data().data() + b.offset(),b.row_stride(),b.col_stride(),
                              out.data().data() + out.offset(),out.row_stride(),out.col_stride(),
                              a.rows(),a.cols());
  }else if(a.backend() == Backend::CUDA){
    nn::ops::matrix::cuda::add(a.data().data() + a.offset(),a.row_stride(),a.col_stride(),
                               b.data().data() + b.offset(),b.row_stride(),b.col_stride(),
                               out.data().data() + out.offset(),out.row_stride(),out.col_stride(),
                               a.rows(),a.cols());
  }
}

//out = alpha * ab + beta * out
template<nn::dtype U,nn::dtype V,nn::mutable_dtype W>
void matmul_impl(const std::type_identity_t<U> alpha,
                 const MatrixView<U> &a,
                 const MatrixView<V> &b,
                 const std::type_identity_t<U> beta,
                 MatrixView<W> &out){

  //U,V,Wがすべて同じ型
  static_assert(
    std::is_same_v<std::remove_const_t<U>,std::remove_const_t<V>>
    &&
    std::is_same_v<std::remove_const_t<V>,std::remove_const_t<W>>
  );

  #ifndef NDEBUG
    if(!nn::same_backend(a,b,out)) throw std::runtime_error("tensor::MatrixView::matmul_impl backend not same");

    if(a.cols() != b.rows() || out.rows() != a.rows() || out.cols() != b.cols()) throw std::invalid_argument("Matrix::matmul dimension mismatch");

    if(out.is_overlapped()) throw std::runtime_error("tensor::MatrixView::matmul out is overlapped");
  #endif

  if(a.backend() == Backend::CPU){
    ops::cpu::gemm_impl(alpha,
                        a.data().data() + a.offset(),a.row_stride(),a.col_stride(),
                        b.data().data() + b.offset(),b.row_stride(),b.col_stride(),
                        beta,
                        out.data().data() + out.offset(),out.row_stride(),out.col_stride(),
                        out.rows(),out.cols(),a.cols());
  }else if(a.backend() == Backend::CUDA){
    ops::cuda::gemm_impl(alpha,
                         a.data().data() + a.offset(),a.row_stride(),a.col_stride(),
                         b.data().data() + b.offset(),b.row_stride(),b.col_stride(),
                         beta,
                         out.data().data() + out.offset(),out.row_stride(),out.col_stride(),
                         out.rows(),out.cols(),a.cols());
  }
}

}//namespace cobalt_715::nn::tensor