#pragma once

#include <vector>
#include <cstdint>
#include <random>
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/ops.hpp"
#include "nn/dtype.hpp"

namespace cobalt_715::nn::layer{

template<nn::mutable_dtype T>
struct Dense{
  Dense(const int64_t in,const int64_t out)
    : W_(tensor::Storage<int64_t>({in,out})),
      b_(tensor::Storage<int64_t>({out})),
      z_(tensor::Storage<int64_t>({1,1})),
      a_(tensor::Storage<int64_t>({1,1})),
      dW_(tensor::Storage<int64_t>({in,out})),
      db_(tensor::Storage<int64_t>({out})),
      delta_(tensor::Storage<int64_t>({1,1})),
      grad_(tensor::Storage<int64_t>({1,1})){}

  const tensor::Tensor<T> *input_ptr_;//逆伝播で必要なため
  tensor::Tensor<T> W_,b_;
  tensor::Tensor<T> z_,a_;
  tensor::Tensor<T> dW_,db_;
  tensor::Tensor<T> delta_,grad_;

  const tensor::Tensor<T>& forward(const tensor::Tensor<T> &input,bool training=true){
    input_ptr_ = &input;

    //サイズが違うときだけ再確保
    if(input.rank() != z_.rank() || !std::equal(input.shape().begin(),input.shape().end() - 1,z_.shape().begin()) || W_.dim(W_.rank() - 1) != z_.dim(z_.rank() - 1)){
      tensor::Storage<int64_t> output_shape = input.shape().toCPU();

      output_shape.at(output_shape.size() - 1) = W_.dim(W_.rank() - 1);

      z_ = tensor::Tensor<T>(output_shape);
      a_ = tensor::Tensor<T>(output_shape);
    }

    tensor::MatrixView<T> z_view = z_.flatten_matrix_view();

    tensor::matmul_impl(1,input.flatten_matrix_view(),W_.flatten_matrix_view(),0,z_view);

    for(int64_t row = 0;row < z_.numel() / z_.dim(z_.rank() - 1);row++) {
      for(int64_t col = 0;col < z_.dim(z_.rank() - 1);col++) {
        z_.data()[row * z_.dim(z_.rank() - 1) + col] += b_.data()[col];
      }
    }

    for(int64_t i = 0;i < z_.numel();i++){
      a_.data()[i] = (z_.data()[i] > 0) ? z_.data()[i]:z_.data()[i] / static_cast<T>(100);
    }

    return a_;
  }

  const tensor::Tensor<T>& backward(const tensor::Tensor<T> &grad_output){
    if(grad_output.rank() != delta_.rank() || !std::equal(grad_output.shape().begin(),grad_output.shape().end() - 1,delta_.shape().begin()))
      delta_ = tensor::Tensor<T>(grad_output.shape());

    const int64_t rows = delta_.numel() / delta_.dim(delta_.rank() - 1);
    const int64_t cols = delta_.dim(delta_.rank() - 1);

    for(int64_t row = 0;row < rows;row++){
      const int64_t front = row * cols;
      for(int64_t col = 0;col < cols;col++){
        const int64_t index = front + col;
        delta_.data()[index] = grad_output.data()[index] * ((z_.data()[index] > 0) ? 1:1 / static_cast<T>(100));

        db_.data()[col] += delta_.data()[index];
      }
    }

    if(grad_.rank() != input_ptr_->rank() || !std::equal(grad_.shape().begin(),grad_.shape().end() - 1,input_ptr_->shape().begin()))
      grad_ = tensor::Tensor<T>(input_ptr_->shape());

    const tensor::MatrixView<const T> input_view = input_ptr_->flatten_matrix_view();
    const tensor::MatrixView<T> W_view = W_.flatten_matrix_view();
    const tensor::MatrixView<T> delta_view = delta_.flatten_matrix_view();

    tensor::MatrixView<T> dW_view = dW_.flatten_matrix_view();
    tensor::MatrixView<T> grad_view = grad_.flatten_matrix_view();

    tensor::matmul_impl(1,input_view.t(),delta_view,1,dW_view);

    tensor::matmul_impl(1,delta_view,W_view.t(),0,grad_view);

    return grad_;
  }

  void step(const T lr,const int64_t batch_size){
    for(int64_t i = 0;i < W_.numel();i++){
      W_.data()[i] -= dW_.data()[i] * lr;
    }

    for(int64_t i = 0;i < b_.numel();i++){
      b_.data()[i] -= db_.data()[i] * lr;
    }
  }

  void zero_grad(){
    for(int64_t i = 0;i < dW_.numel();i++){
      dW_.data()[i] = 0;
    }

    for(int64_t i = 0;i < db_.numel();i++){
      db_.data()[i] = 0;
    }
  }

  virtual void random_init(std::mt19937 &gen){
    T limit = sqrt(6 / (W_.shape().at(0) + W_.shape().at(1)));
    std::uniform_real_distribution<T> dist(-limit,limit);

    T *Wd = W_.data();

    for(int64_t i = 0;i < W_.numel();i++){
      Wd[i] = dist(gen);
    }
  }

  int64_t data_bytes() const{
    return 0;
  }
};

}//namespace cobalt_715::nn::layer