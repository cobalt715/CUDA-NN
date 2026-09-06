#pragma once

#include <string>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <ostream>
#include "nn/Backend.hpp"
#include "nn/cuda/config.cuh"
#include "nn/cuda/util.cuh"
#include "Storage.cuh"

namespace cobalt_715::nn::tensor{

//任意次元テンソル
template<nn::dtype T=float>
struct Tensor{
  //templateがconstだとエラーを投げる
  static_assert(!std::is_const_v<T>,"tensor::Tensor T must not be const");

public:
  Tensor(const Storage<int64_t> &shape,Backend backend=Backend::CPU)
    : data_(0),
      shape_(shape.toCPU()),
      device_shape_(shape.to(backend)),
      stride_(0),
      device_stride_(0){

    data_ = Storage(calculate_data_size(shape_),backend);

    set_stride();
  }

  inline T& at(const Storage<int64_t> &index){
    check_index(index);

    int64_t i = 0;
    for(int64_t i = 0;i < shape_.size();i++){
      i += shape_.at(i) * stride_.at(i);
    }

    return data_.at(i);
  }

  inline T& at(const Storage<int64_t> &index) const{
    check_index(index);

    int64_t i = 0;
    for(int64_t i = 0;i < shape_.size();i++){
      i += shape_.at(i) * stride_.at(i);
    }

    return data_.at(i);
  }

  std::string to_string() const{
    return to_string(shape_);
  }

  std::string to_string(const Storage<int64_t> &range) const{
    std::string text = "tensor::Tensor(shape_={";

    for(int64_t i = 0;i < shape_.size();i++){
      if(i != 0) text += ", ";
      text += std::to_string(shape_.at(i));
    }

    text += "},\n               stride_={";

    for(int64_t i = 0;i < stride_.size();i++){
      if(i != 0) text += ", ";
      text += std::to_string(stride_.at(i));
    }

    text += "},\n               backend_="
             + nn::to_string(data_.backend())
             + ",\n               dtype_="
             + nn::dtype_name<T>();

    return text;
  }

private:
  Storage<T> data_;
  Storage<int64_t> shape_;
  Storage<int64_t> stride_;
  Storage<int64_t> device_shape_;
  Storage<int64_t> device_stride_;

  //data_のsizeを計算する
  inline static int64_t calculate_data_size(const Storage<int64_t>& shape){
    int64_t data_size = 1;
    for(int64_t i = 0;i < shape.size();i++){
      data_size *= shape.at(i);
    }

    return data_size;
  }

  inline void set_stride(){
    stride_ = Storage<int64_t>(shape_.size(),Backend::CPU);

    int64_t base = 1;
    for(int64_t i = shape_.size();i-- > 0;){
      stride_.at(i) = base;
      base *= shape_.at(i);
    }

    device_stride_ = stride_.to(data_.backend());
  }

  //indexが正しいか確認する
  inline void check_index(const Storage<int64_t>& index) const{
    //#ifndef NDEBUG
    if(shape_.size() != index.size())
      throw std::runtime_error("tensor::Tensor check_index dimension mismatch");

    for(int64_t i = 0;i < index.size();i++){
      const T value = index.at(i);
      if(value < 0 || shape_.at(i) <= value)
        throw std::runtime_error("Tensor index out of bounds at dim");
    }
    //#endif
  }
};

template<dtype T>
inline std::ostream& operator<<(std::ostream &o,const Tensor<T> &t){
  return o << t.to_string();
}

}//namespace cobalt_715::nn::tensor