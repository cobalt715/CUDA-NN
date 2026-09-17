#pragma once

#include <string>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <ostream>
#include <cassert>
#include "nn/Backend.hpp"
#include "nn/dtype.hpp"
#include "nn/cuda/config.cuh"
#include "nn/cuda/util.cuh"
#include "Storage.cuh"
#include "MatrixView.hpp"

namespace cobalt_715::nn::tensor{

//任意次元テンソル
template<nn::mutable_dtype T=float>
struct Tensor{
public:
  Tensor(const Storage<int64_t> &shape,Backend backend=Backend::CPU)
    : data_(0),
      shape_(shape.toCPU()),
      device_shape_(shape.to(backend)),
      stride_(0),
      device_stride_(0){

    data_ = Storage<T>(calc_data_size(shape_),backend);

    set_stride();
  }

  Tensor(const Storage<int64_t> &shape,const Storage<T> &data,Backend backend=Backend::CPU)
    : data_(data.to(backend)),
      shape_(shape.toCPU()),
      device_shape_(shape.to(backend)),
      stride_(0),
      device_stride_(0){

    if(data_.size() != calc_data_size(shape_))
      throw std::runtime_error("tensor::Tensor");

    set_stride();
  }

  inline MatrixView<T> flatten_matrix_view(){
    if(rank() == 0){
      return MatrixView<T>(0,0,data_.data_ptr(0));
    }

    return MatrixView<T>(numel() / dim(rank() - 1),dim(rank() - 1),data_.data_ptr(0));
  }

  inline MatrixView<const T> flatten_matrix_view() const{
    if(rank() == 0){
      return MatrixView<const T>(0,0,data_.data_ptr(0));
    }

    return MatrixView<const T>(numel() / dim(rank() - 1),dim(rank() - 1),data_.data_ptr(0));
  }

  inline T* data(){
    return data_.data();
  }

  inline const T* data() const{
    return data_.data();
  }

  inline const Storage<int64_t>& shape() const{
    return shape_;
  }

  inline const Storage<int64_t>& device_shape() const{
    return device_shape_;
  }

  inline const Storage<int64_t>& stride() const{
    return stride_;
  }

  inline const Storage<int64_t>& device_stride() const{
    return device_stride_;
  }

  inline T& at(const Storage<int64_t> &index){
    return data_.at(calc_offset(index));
  }

  inline const T& at(const Storage<int64_t> &index) const{
    return data_.at(calc_offset(index));
  }

  inline DataPtr<T> data_ptr(const Storage<int64_t> &index){
    return data_.data_ptr(calc_offset(index));
  }

  inline DataPtr<const T> data_ptr(const Storage<int64_t> &index) const{
    return data_.data_ptr(calc_offset(index));
  }

  inline int64_t rank() const{
    return shape_.size();
  }

  inline int64_t dim(size_t i) const{
    assert(i < shape_.size());
    return shape_.at(i);
  }

  inline int64_t numel() const noexcept{
    return data_.size();
  }

  inline Backend backend() const noexcept{
    return data_.backend();
  }

  std::string to_string() const{
    return to_string(shape_);
  }

  std::string to_string(const Storage<int64_t> &range) const{
    std::string text = "tensor::Tensor(\n  shape_={";

    for(int64_t i = 0;i < shape_.size();i++){
      if(i != 0) text += ", ";
      text += std::to_string(shape_.at(i));
    }

    text += "},\n  stride_={";

    for(int64_t i = 0;i < stride_.size();i++){
      if(i != 0) text += ", ";
      text += std::to_string(stride_.at(i));
    }

    text += "},\n  backend_="
             + nn::to_string(data_.backend())
             + ",\n  dtype_="
             + nn::dtype_name<T>()
             + ",\n  data_=";

    Storage<int64_t> shape = shape_.toCPU();

    int64_t copy_len = 1;

    for(int64_t i = 0;i < shape.size();i++){
      if(i < range.size()){
        const int64_t value = std::clamp<int64_t>(range.at(i),0,shape_.at(i));
        shape.at(i) = value;
        copy_len *= value;
      }
    }

    Storage<int64_t> stride(shape.size(),Backend::CPU);

    int64_t base = 1;
    for(int64_t i = shape.size();i-- > 0;){
      stride.at(i) = base;
      base *= shape.at(i);
    }

    if(copy_len == 0){
      text += "{\n          ...\n        }";
    }else if(rank() == 1){
      text += "{";
      for(int64_t i = 0;i < copy_len;i++){
        if(i != 0) text += ", ";
        text += std::to_string(data_.at(i));
      }

      if(copy_len < numel()){
        if(copy_len == 0){
          text += "...";
        }else{
          text += ", ...";
        }
      }

      text += "}";
    }else{
      Storage<T> data(0);

      if(backend() == Backend::CPU){
        data = to_string_cpu_copy(copy_len,stride);
      }else if(backend() == Backend::CUDA){
        data = to_string_cuda_copy(copy_len,stride);
      }

      Storage<int64_t> index(shape.size(),Backend::CPU);
      for(int64_t &i:index) i = 0;

      const Tensor<T> te(shape,data,Backend::CPU);

      to_string_recursive(text,index,te,0);
    }

    text += ")";

    return text;
  }

private:
  Storage<T> data_;
  Storage<int64_t> shape_;
  Storage<int64_t> stride_;
  Storage<int64_t> device_shape_;
  Storage<int64_t> device_stride_;

  Storage<T> to_string_cpu_copy(const int64_t copy_len,
                                const Storage<int64_t> &stride) const;

  Storage<T> to_string_cuda_copy(const int64_t copy_len,
                                 const Storage<int64_t> &stride) const;

  inline void to_string_recursive(std::string &text,
                                  Storage<int64_t> &index,
                                  const Tensor<T> &te,
                                  const int64_t dim) const{

    if(dim == 0){
      text += "{\n";
    }else{
      if(dim + 1 == te.rank()){
        text += std::string(8 + 2 * dim,' ') + "{";
      }else{
        text += std::string(8 + 2 * dim,' ') + "{\n";
      }
    }

    for(int64_t i = 0;i < te.dim(dim);i++){
      index.at(dim) = i;

      if(dim + 1 == te.rank()){
        if(i != 0) text += ", ";
        text += std::to_string(te.at(index));
      }else{
        to_string_recursive(text,index,te,dim + 1);
      }
    }

    if(te.dim(dim) < shape_.at(dim)){
      if(dim + 1 == te.rank()){
        text += ", ...";
      }else{
        text += std::string(10 + 2 * dim,' ') + "...\n";
      }
    }

    if(dim + 1 == te.rank()){
      text += "}";
    }else{
      text += std::string(8 + 2 * dim,' ') + "}";
    }

    if(dim > 0){
      if(index.at(dim - 1) + 1 < shape_.at(dim - 1)) text += ",";
    }

    if(dim != 0) text += "\n";
  }

  //data_のsizeを計算する
  inline static int64_t calc_data_size(const Storage<int64_t>& shape){
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
    #ifndef NDEBUG
    if(shape_.size() != index.size())
      throw std::runtime_error("tensor::Tensor check_index dimension mismatch");

    if(index.backend() != Backend::CPU)
      throw std::runtime_error("tensor::Tensor check_index Backend is not CPU");

    for(int64_t i = 0;i < index.size();i++){
      const int64_t value = index.at(i);
      if(value < 0 || shape_.at(i) <= value)
        throw std::runtime_error("Tensor index out of bounds at dim");
    }
    #endif
  }

  //indexから実際にどこにアクセスするか計算する
  inline int64_t calc_offset(const Storage<int64_t> &index) const{
    #ifndef NDEBUG
    check_index(index);
    #endif

    int64_t offset = 0;
    for(int64_t i = 0;i < shape_.size();i++){
      offset += index.at(i) * stride_.at(i);
    }

    return offset;
  }
};

template<dtype T>
inline std::ostream& operator<<(std::ostream &o,const Tensor<T> &t){
  Storage<int64_t> limit = t.shape().toCPU();
  for(int64_t i = 0;i < limit.size();i++){
    if(i + 1 == limit.size()){
      limit.at(i) = 6;
    }else if(i + 2 == limit.size()){
      limit.at(i) = 2;
    }else if(i + 3 == limit.size()){
      limit.at(i) = 2;
    }else{
      limit.at(i) = 1;
    }
  }

  return o << t.to_string(limit);
}

}//namespace cobalt_715::nn::tensor