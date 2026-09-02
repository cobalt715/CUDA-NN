#pragma once

#include <string>
#include <cstdint>
#include <ostream>
#include <type_traits>
#include "Storage.cuh"
#include "nn/Backend.hpp"
#include "nn/Dtype.hpp"

namespace cobalt_715::nn::tensor{

//T[]などから一部を行列として借用する
template<nn::dtype T=float>
struct MatrixView{
private:
  using ValueType = std::remove_const_t<T>;

  using StorageType =
    std::conditional_t<
      std::is_const_v<T>,
      const Storage<ValueType>,
      Storage<ValueType>
    >;

public:
  enum class Layout{
    ROW_MAJOR_CONTIGUOUS,//row_stride == cols_ && col_stride_ == 1
    COL_MAJOR_CONTIGUOUS,//col majr contiguous
    ROW_CONTIGUOUS,//行連続
    COL_CONTIGUOUS,//列連続
    STRIDED_SAFE,//strided_safe
    OVERLAPPED//書き込み不可能
  };

  constexpr MatrixView(const int64_t rows,
                       const int64_t cols,
                       StorageType &data,
                       const int64_t offset = 0
                       ) : rows_(rows),
                           cols_(cols),
                           row_stride_(cols),
                           col_stride_(1),
                           data_(data),
                           offset_(offset){

    update_layout();
  }

  constexpr MatrixView(const int64_t rows,
                       const int64_t cols,
                       const int64_t row_stride,
                       const int64_t col_stride,
                       StorageType &data,
                       const int64_t offset = 0
                       ) : rows_(rows),
                           cols_(cols),
                           row_stride_(row_stride),
                           col_stride_(col_stride),
                           data_(data),
                           offset_(offset){

    update_layout();
  }

  inline StorageType& data() noexcept{
    return data_;
  }

  inline StorageType& data() const noexcept{
    return data_;
  }

  inline int64_t offset() const noexcept{
    return offset_;
  }

  inline T& at(const int64_t row,const int64_t col){
    return data_.at(offset_ + row * row_stride_ + col * col_stride_);
  }

  inline const T& at(const int64_t row,const int64_t col) const{
    return data_.at(offset_ + row * row_stride_ + col * col_stride_);
  }

  inline int64_t rows() const noexcept{
    return rows_;
  }

  inline int64_t cols() const noexcept{
    return cols_;
  }

  inline int64_t row_stride() const noexcept{
    return row_stride_;
  }

  inline int64_t col_stride() const noexcept{
    return col_stride_;
  }

  inline int64_t numel() const noexcept{
    return rows_ * cols_;
  }

  inline Backend backend() const noexcept{
    return data_.backend();
  }

  inline bool empty() const noexcept{
    return rows_ * cols_ == 0;
  }

  inline std::string dtype() const noexcept{
    return nn::dtype_name<T>();
  }

private:
  //data_.data()[offset_ + row * row_stride_ + col * col_stride_]でアクセスする
  StorageType &data_;//constなし
  int64_t offset_;
  int64_t rows_;//行
  int64_t cols_;//列
  int64_t row_stride_;
  int64_t col_stride_;
  Layout layout_;

  inline constexpr void update_layout() noexcept{
    if(std::abs(row_stride_) >= cols_ * std::abs(col_stride_) ||
       std::abs(col_stride_) >= rows_ * std::abs(row_stride_) ||
       row_stride_ == 0 ||
       col_stride_ == 0){

      layout_ = Layout::OVERLAPPED;
    }else if(row_stride_ == cols_){
      if(col_stride_ == 1){
        layout_ = Layout::ROW_MAJOR_CONTIGUOUS;
      }else{
        layout_ = Layout::ROW_CONTIGUOUS;
      }
    }else if(col_stride_ == rows_){
      if(row_stride_ == 1){
        layout_ = Layout::COL_MAJOR_CONTIGUOUS;
      }else{
        layout_ = Layout::COL_CONTIGUOUS;
      }
    }else{
      layout_ = Layout::STRIDED_SAFE;
    }
  }
};

}//namespace cobalt_715::nn::tensor