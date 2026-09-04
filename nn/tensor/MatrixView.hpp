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
                       ) : MatrixView(rows,cols,cols,1,data,offset){}

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

    //負は許さない
    if(rows_ < 0 || cols_ < 0 || offset_ < 0){
      throw std::invalid_argument("tensor::Storage::constructor negative size");
    }

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

  inline bool is_row_major_contiguous() const noexcept{
    return layout_ == Layout::ROW_MAJOR_CONTIGUOUS;
  }

  inline bool is_col_major_contiguous() const noexcept{
    return layout_ == Layout::COL_MAJOR_CONTIGUOUS;
  }

  inline bool is_row_contiguous() const noexcept{
    return layout_ == Layout::ROW_MAJOR_CONTIGUOUS ||
                      Layout::ROW_CONTIGUOUS;
  }

  inline bool is_col_contiguous() const noexcept{
    return layout_ == Layout::COL_MAJOR_CONTIGUOUS ||
                      Layout::COL_CONTIGUOUS;
  }

  inline bool is_stride_safe() const noexcept{
    return layout_ != Layout::OVERLAPPED;
  }

  inline bool is_overlapped() const noexcept{
    return layout_ == Layout::OVERLAPPED;
  }

  //転地を返す
  inline MatrixView<T> t() const noexcept{
    return MatrixView<T>(cols_,rows_,
                         col_stride_,row_stride_,
                         data_,
                         offset_);
  }

  std::string to_string() const{
    return to_string(rows_,cols_);
  }

  std::string to_string(const int64_t row,const int64_t col) const{
    std::string text = "tensor::MatrixView(rows_="
                       + std::to_string(rows_)
                       + ", cols_="
                       + std::to_string(cols_)
                       + ", row_stride_="
                       + std::to_string(row_stride_)
                       + ", col_stride_="
                       + std::to_string(col_stride_)
                       + ", offset_="
                       + std::to_string(offset_)
                       + ", backend_="
                       + nn::to_string(data_.backend())
                       + ", dtype="
                       + nn::dtype_name<T>()
                       + ",\n  data_={\n";

    const int64_t ro = std::clamp<int64_t>(row,0,rows_);

    const int64_t co = std::clamp<int64_t>(col,0,cols_);

    Storage<ValueType> data(0);

    if(data_.backend() == Backend::CPU){
      data = to_string_cpu_copy(ro,co);
    }else if(data_.backend() == Backend::CUDA){
      data = to_string_cuda_copy(ro,co);
    }

    if(ro != 0 && co != 0){
      for(int row = 0;row < ro;row++){
        text += "    {";
        for(int col = 0;col < co;col++){
          text += std::to_string(data.at(row * co + col));

          if(col + 1 < co){
            text += " , ";
          }else if(co < cols_){
            text += " , ...";
          }
        }

        if(row + 1 < rows_){
          text += "},\n";
        }else{
         text += "}\n";
        }
      }
    }

    if(ro < rows_ || co == 0){
      if(rows_ != 0 && cols_ != 0)
        text += "    ...\n";
    }

    text += "  })";

    return text;
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

  Storage<ValueType> to_string_cpu_copy(const int64_t ro,const int64_t co) const;
  Storage<ValueType> to_string_cuda_copy(const int64_t ro,const int64_t co) const;
};

template<class T>
inline std::ostream& operator<<(std::ostream &o,const MatrixView<T> &mv){
  return o << mv.to_string(4,4);
}

}//namespace cobalt_715::nn::tensor