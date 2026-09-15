#pragma once

#include "Backend.hpp"
#include "dtype.hpp"

namespace cobalt_715::nn{

//ポインタとBackendの情報(CPU,CUDAなど)を保持する
template<nn::dtype T>
struct DataPtr{
public:
  DataPtr(T *data,nn::Backend backend)
    : data_(data),backend_(backend){}

  inline T* data() noexcept{
    return data_;
  }

  inline const T* data() const noexcept{
    return data_;
  }

  inline nn::Backend backend() const noexcept{
    return backend_;
  }

private:
  T *data_;
  nn::Backend backend_;
};

}//namespace cobalt_715::nn