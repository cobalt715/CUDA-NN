#pragma once

#include <cstdint>
#include <nn/Dtype.hpp>

namespace cobalt_715::nn::ops::vec{

namespace cpu{

template<nn::mutable_dtype T>
void add(const T *a,const T *b,T *out,const int64_t n) noexcept;

template<nn::mutable_dtype T>
void sub(const T *a,const T *b,T *out,const int64_t n) noexcept;

template<nn::mutable_dtype T>
void mul(const T *a,const T *b,T *out,const int64_t n) noexcept;

template<nn::mutable_dtype T>
void div(const T *a,const T *b,T *out,const int64_t n) noexcept;

template<nn::mutable_dtype T>
void dot(const T *a,const T *b,T *out,const int64_t n) noexcept;

}//namespace cpu

namespace cuda{

template<nn::mutable_dtype T>
void add(const T *a,const T *b,T *out,const int64_t n);

template<nn::mutable_dtype T>
void sub(const T *a,const T *b,T *out,const int64_t n);

template<nn::mutable_dtype T>
void mul(const T *a,const T *b,T *out,const int64_t n);

template<nn::mutable_dtype T>
void div(const T *a,const T *b,T *out,const int64_t n);

template<nn::mutable_dtype T>
void dot(const T *a,const T *b,T *out,const int64_t n);

}//namespace cuda

}//namespace cobalt_715::nn::ops::vec