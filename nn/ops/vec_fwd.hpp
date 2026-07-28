#pragma once

#include <cstddef>
#include "nn/Backend.hpp"

//基本的なベクトル演算を用意している
namespace cobalt_715::nn::ops::vec{

//out[i] = a[i] + b[i]
template<Backend B>
inline void add_alias_safe(const float *a,const float *b,float *out,const size_t n) = delete;

template<Backend B>
inline void add(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) = delete;

//out[i] = a[i] - b[i]
template<Backend B>
inline void sub_alias_safe(const float *a,const float *b,float *out,const size_t n) = delete;

template<Backend B>
inline void sub(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) = delete;

//out[i] = a[i] * b[i]
template<Backend B>
inline void mul_alias_safe(const float *a,const float *b,float *out,const size_t n) = delete;

template<Backend B>
inline void mul(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) = delete;

//out[i] = a[i] / b[i]
template<Backend B>
inline void div_alias_safe(const float *a,const float *b,float *out,const size_t n) = delete;

template<Backend B>
inline void div(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) = delete;

//out[i] = a[i] * b[i] + c[i]
template<Backend B>
inline void fma_alias_safe(const float *a,const float *b,const float *c,float *out,const size_t n) = delete;

template<Backend B>
inline void fma(const float*__restrict a,const float*__restrict b,const float*__restrict c,float*__restrict out,const size_t n) = delete;

//out[i] = a[i] * c
template<Backend B>
inline void scale_alias_safe(const float *a,const float c,float *out,const size_t n) = delete;

template<Backend B>
inline void scale(const float*__restrict a,const float c,float*__restrict out,const size_t n) = delete;

//out[i] = a[i] * c + b[i]
template<Backend B>
inline void axpy_alias_safe(const float *a,const float c,const float *b,float *out,const size_t n) = delete;

template<Backend B>
inline void axpy(const float*__restrict a,const float c,const float*__restrict b,float*__restrict out,const size_t n) = delete;

}//namespace cobalt_715::nn::ops::vec