#include <immintrin.h>
#include <cstddef>
#include <type_traits>
#include "vec.cuh"
#include "nn/dtype.hpp"

namespace cobalt_715::nn::ops::vec::cpu{

template<nn::mutable_dtype T>
void add(const T *a,const T *b,T *out,const int64_t n) noexcept{
  if constexpr (!std::is_same_v<T,float>){
    for(int64_t i = 0;i < n;i++){
      out[i] = a[i] + b[i];
    }
    return;
  }else{

  #ifdef __AVX__
  int64_t i = 0;
  for(;i + 8 <= n;i += 8){
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vo = _mm256_add_ps(va,vb);
    _mm256_storeu_ps(out + i,vo);
  }
  for(;i < n;i++){
    out[i] = a[i] + b[i];
  }
  #else
  for(int64_t i = 0;i < n;i++){
    out[i] = a[i] + b[i];
  }
  #endif
  }
}

template<nn::mutable_dtype T>
void sub(const T *a,const T *b,T *out,const int64_t n) noexcept{
  if constexpr (!std::is_same_v<T,float>){
    for(int64_t i = 0;i < n;i++){
      out[i] = a[i] - b[i];
    }
    return;
  }else{

  #ifdef __AVX__
  int64_t i = 0;
  for(;i + 8 <= n;i += 8){
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vo = _mm256_sub_ps(va,vb);
    _mm256_storeu_ps(out + i,vo);
  }
  for(;i < n;i++){
    out[i] = a[i] - b[i];
  }
  #else
  for(int64_t i = 0;i < n;i++){
    out[i] = a[i] - b[i];
  }
  #endif
  }
}

template<nn::mutable_dtype T>
void mul(const T *a,const T *b,T *out,const int64_t n) noexcept{
  if constexpr (!std::is_same_v<T,float>){
    for(int64_t i = 0;i < n;i++){
      out[i] = a[i] * b[i];
    }
    return;
  }else{

  #ifdef __AVX__
  int64_t i = 0;
  for(;i + 8 <= n;i += 8){
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vo = _mm256_mul_ps(va,vb);
    _mm256_storeu_ps(out + i,vo);
  }
  for(;i < n;i++){
    out[i] = a[i] * b[i];
  }
  #else
  for(int64_t i = 0;i < n;i++){
    out[i] = a[i] * b[i];
  }
  #endif
  }
}

template<nn::mutable_dtype T>
void div(const T *a,const T *b,T *out,const int64_t n) noexcept{
  if constexpr (!std::is_same_v<T,float>){
    for(int64_t i = 0;i < n;i++){
      out[i] = a[i] / b[i];
    }
    return;
  }else{

  #ifdef __AVX__
  int64_t i = 0;
  for(;i + 8 <= n;i += 8){
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vo = _mm256_div_ps(va,vb);
    _mm256_storeu_ps(out + i,vo);
  }
  for(;i < n;i++){
    out[i] = a[i] / b[i];
  }
  #else
  for(int64_t i = 0;i < n;i++){
    out[i] = a[i] / b[i];
  }
  #endif
  }
}

#define INSTANTIATE_ADD_SUB_MUL_DIV(T) \
  template void add(const T *a,const T *b,T *out,const int64_t n) noexcept; \
  template void sub(const T *a,const T *b,T *out,const int64_t n) noexcept; \
  template void mul(const T *a,const T *b,T *out,const int64_t n) noexcept; \
  template void div(const T *a,const T *b,T *out,const int64_t n) noexcept;

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_ADD_SUB_MUL_DIV)

//Kahan summation
template<nn::mutable_dtype T>
void dot(const T *a,const T *b,T *out,const int64_t n) noexcept{
  T sum = 0;
  T c = 0;

  for(int64_t i = 0; i < n; i++){
    const T x = a[i] * b[i];

    const T y = x - c;
    const T t = sum + y;

    c = (t - sum) - y;
    sum = t;
  }

  *out = sum;
}

#define INSTANTIATE_DOT(T) \
  template void dot(const T *a,const T *b,T *out,const int64_t n) noexcept;

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_DOT)

}//namespace cobalt_715::nn::ops::vec::cpu