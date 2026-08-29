#include <immintrin.h>
#include <cstddef>
#include "vec.cuh"

namespace cobalt_715::nn::ops::vec::cpu{

void add(const float *a,const float *b,float *out,const int64_t n) noexcept{
  #ifdef __AVX__
  size_t i = 0;
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
  for(size_t i = 0;i < n;i++){
    out[i] = a[i] + b[i];
  }
  #endif
}

void sub(const float *a,const float *b,float *out,const int64_t n) noexcept{
  #ifdef __AVX__
  size_t i = 0;
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
  for(size_t i = 0;i < n;i++){
    out[i] = a[i] - b[i];
  }
  #endif
}

void mul(const float *a,const float *b,float *out,const int64_t n) noexcept{
  #ifdef __AVX__
  size_t i = 0;
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
  for(size_t i = 0;i < n;i++){
    out[i] = a[i] * b[i];
  }
  #endif
}

void div(const float *a,const float *b,float *out,const int64_t n) noexcept{
  #ifdef __AVX__
  size_t i = 0;
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
  for(size_t i = 0;i < n;i++){
    out[i] = a[i] / b[i];
  }
  #endif
}

}//namespace cobalt_715::nn::ops::vec::cpu