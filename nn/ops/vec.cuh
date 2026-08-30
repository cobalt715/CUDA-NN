#pragma once

#include <cstdint>

namespace cobalt_715::nn::ops::vec{

namespace cpu{

void add(const float *a,const float *b,float *out,const int64_t n) noexcept;
void sub(const float *a,const float *b,float *out,const int64_t n) noexcept;
void mul(const float *a,const float *b,float *out,const int64_t n) noexcept;
void div(const float *a,const float *b,float *out,const int64_t n) noexcept;

void dot(const float *a,const float *b,float *out,const int64_t n) noexcept;

}//namespace cpu

namespace cuda{

void add(const float *a,const float *b,float *out,const int64_t n);
void sub(const float *a,const float *b,float *out,const int64_t n);
void mul(const float *a,const float *b,float *out,const int64_t n);
void div(const float *a,const float *b,float *out,const int64_t n);

void dot(const float *a,const float *b,float *out,const int64_t n);

}//namespace cuda

}//namespace cobalt_715::nn::ops::vec