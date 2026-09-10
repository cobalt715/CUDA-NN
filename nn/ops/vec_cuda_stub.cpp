#ifndef COBALT_715_USE_CUDA

#include <stdexcept>
#include "vec.cuh"
#include "nn/dtype.hpp"
#include "nn/cuda/util.cuh"

namespace cobalt_715::nn::ops::vec::cuda{

template<nn::mutable_dtype T>
void add(const T *a,const T *b,T *out,const int64_t n){
  nn::cuda::throw_not_enabled();
}

template<nn::mutable_dtype T>
void sub(const T *a,const T *b,T *out,const int64_t n){
  nn::cuda::throw_not_enabled();
}

template<nn::mutable_dtype T>
void mul(const T *a,const T *b,T *out,const int64_t n){
  nn::cuda::throw_not_enabled();
}

template<nn::mutable_dtype T>
void div(const T *a,const T *b,T *out,const int64_t n){
  nn::cuda::throw_not_enabled();
}

#define INSTANTIATE_ADD_SUB_MUL_DIV(T) \
  template void add(const T *a,const T *b,T *out,const int64_t n); \
  template void sub(const T *a,const T *b,T *out,const int64_t n); \
  template void mul(const T *a,const T *b,T *out,const int64_t n); \
  template void div(const T *a,const T *b,T *out,const int64_t n);

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_ADD_SUB_MUL_DIV)

template<nn::mutable_dtype T>
void dot(const T *a,const T *b,T *out,const int64_t n){
  nn::cuda::throw_not_enabled();
}

#define INSTANTIATE_DOT(T) \
  template void dot(const T *a,const T *b,T *out,const int64_t n);

COBALT_715_FOR_EACH_DTYPE(INSTANTIATE_DOT)

}//namespace cobalt_715::nn::ops::vec::cuda

#endif //COBALT_715_USE_CUDA