#ifndef COBALT_715_USE_CUDA

#include <stdexcept>
#include "vec.cuh"
#include "nn/cuda/util.cuh"

namespace cobalt_715::nn::ops::vec::cuda{

void add(const float *a,const float *b,float *out,const int64_t n){
  nn::cuda::throw_not_enabled();
}

void sub(const float *a,const float *b,float *out,const int64_t n){
  nn::cuda::throw_not_enabled();
}

void mul(const float *a,const float *b,float *out,const int64_t n){
  nn::cuda::throw_not_enabled();
}

void div(const float *a,const float *b,float *out,const int64_t n){
  nn::cuda::throw_not_enabled();
}

void dot(const float *a,const float *b,float *out,const int64_t n){
  nn::cuda::throw_not_enabled();
}

}//namespace cobalt_715::nn::ops::vec::cuda

#endif //COBALT_715_USE_CUDA