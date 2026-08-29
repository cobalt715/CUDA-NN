#ifndef COBALT_715_USE_CUDA

#include <stdexcept>
#include "vec.cuh"

namespace cobalt_715::nn::ops::vec::cuda{

void add(const float *a,const float *b,float *out,const int64_t n){
  throw std::runtime_error("CUDA is not enabled");
}

void sub(const float *a,const float *b,float *out,const int64_t n){
  throw std::runtime_error("CUDA is not enabled");
}

void mul(const float *a,const float *b,float *out,const int64_t n){
  throw std::runtime_error("CUDA is not enabled");
}

void div(const float *a,const float *b,float *out,const int64_t n){
  throw std::runtime_error("CUDA is not enabled");
}

}//namespace cobalt_715::nn::ops::vec::cuda

#endif //COBALT_715_USE_CUDA