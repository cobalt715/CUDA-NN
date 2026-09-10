#pragma once

#include "nn/dtype.hpp"
#include "nn/cuda/config.cuh"

namespace cobalt_715::nn::ops::op{

template<nn::mutable_dtype T>
struct add{
  #ifdef COBALT_715_USE_CUDA
    __host__ __device__ inline static T f(T a,T b) noexcept{
      return a + b;
    }
  #else
    inline static T f(T a,T b) noexcept{
      return a + b;
    }
  #endif
};

template<nn::mutable_dtype T>
struct sub{
  #ifdef COBALT_715_USE_CUDA
    __host__ __device__ inline static T f(T a,T b) noexcept{
      return a - b;
    }
  #else
    inline static T f(T a,T b) noexcept{
      return a - b;
    }
  #endif
};

template<nn::mutable_dtype T>
struct mul{
  #ifdef COBALT_715_USE_CUDA
    __host__ __device__ inline static T f(T a,T b) noexcept{
      return a * b;
    }
  #else
    inline static float f(T a,T b) noexcept{
      return a * b;
    }
  #endif
};

template<nn::mutable_dtype T>
struct div{
  #ifdef COBALT_715_USE_CUDA
    __host__ __device__ inline static T f(T a,T b) noexcept{
      return a / b;
    }
  #else
    inline static T f(T a,T b) noexcept{
      return a / b;
    }
  #endif
};

}//namespace cobalt_715::nn::ops::op