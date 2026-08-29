#pragma once

#include "nn/cuda/config.cuh"

namespace cobalt_715::nn::ops::op{

struct add{
  #ifdef COBALT_715_USE_CUDA
    __host__ __device__ inline static float f(float a,float b) noexcept{
      return a + b;
    }
  #else
    inline static float f(float a,float b) noexcept{
      return a + b;
    }
  #endif
};

struct sub{
  #ifdef COBALT_715_USE_CUDA
    __host__ __device__ inline static float f(float a,float b) noexcept{
      return a - b;
    }
  #else
    inline static float f(float a,float b) noexcept{
      return a - b;
    }
  #endif
};

struct mul{
  #ifdef COBALT_715_USE_CUDA
    __host__ __device__ inline static float f(float a,float b) noexcept{
      return a * b;
    }
  #else
    inline static float f(float a,float b) noexcept{
      return a * b;
    }
  #endif
};

struct div{
  #ifdef COBALT_715_USE_CUDA
    __host__ __device__ inline static float f(float a,float b) noexcept{
      return a / b;
    }
  #else
    inline static float f(float a,float b) noexcept{
      return a / b;
    }
  #endif
};

}//namespace cobalt_715::nn::ops::op