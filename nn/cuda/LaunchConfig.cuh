#pragma once

#ifdef COBALT_715_USE_CUDA

#include "config.cuh"

namespace cobalt_715::nn::cuda{

struct LaunchConfig{
  const dim3 grid_;
  const dim3 block_;

  LaunchConfig(const dim3 grid,const dim3 block)
    : grid_(grid),
      block_(block){}
};

}//namespace cobalt_715::nn::cuda

#endif //COBALT_715_USE_CUDA