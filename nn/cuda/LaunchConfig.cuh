#pragma once

#ifdef COBALT_715_USE_CUDA

#include "config.cuh"

namespace cobalt_715::nn::cuda{

struct LaunchConfig{
  dim3 grid;
  dim3 block;

  LaunchConfig
};

}//namespace cobalt_715::nn::cuda

#endif //COBALT_715_USE_CUDA