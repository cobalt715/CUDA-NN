#pragma once

#ifdef COBALT_715_USE_CUDA

#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <source_location>

namespace cobalt_715::nn::cuda{

inline void check(cudaError_t err,
                       const std::source_location& location = std::source_location::current()){

  if(err != cudaSuccess){
    throw std::runtime_error(
      std::string(location.file_name())
      + ":"
      + std::to_string(location.line())
      + ": "
      + cudaGetErrorString(err)
    );
  }
}

}//namespace cobalt_715::nn::cuda

#endif //COBALT_715_USE_CUDA