#pragma once

#ifdef COBALT_715_USE_CUDA
#include <cuda_runtime.h>
#endif
#include <string>
#include <stdexcept>
#include <source_location>

namespace cobalt_715::nn::cuda{

#ifdef COBALT_715_USE_CUDA

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

#endif //COBALT_715_USE_CUDA

[[noreturn]]
inline void throw_not_enabled(const std::source_location& location = std::source_location::current()){
  throw std::runtime_error(
    std::string(location.file_name())
    + ":"
    + std::to_string(location.line())
    + ": "
    + "CUDA is not enabled"
  );
}

}//namespace cobalt_715::nn::cuda