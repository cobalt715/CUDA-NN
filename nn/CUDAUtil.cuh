#pragma once

#include <string>
#include <source_location>
#include <cuda_runtime.h>

namespace cobalt_715::nn{

inline void check_cuda(cudaError_t err,
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

}//namespace cobalt_715::nn