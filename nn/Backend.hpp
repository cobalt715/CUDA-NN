#pragma once

#include <string>
#include <ostream>

namespace cobalt_715::nn{

enum class Backend{
  CPU,
  CUDA
};

constexpr const std::string_view to_string(Backend backend) noexcept{
  switch(backend){
    case Backend::CPU:
      return "CPU";
    case Backend::CUDA:
      return "CUDA";
  }

  return "Unknown";
}

inline std::ostream& operator<<(std::ostream &o,Backend backend){
  return o << to_string(backend);
}

}//namespace cobalt_715::nn