#pragma once

#include <string>
#include <ostream>

namespace cobalt_715::nn{

enum class Backend{
  CPU,
  CUDA
};

template<typename T>
concept BackendObject =
requires(const T& x){
  { x.backend() } -> std::same_as<Backend>;
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

//Backendがすべて同じか調べる
template<BackendObject T,BackendObject... Ts>
inline bool same_backend(const T& first,const Ts&... rest){
  return ((rest.backend() == first.backend()) && ...);
}

}//namespace cobalt_715::nn