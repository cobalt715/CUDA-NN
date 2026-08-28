#pragma once

#include <string>
#include <ostream>

namespace cobalt_715::nn{

//どこで動かすか
enum class Backend{
  CPU,
  CUDA
};

//この関数を持っているか
template<typename T>
concept BackendObject =
requires(const T &x){
  { x.backend() } -> std::same_as<Backend>;
};

//文字列にする
constexpr const std::string to_string(Backend backend) noexcept{
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