#pragma once

#include <string>
#include <cstdint>
#include <type_traits>

namespace cobalt_715::nn{

template<class T>
concept dtype =
  std::is_same_v<T,int32_t> ||
  std::is_same_v<T,int64_t> ||
  std::is_same_v<T,float> ||
  std::is_same_v<T,double>;

template<dtype T>
std::string dtype_name() noexcept{
  if constexpr (std::is_same_v<T,int32_t>) return "int32_t";
  else if constexpr (std::is_same_v<T,int64_t>) return "int64_t";
  else if constexpr (std::is_same_v<T,float>) return "float";
  else if constexpr (std::is_same_v<T,double>) return "double";
}

}//namespace cobalt_715::nn