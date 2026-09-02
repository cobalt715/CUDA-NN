#pragma once

#include <string>
#include <cstdint>
#include <type_traits>

namespace cobalt_715::nn{

template<class T>
concept dtype =
  std::is_same_v<std::remove_cv_t<T>,int32_t> ||
  std::is_same_v<std::remove_cv_t<T>,int64_t> ||
  std::is_same_v<std::remove_cv_t<T>,float> ||
  std::is_same_v<std::remove_cv_t<T>,double>;

template<dtype T>
std::string dtype_name() noexcept{
  using U = std::remove_cv_t<T>;

  std::string name;

  if constexpr (std::is_same_v<U,int32_t>){
    name =  "int32_t";
  }else if constexpr (std::is_same_v<U,int64_t>){
    name = "int64_t";
  }else if constexpr (std::is_same_v<U,float>){
    name = "float";
  }else if constexpr (std::is_same_v<U,double>){
    name = "double";
  }

  if constexpr (std::is_const_v<T>){
    name = "const " + name;
  }

  return name;
}

}//namespace cobalt_715::nn