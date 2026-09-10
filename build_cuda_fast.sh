#!/bin/bash

mapfile -t files < <(
    find . -type f \( -name "*.cpp" -o -name "*.cu" \)
)

nvcc -I. -DCOBALT_715_USE_CUDA -std=c++20 -O3 -DNDEBUG -arch=sm_75 -Xcompiler "-O3 -DNDEBUG -march=native" "${files[@]}" -o main