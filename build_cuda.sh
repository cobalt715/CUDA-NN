#!/bin/bash

mapfile -t files < <(
    find . -type f \( -name "*.cpp" -o -name "*.cu" \)
)

nvcc -I. -DCOBALT_715_USE_CUDA -std=c++20 "${files[@]}" -o main