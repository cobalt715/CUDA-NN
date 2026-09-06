#!/bin/bash

mapfile -t files < <(
    find . -type f -name "*.cpp"
)

g++ -I. -std=c++20 "${files[@]}" -o main