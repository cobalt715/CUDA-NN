@echo off

powershell -Command "$root=(Get-Location).Path; $files=(Get-ChildItem -Recurse -File -Include *.cpp,*.cu | ForEach-Object { $_.FullName.Substring($root.Length+1) }); nvcc -DCOBALT_715_USE_CUDA -DNDEBUG -I. -std=c++20 -O3 -Xcompiler '/utf-8 /O2 /arch:AVX2' $files -o main.exe"