@echo off

powershell -Command "$root=(Get-Location).Path; $files=(Get-ChildItem -Recurse -File -Include *.cpp,*.cu | ForEach-Object { $_.FullName.Substring($root.Length+1) }); nvcc -DCOBALT_715_USE_CUDA -I. -std=c++20 -Xcompiler '/utf-8' $files -o main.exe"