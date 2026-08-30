@echo off

powershell -Command "$root=(Get-Location).Path; $files=(Get-ChildItem -Recurse -File -Include *.cpp | ForEach-Object { $_.FullName.Substring($root.Length+1) }); g++ -I. -std=c++20 $files -o main.exe"