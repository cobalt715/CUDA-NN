@echo off

powershell -Command "$root=(Get-Location).Path; $files=(Get-ChildItem -Recurse -File -Include *.cpp | ForEach-Object { $_.FullName.Substring($root.Length+1) }); g++ -DNDEBUG -I. -std=c++20 -Ofast $files -o main.exe"