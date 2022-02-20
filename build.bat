@echo off

nvcc src/AnswerSpace.cpp src/curdle.cpp src/FilterWords.cpp src/kernels.cu -lcublas -O3 -o curdle.exe
