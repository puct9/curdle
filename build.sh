#!/bin/sh
nvcc src/AnswerSpace.cpp src/absurdle.cpp src/FilterWords.cpp src/kernels.cu -lcublas -O3 -o absurdle
