#ifndef CU_Kernels_H
#define CU_Kernels_H

#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>

// vector eq
template <typename T>
__global__ void vector_eq_cuda(T* va, T b, T* vc, uint32_t n);

template <typename T>
void vector_eq(T* va, T b, T* vc, uint32_t n);

template void vector_eq(float*, float, float*, uint32_t);

// magic
__global__ void vector_times_max1_log2_cuda(float* va, float* vb, uint32_t n);

void vector_times_max1_log2(float* va, float* vb, uint32_t n);

// vector add
template <typename T>
__global__ void vector_add_cuda(T* va, T* vb, T* vc, uint32_t n);

template <typename T>
void vector_add(T* d_arr, T* d_brr, T* d_out, uint32_t n);

template void vector_add(float*, float*, float*, uint32_t);

// vector multiply
template <typename T>
__global__ void vector_multiply_scalar_cuda(T* va, T b, T* vc, uint32_t n);

template <typename T>
void vector_multiply_scalar(T* d_arr, T b, T* d_out, uint32_t n);

template void vector_multiply_scalar(float*, float, float*, uint32_t);

// vector add
template <typename T>
__global__ void vector_add_scalar_cuda(T* va, T b, T* vc, uint32_t n);

template <typename T>
void vector_add_scalar(T* d_arr, T b, T* d_out, uint32_t n);

template void vector_add_scalar(float*, float, float*, uint32_t);

#endif
