#ifndef CU_Kernels_H
#define CU_Kernels_H

#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>

// vector eq
__global__ void vector_eq_cuda(float* va, float b, float* vc, uint32_t n);

void vector_eq(float* va, float b, float* vc, uint32_t n);


// magic
__global__ void vector_log2_cuda(float* va, float* vb, uint32_t n);

void vector_log2(float* va, float* vb, uint32_t n);

// vector add
__global__ void vector_reduce_max_cuda(float* va, float* vb, float* vc, uint32_t n);

void vector_reduce_max(float* d_arr, float* d_brr, float* d_out, uint32_t n);


// vector multiply
__global__ void vector_multiply_scalar_cuda(float* va, float b, float* vc, uint32_t n);

void vector_multiply_scalar(float* d_arr, float b, float* d_out, uint32_t n);


// vector add
__global__ void vector_add_scalar_cuda(float* va, float b, float* vc, uint32_t n);

void vector_add_scalar(float* d_arr, float b, float* d_out, uint32_t n);


#endif
