#include <device_launch_parameters.h>
#include <math.h>

#include "kernels.h"
#include "macros.h"

int div_ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void vector_eq_cuda(float* va, float b, float* vc, uint32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        vc[tid] = va[tid] == b ? 1.0f : 0.0f;
    }
}

void vector_eq(float* va, float b, float* vc, uint32_t n)
{
    constexpr int block_size = 256;
    int n_blocks = div_ceil(n, block_size);
    vector_eq_cuda CK(n_blocks, block_size)(va, b, vc, n);
}

__global__ void vector_log2_cuda(float* va, float* vb, uint32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        // Magical formula (use 0.5f or 1.0f your choice)
        vb[tid] = log2f(va[tid]);
    }
}

void vector_log2(float* va, float* vb, uint32_t n)
{
    constexpr int block_size = 256;
    int n_blocks = div_ceil(n, block_size);
    vector_log2_cuda CK(n_blocks, block_size)(va, vb, n);
}

__global__ void vector_reduce_max_cuda(float* va, float* vb, float* vc, uint32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        vc[tid] = fmaxf(va[tid], vb[tid]);
    }
}

void vector_reduce_max(float* d_arr, float* d_brr, float* d_out, uint32_t n)
{
    constexpr int block_size = 256;
    int n_blocks = div_ceil(n, block_size);
    vector_reduce_max_cuda CK(n_blocks, block_size)(d_arr, d_brr, d_out, n);
}

__global__ void vector_multiply_scalar_cuda(float* va, float b, float* vc, uint32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        vc[tid] = va[tid] * b;
    }
}

void vector_multiply_scalar(float* d_arr, float b, float* d_out, uint32_t n)
{
    constexpr int block_size = 256;
    int n_blocks = div_ceil(n, block_size);
    vector_multiply_scalar_cuda CK(n_blocks, block_size)(d_arr, b, d_out, n);
}

__global__ void vector_add_scalar_cuda(float* va, float b, float* vc, uint32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        vc[tid] = va[tid] + b;
    }
}

void vector_add_scalar(float* d_arr, float b, float* d_out, uint32_t n)
{
    constexpr int block_size = 256;
    int n_blocks = div_ceil(n, block_size);
    vector_add_scalar_cuda CK(n_blocks, block_size)(d_arr, b, d_out, n);
}
