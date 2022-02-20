#include <device_launch_parameters.h>
#include <math.h>

#include "kernels.h"
#include "macros.h"

int div_ceil(int a, int b) { return (a + b - 1) / b; }

template <typename T>
__global__ void vector_eq_cuda(T* va, T b, T* vc, uint32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        vc[tid] = va[tid] == b ? 1 : 0; // :O this is legal?
    }
}

template <typename T>
void vector_eq(T* va, T b, T* vc, uint32_t n)
{
    constexpr int block_size = 256;
    int n_blocks = div_ceil(n, block_size);
    vector_eq_cuda<T> CK(n_blocks, block_size)(va, b, vc, n);
}

__global__ void vector_times_max1_log2_cuda(float* va, float* vb, uint32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        // Magical formula (use 0.5f or 1.0f your choice)
        vb[tid] = va[tid] * log2f(fmaxf(1.0f, va[tid]));
    }
}

void vector_times_max1_log2(float* va, float* vb, uint32_t n)
{
    constexpr int block_size = 256;
    int n_blocks = div_ceil(n, block_size);
    vector_times_max1_log2_cuda CK(n_blocks, block_size)(va, vb, n);
}

template <typename T>
__global__ void vector_add_cuda(T* va, T* vb, T* vc, uint32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        vc[tid] = va[tid] + vb[tid];
    }
}

template <typename T>
void vector_add(T* d_arr, T* d_brr, T* d_out, uint32_t n)
{
    constexpr int block_size = 256;
    int n_blocks = div_ceil(n, block_size);
    vector_add_cuda<T> CK(n_blocks, block_size)(d_arr, d_brr, d_out, n);
}

template <typename T>
__global__ void vector_multiply_scalar_cuda(T* va, T b, T* vc, uint32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        vc[tid] = va[tid] * b;
    }
}

template <typename T>
void vector_multiply_scalar(T* d_arr, T b, T* d_out, uint32_t n)
{
    constexpr int block_size = 256;
    int n_blocks = div_ceil(n, block_size);
    vector_multiply_scalar_cuda<T> CK(n_blocks, block_size)(d_arr, b, d_out, n);
}

template <typename T>
__global__ void vector_add_scalar_cuda(T* va, T b, T* vc, uint32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        vc[tid] = va[tid] + b;
    }
}

template <typename T>
void vector_add_scalar(T* d_arr, T b, T* d_out, uint32_t n)
{
    constexpr int block_size = 256;
    int n_blocks = div_ceil(n, block_size);
    vector_add_scalar_cuda<T> CK(n_blocks, block_size)(d_arr, b, d_out, n);
}
