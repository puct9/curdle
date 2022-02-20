#ifndef CU_Array2D_CUDA_H
#define CU_Array2D_CUDA_H

#include "Array2D.h"
#include "kernels.h"
#include "macros.h"
#include <cublas_v2.h>

class Array2D_CUDA
{
private:
    static float onef;
    static float zerof;

    float* _d_data;
    int _r;
    int _c;

public:
    Array2D_CUDA(Array2D<float>& ref)
    {
        _r = ref.rows();
        _c = ref.cols();
        cudaMalloc(&_d_data, ref.numel() * sizeof(float));
        cudaMemcpy(_d_data, ref.data(), ref.numel() * sizeof(float), cudaMemcpyHostToDevice);
    }

    void Equals(float value, Array2D_CUDA* out)
    {
        vector_eq(_d_data, value, out->_d_data, _r * _c);
    }

    void SumRowsBLAS(Array2D_CUDA* out, Array2D_CUDA& ones, cublasHandle_t& cublas)
    {
        // Our matrix: (r, c)
        // Ones matrix: (c, 1)
        // Output: (r, 1)
        int m = _r;
        int k = _c;
        int n = 1;
        cublasSgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &onef, ones._d_data, n, _d_data,
                       k, &zerof, out->_d_data, n);
    }

    void MagicFormula() { vector_times_max1_log2(_d_data, _d_data, _r * _c); }

    void AddTo(Array2D_CUDA* out)
    {
        vector_add(_d_data, out->_d_data, out->_d_data, _r * _c);
    }

    void MultiplyScalar(float value)
    {
        vector_multiply_scalar(_d_data, value, _d_data, _r * _c);
    }

    void AddScalar(float value) { vector_add_scalar(_d_data, value, _d_data, _r * _c); }

    void CopyToHost(Array2D<float>* out)
    {
        cudaMemcpy(out->data(), _d_data, out->numel() * sizeof(float), cudaMemcpyDeviceToHost);
    }

    float* data() { return _d_data; }
    int rows() { return _r; }
    int cols() { return _c; }

    ~Array2D_CUDA() { cudaFree(_d_data); }
};

float Array2D_CUDA::onef = 1.0f;
float Array2D_CUDA::zerof = 0.0f;

#endif
