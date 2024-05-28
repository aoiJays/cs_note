#ifndef _CUDA_RUN_H_
#define _CUDA_RUN_H_

#include <iostream>


// CUDA 错误检查宏
#define cudacheck(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void cu_tile_matrixMul(double *a, double *b, double *c, int n, int m, int k);
__global__ void cu_naive_matrixMul(double * dst, double * a, double * b, int n, int m, int k);
__global__ void cu_matrixAdd(double * dst, double * src, int sz);
__global__ void cu_matrixAdd(double * dst, double * src, int n, int m);
__global__ void cu_matrixSub(double * dst, double * src, int sz);
__global__ void cu_maxtrixReduce(double * dst, double * a, int n, int m);
__global__ void cu_matrixDot(double * dst, double x, int sz);
__global__ void cu_matrixDot(double * dst, double *a, double * b, int sz);
__global__ void cu_matrixTrans(double * dst, double *a, int n, int m);
__global__ void cu_matrixSubmatrix(double * dst, double * a, int l, int r, int n, int m);
__global__ void cu_sum(double * res, int n, double * cnt);


__global__ void cu_linear( double * dst, double * a, int sz);
__global__ void cu_dlinear( double * dst, double * a, int sz);
__global__ void cu_ReLU( double * dst, double * a, int sz);
__global__ void cu_dReLU( double * dst, double * a, int sz);


__global__ void cu_MSE( double * dst, double * a, double * b, int sz);
__global__ void cu_dMSE( double * dst, double * a, double * b, int sz);
__global__ void cu_CrossEntropy( double * dst, double * a, double * b, int sz);
__global__ void cu_dCrossEntropy_2_softmax( double * dst, double * a, double * b, int sz);
__global__ void cu_softmax( double * dst, double * a, int n, int m);
__global__ void cu_dsoftmax( double * dst, int sz);

#endif