#include "cuda_run.h"

const int TILE_WIDTH = 16;
__global__ void cu_tile_matrixMul(double *a, double *b, double *c, int n, int m, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    float cValue = 0.0;
    for (int t = 0, t2 = 0; t2 < (m + TILE_WIDTH - 1); ++t, t2 += TILE_WIDTH) {
        As[ty][tx] = (row < n && t * TILE_WIDTH + tx < m) ? a[row * m + t * TILE_WIDTH + tx] : 0;
        Bs[ty][tx] = (t * TILE_WIDTH + ty < m && col < k) ? b[(t * TILE_WIDTH + ty) * k + col] : 0;
        __syncthreads();
        
        for (int i = 0; i < TILE_WIDTH; ++i)  cValue += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    
    if (row < n && col < k) c[row * k + col] = cValue;
}

__global__ void cu_naive_matrixMul(double * dst, double * a, double * b, int n, int m, int k) {
	
	int tx = threadIdx.x, bx = blockIdx.x;
	int id = bx * blockDim.x + tx;

	if ( id >= n * k ) return;

	int x = id / k, y = id % k;
	float tmp = 0;

	for (int i=0;i<m;++i)
		tmp += a[ x * m + i ] * b[ i * k + y ];

	dst[id] = tmp;
}


__global__ void cu_matrixAdd(double * dst, double * src, int sz) {
	int tx = threadIdx.x, bx = blockIdx.x;
	int id = bx * blockDim.x + tx;
	if ( id < sz ) dst[id] += src[id];
}

__global__ void cu_matrixAdd(double * dst, double * src, int n, int m) {
	int tx = threadIdx.x, bx = blockIdx.x;
	int id = bx * blockDim.x + tx;
	if ( id >= n * m ) return;
	int x = id / m;
	dst[id] += src[ x ];
}

__global__ void cu_maxtrixReduce(double * dst, double * a, int n, int m) {
	const int reduce_num = 1024;
	int id = threadIdx.x;
	__shared__ float sData[reduce_num];

	for (int i=0;i<m;i+=blockDim.x) {
		if ( id + i < m ) sData[id] = a[ m * blockIdx.x + i + id];
		__syncthreads();
		for (int j=reduce_num>>1;j;j>>=1) {
			if ( id + j < reduce_num && id + i + j < m) sData[id] += sData[id + j];
			__syncthreads();
		}
		if (id == 0) dst[blockIdx.x] += sData[id];
	}
}


__global__ void cu_matrixDot(double * dst, double x, int sz) {
	int tx = threadIdx.x, bx = blockIdx.x;
	int id = bx * blockDim.x + tx;
	if ( id < sz ) dst[id] *= x;
}

__global__ void cu_matrixDot(double * dst, double *a, double * b, int sz) {
	int tx = threadIdx.x, bx = blockIdx.x;
	int id = bx * blockDim.x + tx;
	if ( id < sz ) dst[id] = a[id] * b[id];
}

__global__ void cu_matrixTrans(double * dst, double *a, int n, int m) {
	int tx = threadIdx.x, bx = blockIdx.x;
	int id = bx * blockDim.x + tx;
    if ( id >= n * m) return;
	int x = id / m, y = id % m;
	dst[ x * m + y ] = a[ y * n + x ];
}


__global__ void cu_matrixSubmatrix(double * dst, double * a, int l, int r, int n, int m) {
	
	int tx = threadIdx.x, bx = blockIdx.x;
	int id = bx * blockDim.x + tx;
	
	if ( id >= n * (r - l) ) return;
	int x = id / (r - l), y = id % (r - l);

	dst[ x * (r - l) + y ] = a[ x * m + l + y ];
}


__global__ void cu_sum(double * res, int n, double * cnt) {
	const int reduce_num = 1024;
	int id = threadIdx.x;
	__shared__ float sData[reduce_num];

	for (int i=0;i<n;i+=blockDim.x) {
		if ( id + i < n ) sData[id] = res[i + id];
		__syncthreads();
		for (int j=reduce_num>>1;j;j>>=1) {
			if ( id + j < reduce_num && id + i + j < n) sData[id] += sData[id + j];
			__syncthreads();
		}
		if (id == 0) *cnt += sData[id];
	}
}


__global__ void cu_matrixSub(double * dst, double * src, int sz) {
	int tx = threadIdx.x, bx = blockIdx.x;
	int id = bx * blockDim.x + tx;
	if ( id < sz ) dst[id] -= src[id];
}

// 激活函数相关的cuda
// ----------------------------------

__global__ void cu_linear( double * dst, double * a, int sz) {
    int tx = threadIdx.x, bx = blockIdx.x;
    int id = bx * blockDim.x + tx;
    if (id < sz) dst[id] = a[id];
}

__global__ void cu_dlinear( double * dst, double * a, int sz) {
    int tx = threadIdx.x, bx = blockIdx.x;
    int id = bx * blockDim.x + tx;
    if (id < sz) dst[id] = 1;
}

__global__ void cu_ReLU( double * dst, double * a, int sz) {
    int tx = threadIdx.x, bx = blockIdx.x;
    int id = bx * blockDim.x + tx;
    if (id < sz) dst[id] = a[id] > 0 ? a[id] : 0;
}

__global__ void cu_dReLU( double * dst, double * a, int sz) {
    int tx = threadIdx.x, bx = blockIdx.x;
    int id = bx * blockDim.x + tx;
    if (id < sz) dst[id] = a[id] >= 0 ? 1 : 0;
}

// Loss
// --------------------

const double eps = 1e-3;
__global__ void cu_MSE( double * dst, double * a, double * b, int sz) {
    int tx = threadIdx.x, bx = blockIdx.x;
    int id = bx * blockDim.x + tx;
    if (id < sz) dst[id] = (a[id]-b[id])*(a[id]-b[id]) * 0.5;
}

__global__ void cu_dMSE( double * dst, double * a, double * b, int sz) {
    int tx = threadIdx.x, bx = blockIdx.x;
    int id = bx * blockDim.x + tx;
    if (id < sz) dst[id] = -a[id] + b[id];
}

__global__ void cu_CrossEntropy( double * dst, double * a, double * b, int sz) {
    int tx = threadIdx.x, bx = blockIdx.x;
    int id = bx * blockDim.x + tx;
    if (id < sz) dst[id] = a[id] < eps ? 0 : -a[id] * log(b[id]);
}

__global__ void cu_dCrossEntropy_2_softmax( double * dst, double * a, double * b, int sz) {
    int tx = threadIdx.x, bx = blockIdx.x;
    int id = bx * blockDim.x + tx;
    if (id < sz) dst[id] = -a[id] + b[id];
}

__global__ void cu_softmax( double * dst, double * a, int n, int m) {
	
	int tx = threadIdx.x, bx = blockIdx.x;
	int id = bx * blockDim.x + tx;

    if ( id >= n * m ) return;

	float sum = 0;
	for (int i=0;i<n;++i) 
		sum += exp(a[ i * m + id ]);
	for (int i=0;i<n;++i) 
		dst[ i * m + id ] = exp(a[ i * m + id ]) / sum;
	
}

__global__ void cu_dsoftmax( double * dst, int sz) {
	int tx = threadIdx.x, bx = blockIdx.x;
	int id = bx * blockDim.x + tx;
    if ( id < sz ) dst[id] = 1; 
}