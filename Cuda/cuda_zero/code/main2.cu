#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define N 1000
const int loopTime = 1000;

void CPU_MatrixMul(double * a, double * b, double * c, int n) {
	for (int i=0;i<n;++i)
		for (int j=0;j<n;++j) {
			c[i*n + j] = 0;
			for (int k=0;k<n;++k) c[i*n + j] += a[ i*n+k ] * b[ k*n+j ];
		}
}

void print_Time(const timeval &startTime, const timeval& endTime) {
	long long elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000000LL + (endTime.tv_usec - startTime.tv_usec);
	printf("Elapsed time: %lld microseconds\n", elapsedTime); // 微秒
};


void CPU_RUNNING(double * a, double * b, double * c, int n) {
	timeval startTime, endTime;
	puts("CPU:");

	gettimeofday(&startTime, NULL);
	for (int i=0;i<loopTime;++i) CPU_MatrixMul(a, b, c, n);
	gettimeofday(&endTime, NULL);
	
	print_Time(startTime, endTime);
	printf("\n");
}

__global__ void GPU_naive_MatrixMul(double * a, double * b, double * c, int n) {
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ( x >= n || y >= n ) return;
	double temp = 0;
	for (int i=0;i<n;++i)
		temp += a[x * n + i] * b[i * n + y];
	c[x * n + y] = temp;
}

void GPU_NAIVE_RUNNING(double * a, double * b, double * c, int n) {
	CUDA_CHECK( cudaSetDevice(0) );
	timeval startTime, endTime;
	puts("GPU - naive:");

	double * da, *db, *dc;
	CUDA_CHECK( cudaMalloc((void**)&da, N*N*sizeof(double)) );
	CUDA_CHECK( cudaMalloc((void**)&db, N*N*sizeof(double)) );
	CUDA_CHECK( cudaMalloc((void**)&dc, N*N*sizeof(double)) );


	CUDA_CHECK( cudaMemcpy(da, a, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(db, b, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemset(dc, 0.0, N*N*sizeof(double)) );

	dim3 threadNum(32, 32);
	dim3 blockNum( (n + threadNum.x - 1) / threadNum.x, (n + threadNum.y - 1) / threadNum.y );

	gettimeofday(&startTime, NULL);
	for (int i=0;i<loopTime;++i) 
		GPU_naive_MatrixMul<<<blockNum, threadNum>>>(da, db, dc, n);

	gettimeofday(&endTime, NULL);
	
	print_Time(startTime, endTime);
	printf("\n");

	// CUDA_CHECK( cudaMemcpy(c, dc, N*N*sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CHECK( cudaFree(da) );
	CUDA_CHECK( cudaFree(db) );
	CUDA_CHECK( cudaFree(dc) );
	CUDA_CHECK(cudaDeviceReset());
}

#define TILE_WIDTH 32
// 同一个TILE的属于同一个Block 共享一块shared
__global__ void GPU_block_MatrixMul(double * a, double * b, double * c, int n) {
	
	// C[x][y]
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ double A[TILE_WIDTH*TILE_WIDTH], B[TILE_WIDTH*TILE_WIDTH];
	
	// 对应TILE中的位置
	int tx = threadIdx.x, ty = threadIdx.y;	

	double sum = 0;
	// 枚举对应的A和B的tile
	for (int i=0;i<n;i+=TILE_WIDTH) {
		
		// 放入shared
		A[tx*TILE_WIDTH + ty] = i + ty < n ? a[ x * n + i + ty ] : 0;
		B[tx*TILE_WIDTH + ty] = i + tx < n ? b[ (i + tx) * n + y ] : 0;
		__syncthreads();

		// if (x==0&&y==2) {
		// 	for (int ii=0;ii<TILE_WIDTH;++ii) {
		// 		for (int j=0;j<TILE_WIDTH;++j) printf("%.2f ", A[ii*TILE_WIDTH+j]);
		// 		printf("\n");
		// 	}
		// 	printf("\n");
		// 	for (int ii=0;ii<TILE_WIDTH;++ii) {
		// 		for (int j=0;j<TILE_WIDTH;++j) printf("%.2f ", B[ii*TILE_WIDTH+j]);
		// 		printf("\n");
		// 	}
		// 	printf("-----\n");
		// }

		double dot_sum = 0;
		for (int j=0;j<TILE_WIDTH;++j) {
			dot_sum += A[tx*TILE_WIDTH + j] * B[j*TILE_WIDTH + ty];
		}
		__syncthreads();
		sum += dot_sum;
	}
	if (x < n && y < n)
		c[x * n + y] = sum;
}



void GPU_BLOCK_RUNNING(double * a, double * b, double * c, int n) {
	CUDA_CHECK( cudaSetDevice(0) );
	timeval startTime, endTime;
	puts("GPU - block:");

	double * da, *db, *dc;
	CUDA_CHECK( cudaMalloc((void**)&da, N*N*sizeof(double)) );
	CUDA_CHECK( cudaMalloc((void**)&db, N*N*sizeof(double)) );
	CUDA_CHECK( cudaMalloc((void**)&dc, N*N*sizeof(double)) );


	CUDA_CHECK( cudaMemcpy(da, a, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(db, b, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemset(dc, 0.0, N*N*sizeof(double)) );


	dim3 threadNum(TILE_WIDTH, TILE_WIDTH);
	dim3 blockNum( (n + threadNum.x - 1) / threadNum.x, (n + threadNum.y - 1) / threadNum.y );
	// dim3 blockNum( 1,1 );

	// printf("%d %d\n %d %d\n", blockNum.x, blockNum.y, threadNum.x, threadNum.y);
	gettimeofday(&startTime, NULL);
	for (int i=0;i<loopTime;++i) 
		GPU_block_MatrixMul<<<blockNum, threadNum>>>(da, db, dc, n);

	gettimeofday(&endTime, NULL);
	
	print_Time(startTime, endTime);
	printf("\n");

	// CUDA_CHECK( cudaMemcpy(c, dc, N*N*sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CHECK( cudaFree(da) );
	CUDA_CHECK( cudaFree(db) );
	CUDA_CHECK( cudaFree(dc) );
	CUDA_CHECK(cudaDeviceReset());
}


__global__ void matrixMultiplyShared(double *A, double *B, double *C,
int numARows = N, int numAColumns = N, int numBRows = N, int numBColumns = N, int numCRows = N, int numCColumns = N)
{
	//分配共享内存
	__shared__ double sharedM[TILE_WIDTH][TILE_WIDTH];
	__shared__ double sharedN[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	double Csub = 0.0;
	
	//核心：下面将保存在全局内存中的矩阵M&N分块存放到共享内存中
	for (int i = 0; i < (int)(ceil((double)numAColumns / blockDim.x)); i++)//如上图，将一个红框矩形分成多个正方形
	{
		if (i*blockDim.x + tx < numAColumns && row < numARows)//分割M矩阵，边界确定方式结合上图蓝色正方形内数据的位置理解
			sharedM[ty][tx] = A[row*numAColumns + i * blockDim.x + tx];
		else
			sharedM[ty][tx] = 0.0;

		if (i*blockDim.y + ty < numBRows && col < numBColumns)//分割N矩阵
			sharedN[ty][tx] = B[(i*blockDim.y + ty)*numBColumns + col];
		else
			sharedN[ty][tx] = 0.0;
		__syncthreads();//同一线程块中所有线程必须到达运行 __synctrheads()之后才可以做其余操作
		//此操作可以防止当只有部分数据拷贝到共享内存后就提前进行下列计算。

		for (int j = 0; j < blockDim.x; j++)//分块后的矩阵相乘
			Csub += sharedM[ty][j] * sharedN[j][tx];
		__syncthreads();
	}

	if (row < numCRows && col < numCColumns)//将计算后的矩阵块放到结果矩阵C中
		C[row*numCColumns + col] = Csub;
}

void GPU_OTHER_RUNNING(double * a, double * b, double * c, int n) {
	CUDA_CHECK( cudaSetDevice(0) );
	timeval startTime, endTime;
	puts("GPU - other:");

	double * da, *db, *dc;
	CUDA_CHECK( cudaMalloc((void**)&da, N*N*sizeof(double)) );
	CUDA_CHECK( cudaMalloc((void**)&db, N*N*sizeof(double)) );
	CUDA_CHECK( cudaMalloc((void**)&dc, N*N*sizeof(double)) );


	CUDA_CHECK( cudaMemcpy(da, a, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(db, b, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemset(dc, 0.0, N*N*sizeof(double)) );


	dim3 threadNum(TILE_WIDTH, TILE_WIDTH);
	dim3 blockNum( (n + threadNum.x - 1) / threadNum.x, (n + threadNum.y - 1) / threadNum.y );
	// dim3 blockNum( 1,1 );

	// printf("%d %d\n %d %d\n", blockNum.x, blockNum.y, threadNum.x, threadNum.y);
	gettimeofday(&startTime, NULL);
	for (int i=0;i<loopTime;++i) 
		matrixMultiplyShared<<<blockNum, threadNum>>>(da, db, dc);

	gettimeofday(&endTime, NULL);
	
	print_Time(startTime, endTime);
	printf("\n");

	// CUDA_CHECK( cudaMemcpy(c, dc, N*N*sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CHECK( cudaFree(da) );
	CUDA_CHECK( cudaFree(db) );
	CUDA_CHECK( cudaFree(dc) );
	CUDA_CHECK(cudaDeviceReset());
}


int main() {

	// set gpu
	

	double *ha = new double[N*N];
	double *hb = new double[N*N];
	double *hc = new double[N*N];
	double *hc2 = new double[N*N];
	double *hc3 = new double[N*N];

	// initial matrix
	srand(time(0));
	for (int i=0;i<N*N;++i) ha[i] = rand() % 1000000 / 1000.0;
	for (int i=0;i<N*N;++i) hb[i] = rand() % 1000000 / 1000.0;

	// CPU Version
	// CPU_RUNNING(ha, hb, hc, N);

	// GPU Version (devide matrix into blocks)
		GPU_BLOCK_RUNNING(ha, hb, hc3, N);
GPU_NAIVE_RUNNING(ha, hb, hc2, N);

	GPU_OTHER_RUNNING(ha, hb, hc, N);




	delete[] ha; delete[] hb; delete[] hc; delete[] hc2; delete[] hc3;
    return 0;
}

