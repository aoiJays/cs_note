#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#include <algorithm>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void distance(float2 * points, int * res, int n) {
	// 防止总线程数不足点数量 则一个线程负责多个点
	for (int i=threadIdx.x+blockIdx.x*blockDim.x;i<n;i+=blockDim.x*gridDim.x) {
		if ( points[i].x * points[i].x + points[i].y * points[i].y <= 0.5 *0.5 ) res[i] = 1;
		else res[i] = 0;
		// printf("%d: (%f, %f) dis = %f res = %d\n", i, points[i].x, points[i].y, points[i].x * points[i].x + points[i].y * points[i].y, res[i]);
	}
}

__global__ void sum(int * res, int n, int * cnt) {
	// 每512个数字进行求和
	
	int id = threadIdx.x;
	__shared__ int sData[512];

	for (int i=0;i<n;i+=blockDim.x) {
		if ( id + i < n ) sData[id] = res[i + id];
		__syncthreads();
		for (int j=256;j;j>>=1) {
			if ( id + j < 512 && id + i + j < n) sData[id] += sData[id + j];
			__syncthreads();
		}
		if (id == 0) *cnt += sData[id];
	}

}

#include <iostream>
int main() {

	int testCase = 100000000;
	srand(time(0));

	float2 * points = new float2[testCase];
	for (int i=0;i<testCase;++i) {
		points[i].x = rand() % 10000 * 1.0 / 10000 - 0.5;
		points[i].y = rand() % 10000 * 1.0 / 10000 - 0.5;
	}

	float2 * points_gpu; int * res_gpu, * cnt_gpu;
	CUDA_CHECK(cudaMalloc((void**)&points_gpu, testCase * sizeof(float2)));
	CUDA_CHECK(cudaMalloc((void**)&res_gpu, testCase * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&cnt_gpu, 1 * sizeof(int)));

	CUDA_CHECK(cudaMemcpy(points_gpu, points, testCase*sizeof(float2), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemset(cnt_gpu, 0, 1 * sizeof(int)));

	int threadNum = 1024, blockNum = 512;
	distance<<<blockNum, threadNum>>>(points_gpu, res_gpu, testCase);

	int * cnt = new int;

	sum<<<1, 512>>>(res_gpu, testCase, cnt_gpu);

	CUDA_CHECK(cudaMemcpy(cnt, cnt_gpu, 1*sizeof(int), cudaMemcpyDeviceToHost));

	printf("PI = %.10f", ((float)*cnt / testCase) * 4);





	delete[] points; delete cnt;
	CUDA_CHECK(cudaFree(res_gpu)); 
	CUDA_CHECK(cudaFree(cnt_gpu)); 
	CUDA_CHECK(cudaFree(points_gpu)); 
    return 0;
}
