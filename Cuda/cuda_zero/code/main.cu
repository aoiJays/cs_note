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

__global__ void add(double * a) {
	for (int i=0;i<10000;++i) {
		a[ blockIdx.x * blockDim.x + threadIdx.x] += i;
	}
}

__global__ void add2(double * a) {
	__shared__ double b[N];
	b[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();

	for (int i=0;i<10000;++i) {
		b[threadIdx.x] += i;
	}
	__syncthreads();
	a[blockIdx.x * blockDim.x + threadIdx.x] = b[threadIdx.x];
}
int main() {

	// set gpu
	

	double *ha = new double[N*N];
	srand(time(0));
	for (int i=0;i<N;++i) ha[i] = 0;

	double *da;
	CUDA_CHECK(cudaMalloc( (void**)&da, N*N*sizeof(double)));
	CUDA_CHECK(cudaMemcpy(da, ha, N*N*sizeof(double), cudaMemcpyHostToDevice));

	
	timeval startTime, endTime;
	auto print_Time = [&](const timeval &startTime, const timeval& endTime) {
    	long long elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000000LL + (endTime.tv_usec - startTime.tv_usec);
    	printf("Elapsed time: %lld microseconds\n", elapsedTime); // 微秒
	};




	gettimeofday(&startTime, NULL);

	for (int i=0;i<loopTime;++i)
		add2<<<N, N>>>(ha);

	gettimeofday(&endTime, NULL);
	print_Time(startTime, endTime);

	gettimeofday(&startTime, NULL);

	for (int i=0;i<loopTime;++i)
		add<<<N, N>>>(ha);

	gettimeofday(&endTime, NULL);
	print_Time(startTime, endTime);
	
	delete[] ha; 
    return 0;
}

