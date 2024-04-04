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

__host__ void cpu_sum(int * a, int * res, int n) {
	*res = 0;
	for (int i=0;i<n;++i) *res += a[i];
}


#define BLOCK_NUM 16
#define THREAD_NUM 1024


__global__ void gpu_sum(int * a, int * res, int n) {
	
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	if ( id >= n ) return;

	__shared__ int sData[THREAD_NUM];
	sData[tid] = 0;
	for (int i=id;i<n;i+=BLOCK_NUM*THREAD_NUM) sData[tid] += a[i];
	__syncthreads();

	for (int i=THREAD_NUM/2;i;i>>=1) {
		if (tid < i) sData[tid] += sData[tid + i];
		__syncthreads();
	}

	if (tid == 0) {
		res[blockIdx.x] = sData[0];
	}
}

int main() {

	
	timeval startTime, endTime;
	auto print_Time = [&](const timeval &startTime, const timeval& endTime) {
    	long long elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000000LL + (endTime.tv_usec - startTime.tv_usec);
    	printf("Elapsed time: %lld microseconds\n", elapsedTime); // 微秒
	};

	int n = 2042039;
	int *a = new int[n], res = 0;

	srand(time(0));
	for (int i=0;i<n;++i) {
		a[i] = rand() % 10;
	}

	// CPU 测速
	const int loopTime = 100;
	puts("Pending CPU:");
	gettimeofday(&startTime, NULL);

	for (int i=0;i<loopTime;++i)
		cpu_sum(a, &res, n);

	gettimeofday(&endTime, NULL);
	printf("CPU Version Ans = %d\n", res);
	print_Time(startTime, endTime);

	int *a_gpu, *block_res_gpu, *block_res = new int[BLOCK_NUM];

	CUDA_CHECK(cudaMalloc((void**)&a_gpu, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&block_res_gpu, BLOCK_NUM * sizeof(int)));

	CUDA_CHECK(cudaMemcpy(a_gpu, a, n*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemset(block_res_gpu, 0, BLOCK_NUM*sizeof(int)));

	puts("Pending GPU:");
	gettimeofday(&startTime, NULL);

	for (int i=0;i<loopTime;++i)
		gpu_sum<<<BLOCK_NUM, THREAD_NUM>>>(a_gpu, block_res_gpu, n);

	CUDA_CHECK(cudaMemcpy(block_res, block_res_gpu, BLOCK_NUM*sizeof(int), cudaMemcpyDeviceToHost));
	res = 0;
	for (int i=0;i<BLOCK_NUM;++i) res += block_res[i];

	gettimeofday(&endTime, NULL);
	printf("GPU Version Ans = %d\n", res);
	print_Time(startTime, endTime);

	CUDA_CHECK(cudaFree(a_gpu));
	CUDA_CHECK(cudaFree(block_res_gpu));
	delete[] a; delete[] block_res;
    return 0;
}
