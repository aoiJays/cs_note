#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


__global__ void gpu_add(int *a, int *hist, int num) {	
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= num) return;
	// atomicAdd 函数的返回值为执行原子加法操作之前 指向的变量的值
	atomicAdd( hist + a[id], 1 ); // hist[a[id]]的地址 加上1
}

void cpu_add(int * a, int *hist, int num) {
	for (int i=0;i<num;++i) ++ hist[a[i]];
}

int main() {

	int num = 90000; const int SIZE = 10;
	int * a = new int[num];

	srand(time(0));
	for (int i=0;i<num;++i) 
		a[i] = 1LL * rand() % SIZE;
	
	int * hist = new int[SIZE];

	int *a_gpu, *hist_gpu;
	CUDA_CHECK(cudaMalloc( (void**)&a_gpu, num * sizeof(int)));
	CUDA_CHECK(cudaMalloc( (void**)&hist_gpu, SIZE * sizeof(int)));

	CUDA_CHECK( cudaMemcpy(a_gpu, a, num *sizeof(int), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemset(hist_gpu, 0, SIZE * sizeof(int)));


	auto getMaxThreadCount = [&](int deviceID) {
		cudaDeviceProp prop;
		CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceID));
		printf("maxGridSize: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
		return prop.maxThreadsPerBlock;
	};

	// maxGridSize: (2147483647, 65535, 65535)
	// maxThreadsPerBlock: 1024
	int threadNum = getMaxThreadCount(0);
	
	// 把数据分成若干块，每块1024大小（Thread数量）
	int blockNum = ( num + threadNum - 1 ) / threadNum; // 向上取整


	timeval startTime, endTime;
	auto print_Time = [&](const timeval &startTime, const timeval& endTime) {
    	long long elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000000LL + (endTime.tv_usec - startTime.tv_usec);
    	printf("Elapsed time: %lld microseconds\n", elapsedTime); // 微秒
	};
	const int loopTime = 10;

	// GPU 版本
	puts("GPU:");
	gettimeofday(&startTime, NULL);

	for (int i=0;i<loopTime;++i)
		gpu_add<<<blockNum, threadNum>>>(a_gpu, hist_gpu, num);

	gettimeofday(&endTime, NULL);
	print_Time(startTime, endTime);

	
	CUDA_CHECK( cudaMemcpy(hist, hist_gpu, SIZE *sizeof(int), cudaMemcpyDeviceToHost) );
	for (int i=0;i<SIZE;++i) {
		printf("%d ", hist[i]); hist[i] = 0;
	}
	puts("");

	// CPU 版本
	puts("CPU:");
	gettimeofday(&startTime, NULL);

	for (int i=0;i<loopTime;++i)
		cpu_add(a, hist, num);

	gettimeofday(&endTime, NULL);
	print_Time(startTime, endTime);
	
	for (int i=0;i<SIZE;++i) {
		printf("%d ", hist[i]); 
	}
	puts("");
	
	// 释放
	CUDA_CHECK( cudaFree(a_gpu));
	CUDA_CHECK( cudaFree(hist_gpu));
	delete[] a; delete[] hist;
    return 0;
}
