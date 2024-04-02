#include <stdio.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void paraSum(int * a, int * b) {
	
	int id = threadIdx.x;
	if (id >= 16) return;

	__shared__ int shared_data[16]; // 同一个block公用一个
	shared_data[id] = a[id];
	__syncthreads(); // 同步所有的线程都完成了赋值

	for (int i=8;i;i>>=1) {
		if (id + i < 16) shared_data[id] += shared_data[i + id];
		__syncthreads(); // 同步
	}

	if ( id == 0 ) {
		*b = shared_data[0];
	}
}

int main() {

	int a[16];
	for (int i=0;i<16;++i) a[i] = i;
	int real_ans = (0 + 15) * 16 / 2;

	int * agpu;
	CUDA_CHECK(cudaMalloc((void**)&agpu, 16 * sizeof(int)));	
	CUDA_CHECK(cudaMemcpy(agpu, a, 16 * sizeof(int), cudaMemcpyHostToDevice));

	int * bgpu;
	CUDA_CHECK(cudaMalloc((void**)&bgpu, 1 * sizeof(int)));	

	paraSum<<<1, 16>>>(agpu, bgpu);
	int b;
	CUDA_CHECK(cudaMemcpy(&b, bgpu, 1 * sizeof(int), cudaMemcpyDeviceToHost));

	printf("%d %d\n", real_ans, b);
	CUDA_CHECK(cudaFree(agpu));
	CUDA_CHECK(cudaFree(bgpu));
    return 0;
}
