#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

__global__ void add(int *a, int *b, int *c, int num) {
	if ( threadIdx.x < num ) 
		c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main(int argc, char ** argv) {
	
	int num = 10;
	int a[num], b[num], c[num];
	
	for (int i=0;i<num;++i) a[i] = i;
	for (int i=0;i<num;++i) b[i] = i * i;

	int *agpu, *bgpu, *cgpu;
	
	cudaMalloc((void**)&agpu, num * sizeof(int));
	cudaMalloc((void**)&bgpu, num * sizeof(int));
	cudaMalloc((void**)&cgpu, num * sizeof(int));

	cudaMemcpy(agpu, a, num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(bgpu, b, num * sizeof(int), cudaMemcpyHostToDevice);

	// 加法
	add<<<1, 10>>>(agpu, bgpu, cgpu, num);
	cudaMemcpy(c, cgpu, num * sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("add:\n");
	for (int i=0;i<num;++i) printf("%d + %d = %d\n", a[i], b[i], c[i]);

	cudaFree(agpu); 
	cudaFree(bgpu); 
	cudaFree(cgpu); 
    cudaDeviceReset(); 
}