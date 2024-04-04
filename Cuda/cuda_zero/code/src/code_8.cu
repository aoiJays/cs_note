#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
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

#define SIZE 31

__global__ void adder(bool * bit_a, bool * bit_b, bool * bit_s, int pos, bool * Clast ) {
	
	if ( pos + threadIdx.x >= SIZE ) return;
	bool Ci = *Clast;

	for (int i=pos;i<pos+threadIdx.x;++i) {
		Ci = (bit_a[i] & bit_b[i]) | ((bit_a[i] ^ bit_b[i]) & Ci);
	}

	bit_s[pos + threadIdx.x] = bit_a[pos + threadIdx.x] ^ bit_b[pos + threadIdx.x] ^ Ci;

	// printf("%d %d: %d+%d=%d...%d\n", threadIdx.x, pos + threadIdx.x, bit_a[threadIdx.x+pos], bit_b[threadIdx.x+pos], bit_s[threadIdx.x+pos], Ci);
	__syncthreads();
	if ( threadIdx.x == blockDim.x - 1) 
		*Clast = (bit_a[pos + threadIdx.x] & bit_b[pos + threadIdx.x]) | ((bit_a[pos + threadIdx.x] ^ bit_b[pos + threadIdx.x]) & Ci);
}

// __global__ void check(bool * bit_a, bool * bit_b, bool * bit_s) {
// 	for (int i=0;i<SIZE;++i) printf("%d", bit_a[i]); printf("\n");
// 	for (int i=0;i<SIZE;++i) printf("%d", bit_b[i]); printf("\n");
// 	for (int i=0;i<SIZE;++i) printf("%d", bit_s[i]); printf("\n");
// }

int solve() {

	srand(time(0));
	int a = rand() % 300000, b = rand() % 300000;
	// int a = 11, b = 5;
	// printf("checking %d + %d = %d\n", a, b, a + b);

	bool * bit_a = new bool[SIZE], * bit_b = new bool[SIZE], * bit_s = new bool[SIZE];
	for (int i=0;i<SIZE;++i) bit_a[i] = (1 << i) & a;
	for (int i=0;i<SIZE;++i) bit_b[i] = (1 << i) & b;

	bool * bit_a_gpu, * bit_b_gpu, * bit_s_gpu, *Clast_gpu;
	CUDA_CHECK(cudaMalloc((void**)&bit_a_gpu, SIZE*sizeof(bool)));
	CUDA_CHECK(cudaMalloc((void**)&bit_b_gpu, SIZE*sizeof(bool)));
	CUDA_CHECK(cudaMalloc((void**)&bit_s_gpu, SIZE*sizeof(bool)));
	CUDA_CHECK(cudaMalloc((void**)&Clast_gpu, 1*sizeof(bool)));

	CUDA_CHECK(cudaMemcpy(bit_a_gpu, bit_a, SIZE*sizeof(bool), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(bit_b_gpu, bit_b, SIZE*sizeof(bool), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemset(Clast_gpu, 0, 1*sizeof(bool)));

	const int threadNum = 4;
	const int blockNum = 1;

	// check<<<1, 1>>>(bit_a_gpu, bit_b_gpu, bit_s_gpu);

	for (int i=0;i<SIZE;i+=threadNum) {
		adder<<<blockNum, threadNum>>>(bit_a_gpu, bit_b_gpu, bit_s_gpu, i, Clast_gpu);

		// int Clast;
		// CUDA_CHECK(cudaMemcpy(&Clast, Clast_gpu, 1*sizeof(bool), cudaMemcpyDeviceToHost));
		// printf("%d: %d\n", i, Clast);
	}

	CUDA_CHECK(cudaMemcpy(bit_s, bit_s_gpu, SIZE*sizeof(bool), cudaMemcpyDeviceToHost));

	int ans = 0;
	for (int i=SIZE-1;i>=0;--i) ans = ans << 1 | bit_s[i];
	// printf("GPU Adder: %d\n", ans);

	delete[] bit_a; delete[] bit_b; delete[] bit_s;
	CUDA_CHECK(cudaFree(bit_a_gpu));
	CUDA_CHECK(cudaFree(bit_b_gpu));
	CUDA_CHECK(cudaFree(bit_s_gpu));


	assert(ans==a+b);
    return 0;
}

int main() {

	int LoopTime = 100000;
	while(LoopTime -- ) solve();
}