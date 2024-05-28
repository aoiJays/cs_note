#include "matrix.h"

#include <iostream>
#include <assert.h>

#include "cuda_run.h"

void Matrix::print() {
	tocpu();
	for (int i=0;i<n;++i) 
		for (int j=0;j<m;++j) std::cout << matrix[i * m + j] << " \n"[j==m-1];
	std::cout << std::endl;
}

double& Matrix::operator()(int x, int y) {
	assert( x >= 0 && x < n && y>=0 && y<m );
	return matrix[x * m + y];
}


const int TILE_WIDTH = 16;

void Matrix::matrixMul( Matrix & a,  Matrix & b) {
	assert(n==a.n && a.m==b.n && b.m==m);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    cu_tile_matrixMul<<<gridDim, blockDim>>>(a.gpu, b.gpu, gpu, n, a.m, m);
	cudacheck(cudaDeviceSynchronize());	
}


void Matrix::matrixAdd( Matrix & a) {
		
	// 完全相同
	if ( n == a.n && m==a.m ) {
		int sz = n * m;
		int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
		cu_matrixAdd<<<blockNum, threadNum>>>(gpu, a.gpu, sz);
	}
	// (n, 1)
	else if ( n == a.n && a.m == 1 ) {		
		int sz = n * m;
		int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
		cu_matrixAdd<<<blockNum, threadNum>>>(gpu, a.gpu, n, m);

	}
	else assert(false);
	cudacheck(cudaDeviceSynchronize());
}


void Matrix::matrixSub( Matrix & a) {

	assert( n == a.n && m==a.m );
	int sz = n * m;

	int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
	cu_matrixSub<<<blockNum, threadNum>>>(gpu, a.gpu, sz);
	cudacheck(cudaDeviceSynchronize());	
}

const int reduce_num = 1024;
void Matrix::matrixReduce( Matrix & a) {
	// ok
	assert(a.n==n && m == 1);

	cu_maxtrixReduce<<<a.n, reduce_num>>>(gpu, a.gpu, a.n, a.m);
	cudacheck(cudaDeviceSynchronize());	
}

void Matrix::matrixDot( Matrix & a,  Matrix & b ) {
	// ok
	assert(this->n == a.n && this->m == a.m);
	assert(this->n == b.n && this->m == b.m);
	int sz = n * m;
	int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
	cu_matrixDot<<<blockNum, threadNum>>>(gpu, a.gpu, b.gpu, n * m);
	cudacheck(cudaDeviceSynchronize());
}

void Matrix::matrixDot(double x) {
	// ok
	int sz = n * m; 
	int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
	cu_matrixDot<<<blockNum, threadNum>>>(gpu, x, n * m);
	cudacheck(cudaDeviceSynchronize()); 	
}

void Matrix::matrixTrans( Matrix & a) {
	// ok
	assert(n==a.m && m==a.n); 
	int sz = n * m;
	int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
	cu_matrixTrans<<<blockNum, threadNum>>>(gpu, a.gpu, n, m);
	cudacheck(cudaDeviceSynchronize());
}


void Matrix::matrixSubmatrix( Matrix & a, int l, int r) {
	// ok
	assert( a.n == n && (r -l) == m  && l >= 0 && r <= a.m && l < r);
	int sz = n * m;
	int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
	cu_matrixSubmatrix<<<blockNum, threadNum>>>(gpu, a.gpu, l, r, a.n, a.m);
	cudacheck(cudaDeviceSynchronize());
}

double Matrix::sum() {
	// ok
	double s = 0; int sz = n * m; double * sgpu; 
	cudacheck( cudaMalloc( (void**)&sgpu, sizeof(double) ) );
	cudacheck( cudaMemset( sgpu, 0, sizeof(double) ) );
	cu_sum<<<1, reduce_num>>>(gpu, sz, sgpu);
	cudacheck( cudaMemcpy(&s, sgpu, sizeof(double), cudaMemcpyDeviceToHost) );
	cudacheck( cudaFree(sgpu) );
	return s;
}
