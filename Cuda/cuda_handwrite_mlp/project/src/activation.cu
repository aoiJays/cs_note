#include "activation.h"

#include <algorithm>
#include "matrix.h"


void Activation::linear(Matrix & a,  Matrix & b) {
	 
	assert(a.n==b.n && a.m == b.m);
    int sz = a.n * a.m;
    int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
    cu_linear<<<blockNum, threadNum>>>(a.gpu, b.gpu, sz);
    cudacheck(cudaDeviceSynchronize());
	
}


void Activation::dlinear(Matrix & a,  Matrix & b) {
	 
	assert(a.n==b.n && a.m == b.m);
    int sz = a.n * a.m;
    int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
    cu_dlinear<<<blockNum, threadNum>>>(a.gpu, b.gpu, sz);
    cudacheck(cudaDeviceSynchronize());
	
}


void Activation::ReLU(Matrix & a,  Matrix & b) {
	 
	assert(a.n==b.n && a.m == b.m);
    int sz = a.n * a.m;
    int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
    cu_ReLU<<<blockNum, threadNum>>>(a.gpu, b.gpu, sz);
    cudacheck(cudaDeviceSynchronize());
	
}


void Activation::dReLU(Matrix & a,  Matrix & b) {
		 
	assert(a.n==b.n && a.m == b.m);
    int sz = a.n * a.m;
    int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
    cu_dReLU<<<blockNum, threadNum>>>(a.gpu, b.gpu, sz);
    cudacheck(cudaDeviceSynchronize());
		
}

void Activation::softmax(Matrix & a,  Matrix & b) {
	 
	assert(a.n==b.n && a.m == b.m);
    int threadNum = 1024, blockNum = (a.m + threadNum - 1) / threadNum;
    cu_softmax<<<blockNum, threadNum>>>(a.gpu, b.gpu, a.n, a.m);
    cudacheck(cudaDeviceSynchronize());
	
}

void Activation::dsoftmax(Matrix & a,  Matrix & b) {
	assert(a.n==b.n && a.m == b.m); int sz = a.n * a.m;
    int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
    cu_dsoftmax<<<blockNum, threadNum>>>(a.gpu, sz);
    cudacheck(cudaDeviceSynchronize());
}