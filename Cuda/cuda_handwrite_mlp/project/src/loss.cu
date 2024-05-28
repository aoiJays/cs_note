#include "loss.h"	

#include "matrix.h"

#define eps (1e-3)


void Loss::MSE(Matrix & dst, Matrix & y, Matrix & yhat) {

		
	assert( y.n == yhat.n && y.m == yhat.m );
	assert( dst.n == yhat.n && dst.m == yhat.m );

    int sz = dst.n * dst.m;
    int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
    cu_MSE<<<blockNum, threadNum>>>(dst.gpu, y.gpu, yhat.gpu, sz);

    cudacheck(cudaDeviceSynchronize());
		
}


void Loss::dMSE(Matrix & dst, Matrix & y, Matrix & yhat) {
		
	assert( y.n == yhat.n && y.m == yhat.m );
	assert( dst.n == yhat.n && dst.m == yhat.m );
    int sz = dst.n * dst.m;
    int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
    cu_dMSE<<<blockNum, threadNum>>>(dst.gpu, y.gpu, yhat.gpu, sz);

    cudacheck(cudaDeviceSynchronize());
		
}


void Loss::CrossEntropy(Matrix & dst, Matrix & y, Matrix & yhat) {
	
	assert( y.n == yhat.n && y.m == yhat.m );
	assert( dst.n == yhat.n && dst.m == yhat.m );
    int sz = dst.n * dst.m;
    int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
    cu_CrossEntropy<<<blockNum, threadNum>>>(dst.gpu, y.gpu, yhat.gpu, sz);
    cudacheck(cudaDeviceSynchronize());
	
}

void Loss::dCrossEntropy_2_softmax(Matrix & dst, Matrix & y, Matrix & yhat) {
	
	assert( y.n == yhat.n && y.m == yhat.m );
	assert( dst.n == yhat.n && dst.m == yhat.m );
    int sz = dst.n * dst.m;
    int threadNum = 1024, blockNum = (sz + threadNum - 1) / threadNum;
    cu_dCrossEntropy_2_softmax<<<blockNum, threadNum>>>(dst.gpu, y.gpu, yhat.gpu, sz);
    cudacheck(cudaDeviceSynchronize());
	
}