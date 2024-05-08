#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <memory>
#include <assert.h>
#include <random>

#include "cuda_run.h"

struct Matrix {

	double * matrix;
	double * gpu;

	int n, m;

	void togpu() {
		cudacheck( cudaMemcpy(gpu, matrix, n*m*sizeof(double), cudaMemcpyHostToDevice ));
	}

	void tocpu() {
		cudacheck( cudaMemcpy(matrix, gpu, n*m*sizeof(double), cudaMemcpyDeviceToHost ));
	}

    double matrix_gen() {
        static std::random_device rd;
        static std::mt19937 seed(rd()); 
        static std::uniform_real_distribution<double> dis(-0.5,0.5);
	    return dis(seed);
    }
    
	// debug用
    void print();

	// 构造函数
	Matrix() { m = n = 0; matrix = NULL; gpu = NULL; }
	Matrix(int n, int m) {

		assert(n > 0 && m > 0);
		this->n = n; this->m = m; int size = n * m;
	
		matrix = new double[size];
		for (int i=0;i<size;++i) {
			matrix[i] = matrix_gen();
		}


		cudacheck(cudaMalloc( (void**)&gpu, n * m *sizeof(double)));
		togpu();
	}


	// 快捷访问数据
	double& operator()(int x, int y);

	void matrixMul( Matrix & a,  Matrix & b);
	
	// 含有广播机制的矩阵加法
	void matrixAdd( Matrix & a);

	void matrixSub( Matrix & a);

	// 压缩矩阵为列向量
	void matrixReduce( Matrix & a);
	

	void matrixDot( Matrix & a,  Matrix & b );

	void matrixDot(double x);

	void matrixTrans( Matrix & a);

	void matrixSubmatrix( Matrix & a, int l, int r);

	double sum() ;

	~Matrix() {
		delete []matrix;
		cudacheck(cudaFree(gpu));
	}
};


#endif