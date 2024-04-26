#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <memory>
#include <assert.h>
#include <random>


struct Matrix {

	std::unique_ptr<double[]> matrix;
	int n, m;

    double matrix_gen() {
        static std::random_device rd;
        static std::mt19937 seed(rd()); 
        static std::uniform_real_distribution<double> dis(-1,1);
	    return dis(seed);
    }
    
	// debug用
    void print() const;

	// 构造函数
	Matrix() { m = n = 0; matrix = NULL; }
	Matrix(int n, int m) {
		assert(n > 0 && m > 0);
		this->n = n; this->m = m; int size = n * m;
		matrix.reset(new double[size]);
		for (int i=0;i<size;++i) {
			matrix[i] = matrix_gen();
		}
	}

	// 快捷访问数据
	double& operator()(int x, int y);

	void matrixMul(const Matrix & a, const Matrix & b);
	
	// 含有广播机制的矩阵加法
	void matrixAdd(const Matrix & a);

	void matrixSub(const Matrix & a);

	// 压缩矩阵为列向量
	void matrixReduce(const Matrix & a);
	
	void matrixFun(const Matrix & a, double (*fun)(double));

	void matrixDot(const Matrix & a, const Matrix & b );

	void matrixDot(double x);

	void matrixTrans(const Matrix & a);
};


#endif