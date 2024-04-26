#include "matrix.h"

#include <iostream>
#include <assert.h>

void Matrix::print() const {
	for (int i=0;i<n;++i) 
		for (int j=0;j<m;++j) std::cout << matrix[i * m + j] << " \n"[j==m-1];
	std::cout << std::endl;
}

double& Matrix::operator()(int x, int y) {
	assert( x >= 0 && x < n && y>=0 && y<m );
	return matrix[x * m + y];
}

void Matrix::matrixMul(const Matrix & a, const Matrix & b) {
	assert(n==a.n && a.m==b.n && b.m==m);
	for (int i=0;i<n;++i)
		for (int j=0;j<m;++j) {
			matrix[i * m + j] = 0;
			for (int k=0;k<a.m;++k)
				matrix[i * m + j] += a.matrix[i *  a.m + k] * b.matrix[k *  b.m + j];
		}
}

void Matrix::matrixAdd(const Matrix & a) {
		
	// 完全相同
	if ( n == a.n && m==a.m ) {
		int size = n * m;
		for (int i=0;i<size;++i) matrix[i] += a.matrix[i];
	}

	// (n, 1)
	else if ( n == a.n && a.m == 1 ) {
		for (int i=0;i<n;++i) 
			for (int j=0;j<m;++j)
				matrix[i * n + j] += a.matrix[i];
	}

	// (1, m)
	else if ( 1 == a.n && a.m == m ) {
		for (int i=0;i<n;++i)
			for (int j=0;j<m;++j)
				matrix[i * n + j] += a.matrix[j];
	}

	// (1,1)
	else if ( 1 == a.n && 1==a.m ) {
		int size = n * m;
		for (int i=0;i<size;++i) matrix[i] += a.matrix[0];
	}

	else assert(false);
}

void Matrix::matrixSub(const Matrix & a) {
	assert( n == a.n && m==a.m );
	int size = n * m;
	for (int i=0;i<size;++i) matrix[i] -= a.matrix[i];
}


void Matrix::matrixReduce(const Matrix & a) {
	assert(a.n==n);
	for (int i=0;i<n;++i) {
		matrix[i] = 0;
		for (int j=0;j<a.m;++j)
			matrix[i] += a.matrix[i * a.m + j];
	}
}


void Matrix::matrixFun(const Matrix & a, double (*fun)(double)) {
	assert(this->n == a.n && this->m == a.m); int size = n * m;
	for (int i=0;i<size;++i)
		matrix[i] = fun(a.matrix[i]);
}

void Matrix::matrixDot(const Matrix & a, const Matrix & b ) {
	assert(this->n == a.n && this->m == a.m);
	assert(this->n == b.n && this->m == b.m);
	int size = n * m;
	for (int i=0;i<size;++i)
		matrix[i] = a.matrix[i] * b.matrix[i];
}

void Matrix::matrixDot(double x) {
	int size = n * m;
	for (int i=0;i<size;++i)
		matrix[i] *= x;
}

void Matrix::matrixTrans(const Matrix & a) {
	assert(n==a.m && m==a.n);
	for (int i=0;i<n;++i)
		for (int j=0;j<m;++j)
			matrix[i * m + j] = a.matrix[ j*a.m + i ];
}