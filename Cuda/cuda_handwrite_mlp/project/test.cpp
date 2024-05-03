#include <bits/stdc++.h>
#include <memory>


const int midLayer = 10;
const int num_epochs = 1000;
const double lr = 0.1;


// 定义矩阵类
struct Matrix {

	std::unique_ptr<double[]> matrix;
	int n, m;


	double gen() {
		static std::random_device rd;
		static std::mt19937 seed(rd()); 
		static std::uniform_real_distribution<double> dis(-1,1);	
		return dis(seed);
	}

	// debug用
	void print() const {
		for (int i=0;i<n;++i) 
			for (int j=0;j<m;++j) std::cout << matrix[i * m + j] << " \n"[j==m-1];
		std::cout << std::endl;
	}

	// 构造函数
	Matrix() { m = n = 0; matrix = NULL; }
	Matrix(int n, int m) {
		assert(n > 0 && m > 0);
		this->n = n; this->m = m; int size = n * m;
		matrix.reset(new double[size]);
		for (int i=0;i<size;++i) {
			matrix[i] = gen();
		}
	}

	// 快捷访问数据
	double& operator()(int x, int y) {
		assert( x >= 0 && x < n && y>=0 && y<m );
		return matrix[x * m + y];
	}

	double getVal(int x, int y) {
		assert( x >= 0 && x < n && y>=0 && y<m );
		return matrix[x * m + y];
	}

	void matrixMul(const Matrix & a, const Matrix & b) {
		assert(n==a.n && a.m==b.n && b.m==m);
		for (int i=0;i<n;++i)
			for (int j=0;j<m;++j) {
				matrix[i * m + j] = 0;
				for (int k=0;k<a.m;++k)
					matrix[i * m + j] += a.matrix[i *  a.m + k] * b.matrix[k *  b.m + j];
			}
	}
	
	// 含有广播机制的矩阵加法
	void matrixAdd(const Matrix & a) {
		
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

	void matrixSub(const Matrix & a) {
		assert( n == a.n && m==a.m );
		int size = n * m;
		for (int i=0;i<size;++i) matrix[i] -= a.matrix[i];
	}

	// 压缩矩阵为列向量
	void matrixReduce(const Matrix & a) {
		
		assert(a.n==n);
		for (int i=0;i<n;++i) {
			matrix[i] = 0;
			for (int j=0;j<a.m;++j)
				matrix[i] += a.matrix[i * a.m + j];
		}

	}
	
	void matrixFun(const Matrix & a, double (*fun)(double)) {
		assert(this->n == a.n && this->m == a.m); int size = n * m;
		for (int i=0;i<size;++i)
			matrix[i] = fun(a.matrix[i]);
	}

	void matrixDot(const Matrix & a, const Matrix & b ) {
		assert(this->n == a.n && this->m == a.m);
		assert(this->n == b.n && this->m == b.m);
		int size = n * m;
		for (int i=0;i<size;++i)
			matrix[i] = a.matrix[i] * b.matrix[i];
	}

	void matrixDot(double x) {
		int size = n * m;
		for (int i=0;i<size;++i)
			matrix[i] *= x;
	}

	void matrixTrans(const Matrix & a) {
		assert(n==a.m && m==a.n);
		for (int i=0;i<n;++i)
			for (int j=0;j<m;++j)
				matrix[i * m + j] = a.matrix[ j*a.m + i ];
	}
};

double ReLU(double x) {
	return std::max(x, 0.0);
    // return 1 / ( 1 + exp(-x));
}

double dReLU(double x) {
	return x>=0?1:0;
    // double sig = sigma(x);
    // return sig * (1 - sig);
}

double linear(double x) {
	return x;
    // return 1 / ( 1 + exp(-x));
}

double dlinear(double x) {
	return 1;
    // double sig = sigma(x);
    // return sig * (1 - sig);
}


std::vector<Matrix> features, labels;

double Loss(int l, int r, const Matrix & W1, const Matrix & b1,const Matrix & W2, const Matrix & b2) {

	int batch_size = r - l;
	Matrix A0(1, batch_size);
	Matrix A1(midLayer, batch_size), Z1(midLayer, batch_size);
	Matrix A2(1, batch_size), Z2(1, batch_size);
		// 填充输入层
		for (int i=l;i<r;++i)
			for (int j=0;j<1;++j)
				A0(j,i-l) = features[i](j,0);

	Z1.matrixMul(W1, A0); Z1.matrixAdd(b1); // wx + b
	A1.matrixFun(Z1, ReLU); // 激活函数

	Z2.matrixMul(W2, A1); Z2.matrixAdd(b2); // wx + b
	A2.matrixFun(Z2, linear); // 激活函数

	double loss = 0;
	std::ofstream output("./dataset/data.csv", std::ofstream::trunc);
	for (int i=l;i<r;++i) {
		double item = A2(0, i-l) - (labels[i](0,0));
		loss += item * item * 0.5;

		output << features[i](0,0) << " " << A2(0, i-l) << "\n";
		// std::cout << "temp = " << loss << "\n";
	}


    
	return loss;
}

int main() {

	// 创造数据集
	const int num_examples = 100;

    std::ifstream input("./dataset/data");
    for (int i=0;i<num_examples;++i) {

        features.emplace_back( Matrix(1, 1) );
		
		labels.emplace_back( Matrix(1, 1) );
		labels.back().matrixMul( features.back(), features.back() );

        input >> features.back()(0, 0);
            // printf( "%.20lf ", train(j,i) );
        input >> labels.back()(0,0);
        // printf( "%.20lf\n", train(len_X, i) );

    }

	// 定义网络
	const int batch_size = 100;
	// 输入层
	Matrix A0(1, batch_size);
	Matrix A0T(batch_size, 1);


	// 1层
	Matrix A1(midLayer, batch_size),A1T(batch_size, midLayer), Z1(midLayer, batch_size), W1(midLayer, 1),W1T(1,midLayer), b1(midLayer, 1);
	Matrix dA1(midLayer, batch_size), dZ1(midLayer, batch_size), dW1(midLayer, 1), db1(midLayer, 1);
	Matrix sigmadZ1(midLayer, batch_size);

	// 2层
	Matrix A2(1, batch_size), Z2(1, batch_size), W2(1, midLayer),W2T(midLayer, 1), b2(1, 1);
	Matrix dA2(1, batch_size), dZ2(1, batch_size), dW2(1, midLayer), db2(1, 1);
	Matrix sigmadZ2(1, batch_size);


	
	auto forward = [&](int l, int r) {

		// 填充输入层
		for (int i=l;i<r;++i)
			for (int j=0;j<1;++j)
				A0(j,i-l) = features[i](j,0);
		A0T.matrixTrans(A0);

		Z1.matrixMul(W1, A0); Z1.matrixAdd(b1); // wx + b
		A1.matrixFun(Z1, ReLU); // 激活函数
		sigmadZ1.matrixFun(Z1, dReLU);
		A1T.matrixTrans(A1);

		Z2.matrixMul(W2, A1); Z2.matrixAdd(b2); // wx + b
		A2.matrixFun(Z2, linear); // 激活函数
		sigmadZ2.matrixFun(Z2, dlinear);

		double loss = 0;
		for (int i=l;i<r;++i) {
			double item = A2(0, i-l) - (labels[i](0,0));
			loss += item * item * 0.5;
			dA2(0, i-l) = item;
		}

		return loss / (r-l);
	};

	auto backward = [&]() {

		
		dZ2.matrixDot(dA2, sigmadZ2 );
		db2.matrixReduce(dZ2);
		dW2.matrixMul(dZ2, A1T);

		dA1.matrixMul(W2T, dZ2);

		dZ1.matrixDot(dA1, sigmadZ1 );
		db1.matrixReduce(dZ1);

	// 		std::cout << "!"  << std::endl;
	// std::cout << "!"  << std::endl;
		dW1.matrixMul(dZ1, A0T);
	// std::cout << "!"  << std::endl;
	// std::cout << "!"  << std::endl;

	};



	auto sgd = [&]() {
		
		dW1.matrixDot( lr / batch_size );
		W1.matrixSub(dW1);
			// std::cout << "!"  << std::endl;

		db1.matrixDot( lr / batch_size );
		b1.matrixSub( db1 );
	// std::cout << "!"  << std::endl;


		dW2.matrixDot( lr / batch_size );
		W2.matrixSub(dW2);
	// std::cout << "!"  << std::endl;


		db2.matrixDot( lr / batch_size );
		b2.matrixSub( db2 );
	// std::cout << "!"  << std::endl;

	};


	for (int epoch=1;epoch<=num_epochs;++epoch) {
		
		
		for (int i=0;i<num_examples;i+=batch_size) {
			int l = i, r = std::min(num_examples, i + batch_size);
			
			// std::cout << "!"  << std::endl;
			forward(l, r);
			// std::cout << "!!"  << std::endl;
			backward();
			// std::cout << "!!!"  << std::endl;
			sgd();
			// std::cout << "!!!!"  << std::endl;
		}

		if (epoch%100==0) {
			double loss = Loss(0, num_examples, W1, b1, W2, b2) / num_examples;
			printf("epoch = %d/%d, loss = %f\n", epoch, num_epochs, loss);
		}
		// W1.print();
	}


	// W1.print(); b1.print();
	// W2.print(); b.print();

	return 0;
}