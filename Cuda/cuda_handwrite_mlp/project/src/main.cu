#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>


#include "mlp.h"
#include "matrix.h"

#define MNIST

int main() {


#ifdef MNIST
	const int num_examples = 60000;
	const int test_num_examples = 10000;
    const int len_X = 28*28, len_Y = 10;

    const double lr = 0.01;
    const int batch_size = 1024;
    const int num_epochs = 100;
    const int check_epochs = 2;

    std::ifstream input("../dataset/mnist");
    std::ifstream input2("../dataset/mnist_test");
    MLP mlp(lr, batch_size, test_num_examples, "CrossEntropyLoss");

    mlp.setInputLayer(len_X); // 输入层
    mlp.addLayer(128, "ReLU");
    mlp.addLayer(64, "ReLU");
    mlp.addLayer(10, "softmax");

#endif


    Matrix train_x(len_X, num_examples);
    Matrix train_y(len_Y, num_examples);


    Matrix test_x(len_X, test_num_examples);
    Matrix test_y(len_Y, test_num_examples);


    for (int i=0;i<num_examples;++i) {
        for (int j=0;j<len_X;++j) {
            input >> train_x(j, i);
        }
        for (int j=0;j<len_Y;++j) {
            input >> train_y(j, i);
        }
    }

    for (int i=0;i<test_num_examples;++i) {
        for (int j=0;j<len_X;++j) input2 >> test_x(j, i);
        for (int j=0;j<len_Y;++j) input2 >> test_y(j, i);
    }
    
	train_x.togpu();
	train_y.togpu();
    
    test_x.togpu();
	test_y.togpu();
    
    timeval startTime, endTime;
    auto print_Time = [&](const timeval &startTime, const timeval& endTime) {
        long long elapsedTimeMicro = (endTime.tv_sec - startTime.tv_sec) * 1000000LL + (endTime.tv_usec - startTime.tv_usec);
        double elapsedTimeSec = elapsedTimeMicro / 1000000.0; // 将微秒转换为秒
        printf("Training time: %.10lf seconds\n", elapsedTimeSec);
    };


    std::cout << "Start to Train" << std::endl;
    gettimeofday(&startTime, NULL);


    for (int epoch=0;epoch<num_epochs;++epoch) {
        if (epoch%check_epochs==0)
            std::cout << "epoch = " << epoch + 1 << "/" << num_epochs << ": "; std::cout.flush();

        float lossSum = 0;
        for (int i=0;i+batch_size<=num_examples;i+=batch_size) {
            int l = i, r = i + batch_size;
            mlp.forward(train_x, l, r);
            lossSum += mlp.backward(train_y, l, r);
        }        

        if (epoch%check_epochs==0) {
            std::cout << "loss = " << lossSum / num_examples << std::endl;
        }            
    }
    

    gettimeofday(&endTime, NULL);
    std::cout << "Finshed" << std::endl;
    print_Time(startTime, endTime);

    
    std::cout << "loss = " << mlp.getLoss(test_x, test_y) << std::endl;
    std::cout << "acc = " << std::fixed << std::setprecision(1) <<  mlp.getAccuracy(test_x, test_y) << "%" << std::endl;

    return 0;
}