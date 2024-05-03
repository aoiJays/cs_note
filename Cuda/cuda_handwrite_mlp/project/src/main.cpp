#include <iostream>
#include <vector>
#include <fstream>

#include "mlp.h"
#include "matrix.h"

int main() {

	// 数据集
	const int num_examples = 200;
	const int test_num_examples = 200;
    const int len_X = 1, len_Y = 1;

    const double lr = 0.0001;
    const int batch_size = num_examples;
    const int num_epochs = 200000;
    const int check_epochs = 1000;


    MLP mlp(lr, batch_size, test_num_examples, "MSE");

    mlp.setInputLayer(len_X); // 输入层
    mlp.addLayer(32, "ReLU");
    mlp.addLayer(32, "ReLU");
    mlp.addLayer(1, "Linear");


    Matrix train(len_X, num_examples);
    Matrix label(len_Y, num_examples);


    std::ifstream input("../dataset/data");

    for (int i=0;i<num_examples;++i) {
        for (int j=0;j<len_X;++j) input >> train(j, i);
        for (int j=0;j<len_Y;++j) input >> label(j, i);
    }

    
    for (int epoch=0;epoch<num_epochs;++epoch) {
        if (epoch%check_epochs==0)
            std::cout << "epoch = " << epoch + 1 << "/" << num_epochs << ": "; std::cout.flush();
        for (int i=0;i+batch_size<=num_examples;i+=batch_size) {
            
            int l = i, r = i + batch_size;
            mlp.forward(train, l, r);
            mlp.backward(label, l, r);
        }        

        if (epoch%check_epochs==0) {
            // std::cout << "!" << std::endl;
            std::cout << "loss = " << mlp.getLoss(train, label) << std::endl;
        }            
    }

    std::cout << "Finshed" << std::endl;
    std::cout << "loss = " << mlp.getLoss(train, label) << std::endl;
}