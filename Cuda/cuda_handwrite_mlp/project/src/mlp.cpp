#include "mlp.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

#include "matrix.h"
#include "activation.h"
#include "loss.h"

Layer::Layer(int lastNodeNum, int NodeNum, int batch_size, std::string Activation_type) : 
    A(NodeNum, batch_size), Z(NodeNum, batch_size), AT(batch_size, NodeNum), W(NodeNum, lastNodeNum), WT(lastNodeNum, NodeNum), b(NodeNum, 1),
    dA(NodeNum, batch_size), dZ(NodeNum, batch_size), dW(NodeNum, lastNodeNum), db(NodeNum, 1), sigmadZ(NodeNum, batch_size), nodeNum(NodeNum) {
        if (Activation_type == "Linear") {
            actFun = &Activation::linear; 
            dactFun = &Activation::dlinear;
        }
        else {
            actFun = &Activation::Relu; 
            dactFun = &Activation::dRelu;
        }
    }

Layer::Layer(int intputNodeNum, int batch_size) : 
    A(intputNodeNum, batch_size),  AT(batch_size, intputNodeNum), dA(intputNodeNum, batch_size), nodeNum(intputNodeNum) {}
    

LossLayer::LossLayer(int lastNodeNum, int NodeNum, int batch_size, std::string Activation_type) : 
    A(NodeNum, batch_size), Z(NodeNum, batch_size) {
        if (Activation_type == "Linear") {
            actFun = &Activation::linear; 
            dactFun = &Activation::dlinear;
        }
        else {
            actFun = &Activation::Relu; 
            dactFun = &Activation::dRelu;
        }
    }

LossLayer::LossLayer(int intputNodeNum, int batch_size) : A(intputNodeNum, batch_size) {}
    


void MLP::setInputLayer(int intputNodeNum) {
    assert( seq.size() == 0 );
    seq.push_back( std::make_unique<Layer>(intputNodeNum, batch_size) );
    seq_in_loss.push_back( std::make_unique<LossLayer>(intputNodeNum, total_size) );
}

void MLP::addLayer(int NodeNum, std::string Activation_type) {

    assert( seq.size() > 0 ); int last_node_num = seq.back()->nodeNum;
    // std::cout << "Add " << NodeNum << " " << last_node_num << std::endl; 
    seq.push_back( std::make_unique<Layer>(last_node_num, NodeNum, batch_size,Activation_type) );

    // std::cout << seq.back()->W.n << " " << seq.back()->W.m << std::endl; 
    seq_in_loss.push_back( std::make_unique<LossLayer>(last_node_num, NodeNum, total_size,Activation_type) );
}

void MLP::forward(const Matrix & train, int l, int r) {
    
    // 填充数据进入A0
    auto it = seq.begin();

    (*it)->A.matrixSubmatrix(train, l, r);
    // for (int i=l;i<r;++i) {
    //     for (int j=0;j<(*it)->nodeNum;++j) 
    //         (*it)->A(j,i-l) = train.matrix[ j * train.m + i ];
    // }

    (*it)->AT.matrixTrans( (*it)->A );

    // 从第一个隐藏层开始遍历
    for (++it;it!=seq.end();++it) {
        
        auto lastLayer = std::prev(it);
        
        // // wx + b
        // std::cout << "!" << std::endl;

        // std::cout << (*lastLayer)->A.n << " " << (*lastLayer)->A.m << std::endl;
        // std::cout << (*it)->W.n << " " << (*it)->W.m << std::endl;

        (*it)->Z.matrixMul( (*it)->W, (*lastLayer)->A ); 

        (*it)->Z.matrixAdd( (*it)->b );

        

        // 激活函数
        (*it)->A.matrixFun( (*it)->Z, (*it)->actFun );
        (*it)->sigmadZ.matrixFun( (*it)->Z, (*it)->dactFun );

        (*it)->AT.matrixTrans( (*it)->A );
        (*it)->WT.matrixTrans( (*it)->W );
    }

}

void MLP::backward(const Matrix & label, int l, int r){


    static Matrix yhat(label.n, r - l), y(label.n, r - l);

    y.matrixSubmatrix(label, l, r);


    auto last_it = seq.rbegin();
    for (auto it=seq.rbegin();(it+1)!=seq.rend();++it) {

        //求dA dZ
        if ( it == seq.rbegin() ) {
            // 使用损失函数进行填充

            yhat.matrixSubmatrix((*it)->A, 0, (*it)->A.m);
            (*it)->dA.matrixFun2(y, yhat, dloss);
            // for (int i=l;i<r;++i) {

                // double yhat = (*it)->A(0, i-l);
                // double y = train.matrix[ (train.n - 1) * train.m + i ];
                // (*it)->dA(0, i-l) = dloss(y, yhat);
            // }
        }
        else (*it)->dA.matrixMul( (*last_it)->WT, (*last_it)->dZ );
        (*it)->dZ.matrixDot((*it)->dA, (*it)->sigmadZ);

        // db
        (*it)->db.matrixReduce((*it)->dZ);

        // dw
        (*it)->dW.matrixMul((*it)->dZ, (*next(it))->AT);
        last_it = it;
    }
    
    // SGD


    for (auto it=seq.begin() + 1;it!=seq.end();++it) {

        (*it)->db.matrixDot(lr/batch_size);
        (*it)->dW.matrixDot(lr/batch_size);

        (*it)->W.matrixSub( (*it)->dW );
        (*it)->b.matrixSub( (*it)->db );
    }

}

double MLP::getLoss(const Matrix & train, const Matrix & label) {

    // 填充数据进入A0
    auto it = seq_in_loss.begin();


    // std::cout << train.n << " " << train.m << "\n";
            // std::cout << "---" << std::endl;

    
    (*it)->A.matrixSubmatrix(train, 0, train.m);

    
            // std::cout << "---" << std::endl;

    // for (int i=0;i<train.m;++i) {
    //     for (int j=0;j<train.n-1;++j) 
    //         (*it)->A(j,i) = train.matrix[ j * train.m + i ];
    // }

    // 从第一个隐藏层开始遍历
    int cnt = 1;
    for (++it;it!=seq_in_loss.end();++it, ++cnt) {
        
        auto lastLayer = std::prev(it);
        
        // wx + b
        (*it)->Z.matrixMul( seq[cnt]->W, (*lastLayer)->A ); 
        (*it)->Z.matrixAdd( seq[cnt]->b );

        // 激活函数
        (*it)->A.matrixFun( (*it)->Z, (*it)->actFun );

    }

    
    double loss_sum = 0;

    static Matrix yhat(label.n, label.m), y(label.n, label.m), sumLoss(label.n, label.m);
    y.matrixSubmatrix(label, 0, label.m);

    yhat.matrixSubmatrix(seq_in_loss.back()->A, 0, seq_in_loss.back()->A.m);

    sumLoss.matrixFun2(y, yhat, loss);
    
    

    std::ofstream output("../dataset/data.csv", std::ofstream::trunc);
    for (int i=0;i<label.m;++i) {
        for (int j=0;j<label.n;++j) loss_sum += sumLoss(j,i);
        output << train.matrix[ 0 * train.m + i ] << " " << seq_in_loss.back()->A(0, i) << "\n";
        // std::cout << train.matrix[ 0 * train.m + i ] << " " << seq_in_loss.back()->A(0, i) << "\n";
        // std::cout << loss(seq_in_loss.back()->A(0, i), train.matrix[ (train.n - 1) * train.m + i ]) << "\n";
    }
    
    return loss_sum/total_size;
}

MLP::MLP(double lr, int batch_size, int total_size, std::string loss_type) : lr(lr), batch_size(batch_size), total_size(total_size) {
    

    if (loss_type == "CrossEntropyLoss") {
        // loss = &Loss::CrossEntropyLoss;
        // dloss = &Loss::dCrossEntropyLoss;
    }
    else {
        loss = &Loss::MSE;
        dloss = &Loss::dMSE;
    }

}