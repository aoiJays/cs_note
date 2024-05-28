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
        if (Activation_type == "linear") {
            actFun = &Activation::linear; 
            dactFun = &Activation::dlinear;
        }
        else if ( Activation_type == "ReLU" ){
            actFun = &Activation::ReLU; 
            dactFun = &Activation::dReLU;
        }
        else if ( Activation_type == "softmax" ){
            actFun = &Activation::softmax; 
            dactFun = &Activation::dsoftmax;
        }
        else assert(false);
    }

Layer::Layer(int intputNodeNum, int batch_size) : 
    A(intputNodeNum, batch_size),  AT(batch_size, intputNodeNum), dA(intputNodeNum, batch_size), nodeNum(intputNodeNum) {}
    

LossLayer::LossLayer(int lastNodeNum, int NodeNum, int batch_size, std::string Activation_type) : 
    A(NodeNum, batch_size), Z(NodeNum, batch_size) {
        if (Activation_type == "linear") {
            actFun = &Activation::linear; 
            dactFun = &Activation::dlinear;
        }
        else if ( Activation_type == "ReLU" ){
            actFun = &Activation::ReLU; 
            dactFun = &Activation::dReLU;
        }
        else if ( Activation_type == "softmax" ){
            actFun = &Activation::softmax; 
            dactFun = &Activation::dsoftmax;
        }
        else assert(false);
    }

LossLayer::LossLayer(int intputNodeNum, int batch_size) : A(intputNodeNum, batch_size) {}
    


void MLP::setInputLayer(int intputNodeNum) {
    assert( seq.size() == 0 );
    seq.push_back( std::make_unique<Layer>(intputNodeNum, batch_size) );
    seq_in_loss.push_back( std::make_unique<LossLayer>(intputNodeNum, total_size) );
}

void MLP::addLayer(int NodeNum, std::string Activation_type) {

    assert( seq.size() > 0 ); int last_node_num = seq.back()->nodeNum;
    seq.push_back( std::make_unique<Layer>(last_node_num, NodeNum, batch_size,Activation_type) );

    seq_in_loss.push_back( std::make_unique<LossLayer>(last_node_num, NodeNum, total_size,Activation_type) );
}

void MLP::forward( Matrix & train, int l, int r) {
    
    // 填充数据进入A0
    auto it = seq.begin();

    (*it)->A.matrixSubmatrix(train, l, r);
    (*it)->AT.matrixTrans( (*it)->A );

    // 从第一个隐藏层开始遍历
    for (++it;it!=seq.end();++it) {
        
        auto lastLayer = std::prev(it);
        
        (*it)->Z.matrixMul( (*it)->W, (*lastLayer)->A ); 
        (*it)->Z.matrixAdd( (*it)->b );

        // 激活函数
        (*it)->actFun((*it)->A, (*it)->Z);
        (*it)->dactFun((*it)->sigmadZ, (*it)->Z);


        (*it)->AT.matrixTrans( (*it)->A );
        (*it)->WT.matrixTrans( (*it)->W );
    }

}

double MLP::backward( Matrix & label, int l, int r){


    static Matrix yhat(label.n, r - l), y(label.n, r - l);
    y.matrixSubmatrix(label, l, r);

    static Matrix sumLoss(label.n, r - l);

    auto last_it = seq.rbegin();
    for (auto it=seq.rbegin();(it+1)!=seq.rend();++it) {

        //求dA dZ
        if ( it == seq.rbegin() ) {

            // 使用损失函数进行填充
            yhat.matrixSubmatrix((*it)->A, 0, (*it)->A.m);
            dloss( (*it)->dA, y, yhat ); loss(sumLoss, y, yhat);
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

    double loss_sum = sumLoss.sum();
    return loss_sum;
}

double MLP::getLoss( Matrix & train,  Matrix & label) {

    // 填充数据进入A0
    auto it = seq_in_loss.begin();
    (*it)->A.matrixSubmatrix(train, 0, train.m);

    // 从第一个隐藏层开始遍历
    int cnt = 1;
    for (++it;it!=seq_in_loss.end();++it, ++cnt) {
        
        auto lastLayer = std::prev(it);
        
        // wx + b
        (*it)->Z.matrixMul( seq[cnt]->W, (*lastLayer)->A ); 
        (*it)->Z.matrixAdd( seq[cnt]->b );

        // 激活函数
        (*it)->actFun((*it)->A, (*it)->Z);

    }

    static Matrix yhat(label.n, label.m), y(label.n, label.m), sumLoss(label.n, label.m);
    y.matrixSubmatrix(label, 0, label.m);

    yhat.matrixSubmatrix(seq_in_loss.back()->A, 0, seq_in_loss.back()->A.m);

    loss(sumLoss, y, yhat);
    double loss_sum = sumLoss.sum();
    
    return loss_sum/total_size;
}

double MLP::getAccuracy( Matrix & train,  Matrix & label) {

    // 填充数据进入A0
    auto it = seq_in_loss.begin();
   
    (*it)->A.matrixSubmatrix(train, 0, train.m);

    int cnt = 1;
    for (++it;it!=seq_in_loss.end();++it, ++cnt) {
        
        auto lastLayer = std::prev(it);
        
        // wx + b
        (*it)->Z.matrixMul( seq[cnt]->W, (*lastLayer)->A ); 
        (*it)->Z.matrixAdd( seq[cnt]->b );

        // 激活函数
        (*it)->actFun((*it)->A, (*it)->Z);

    }

    static Matrix yhat(label.n, label.m), y(label.n, label.m);
    y.matrixSubmatrix(label, 0, label.m);
    yhat.matrixSubmatrix(seq_in_loss.back()->A, 0, seq_in_loss.back()->A.m);

    yhat.tocpu(); y.tocpu();
    int count = 0;
    for (int i=0;i<label.m;++i) {
        int p = 0;
        for (int j=0;j<label.n;++j)
            if ( yhat(j,i) > yhat(p, i) ) p = j;
        int p2 = 0;
        for (int j=0;j<label.n;++j)
            if ( y(j,i) > y(p2, i) ) p2 = j;
        count += p==p2;
    }

    return (1.0*count/label.m)*100;
}

MLP::MLP(double lr, int batch_size, int total_size, std::string loss_type) : lr(lr), batch_size(batch_size), total_size(total_size) {
    

    if (loss_type == "CrossEntropy") {
        loss = &Loss::CrossEntropy;
        dloss = &Loss::dCrossEntropy_2_softmax;
    }
    else {
        loss = &Loss::MSE;
        dloss = &Loss::dMSE;
    }

}