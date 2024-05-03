#ifndef _MLP_H_
#define _MLP_H_


#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "matrix.h"
#include "activation.h"

struct Layer {

    Matrix A, Z, AT, WT, W, b;
	Matrix dA, dZ, dW, db, sigmadZ;
    int nodeNum;
    
    typedef double (*ActivationFunction)(double);
    ActivationFunction actFun, dactFun;

    Layer(int lastNodeNum, int NodeNum, int batch_size, std::string Activation_type);
    Layer(int intputNodeNum, int batch_size);

};

struct LossLayer {

    Matrix A, Z;

    typedef double (*ActivationFunction)(double);
    ActivationFunction actFun, dactFun;

    LossLayer(int lastNodeNum, int NodeNum, int batch_size, std::string Activation_type);
    LossLayer(int intputNodeNum, int batch_size);

};

struct MLP {
    
    std::vector< std::unique_ptr<Layer> > seq;
    std::vector< std::unique_ptr<LossLayer> > seq_in_loss;

    typedef double (*LossFunction)(double, double);
    LossFunction loss, dloss;

    double lr; int batch_size, total_size;

    void setInputLayer(int intputNodeNum);
    void addLayer(int NodeNum,std::string Activation_type );

    void forward(const Matrix & train, int l, int r);
    void backward(const Matrix & train, int l, int r);
    double getLoss(const Matrix & train, const Matrix & label);

    MLP(double lr, int batch_size, int total_size, std::string loss_type);

};

#endif