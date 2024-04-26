#ifndef _MLP_H_
#define _MLP_H_


#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "matrix.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"

struct Layer {

    Matrix A, Z, AT, WT, W, b;
	Matrix dA, dZ, dW, db, sigmadZ;
    Activation* act;
    int nodeNum;
    
    Layer(int lastNodeNum, int NodeNum, int batch_size, std::string activation_type);
    Layer(int intputNodeNum, int batch_size);

    ~Layer() {
        if (act != NULL) delete act;
    }
};

struct LossLayer {

    Matrix A, Z, W, b;
    Activation* act;

    LossLayer(int lastNodeNum, int NodeNum, int batch_size, std::string activation_type);
    LossLayer(int intputNodeNum, int batch_size);

    ~LossLayer() {
        if (act != NULL) delete act;
    }
};

struct MLP {
    
    std::vector<Layer> seq;
    std::vector<LossLayer> seq_in_loss;
    
    Loss * loss;
    Optimizer * optimizer;
    
    void setInputLayer(int intputNodeNum, int batch_size, int total_size);
    void addLayer(int NodeNum, int batch_size, int total_size, std::string activation_type);

    void forward();
    void backward();


    MLP(std::string loss_function, std::string optimizer_type, double lr = 0.03 ) {
        
        if ( loss_function == "MSE" ) loss = new MSE;
        else loss = new MSE;

        if ( optimizer_type == "SGD" ) optimizer = new SGD;
        else optimizer = new SGD;
    }

    ~MLP() {
        if (loss!=NULL) delete loss;
        if (optimizer!=NULL) delete optimizer;
    }
};

#endif