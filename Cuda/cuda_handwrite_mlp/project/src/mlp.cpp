#include "mlp.h"

#include <algorithm>
#include <cmath>

#include "matrix.h"
#include "activation.h"

Layer::Layer(int lastNodeNum, int NodeNum, int batch_size, std::string activation_type = "Relu") : 
    A(NodeNum, batch_size), Z(NodeNum, batch_size), AT(batch_size, NodeNum), W(NodeNum, lastNodeNum), WT(lastNodeNum, NodeNum), b(NodeNum, 1),
    dA(NodeNum, batch_size), dZ(NodeNum, batch_size), dW(NodeNum, lastNodeNum), db(NodeNum, 1), sigmadZ(NodeNum, batch_size), nodeNum(NodeNum) {

        if (activation_type == "Relu") act = new Relu;
        else if (activation_type == "Linear") act = new Linear;
    }

Layer::Layer(int intputNodeNum, int batch_size) : 
    A(intputNodeNum, batch_size),  AT(batch_size, intputNodeNum), dA(intputNodeNum, batch_size), nodeNum(intputNodeNum) {}
    

LossLayer::LossLayer(int lastNodeNum, int NodeNum, int batch_size, std::string activation_type = "Relu") : 
    A(NodeNum, batch_size), Z(NodeNum, batch_size), W(NodeNum, lastNodeNum), b(NodeNum, 1) {

        if (activation_type == "Relu") act = new Relu;
        else if (activation_type == "Linear") act = new Linear;
    }

LossLayer::LossLayer(int intputNodeNum, int batch_size) : A(intputNodeNum, batch_size) {}
    


void MLP::setInputLayer(int intputNodeNum, int batch_size, int total_size) {
    seq.push_back( Layer(intputNodeNum, batch_size) ); 
    seq_in_loss.push_back( LossLayer(intputNodeNum, total_size) ); 
}

void MLP::addLayer(int NodeNum, int batch_size, int total_size, std::string activation_type = "Relu") {
    assert( seq.size() > 0 ); int last_node_num = seq.back().nodeNum;
    seq.push_back( Layer( last_node_num , NodeNum, batch_size, activation_type) ); 
    seq_in_loss.push_back( LossLayer(last_node_num , NodeNum, total_size, activation_type) ); 
}

void MLP::forward() {

}

void MLP::backward() {

    
}