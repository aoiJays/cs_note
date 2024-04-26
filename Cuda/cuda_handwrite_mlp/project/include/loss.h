#ifndef _LOSS_H_
#define _LOSS_H_

#include <algorithm>
#include <cmath>


class Loss {
    
    virtual double cost(double yhat, double y) = 0;
    virtual double dcost(double yhat, double y) = 0;
};

class MSE: public Loss {

    double cost(double yhat, double y) {
        return (yhat-y) * (yhat - y) * 0.5;
    }

    double dcost(double yhat, double y) {
        return (yhat - y);
    }
};




#endif