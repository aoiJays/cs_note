#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <algorithm>
#include <cmath>


class Activation {
    
    virtual double sigma(double x) = 0;
    virtual double dsigma(double x) = 0;
};

class Linear: public Activation {

    double sigma(double x) {
        return x;
    }
    double dsigma(double x) {
        return 1;
    }
};



class Relu: public Activation {

    double sigma(double x) {
        return std::max(x, 0.0);
    }
    double dsigma(double x) {
        return x >=0 ? 1 : 0;
    }
};


#endif