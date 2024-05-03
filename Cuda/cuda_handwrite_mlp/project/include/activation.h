#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <algorithm>

class Activation {

    public:
    	static double linear(double x) {
    	    return x;
    	}
	
    	static double dlinear(double x) {
    	    return 1;
    	}
	
    	static double Relu(double x) {
    	    return std::max(x, 0.0);
    	}
	
    	static double dRelu(double x) {
    	    return x >=0 ? 1 : 0;
    	}

};


#endif