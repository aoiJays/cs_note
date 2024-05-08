#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <algorithm>
#include "matrix.h"

class Activation {

    public:

    	static void linear(Matrix & a,  Matrix & b);

    	static void dlinear(Matrix & a,  Matrix & b);
	
    	static void ReLU(Matrix & a,  Matrix & b);

    	static void dReLU(Matrix & a,  Matrix & b);

    	static void softmax(Matrix & a,  Matrix & b);

    	static void dsoftmax(Matrix & a,  Matrix & b);

};


#endif