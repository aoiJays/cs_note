#ifndef _LOSS_H_
#define _LOSS__H_

#include <algorithm>

#include "matrix.h"
class Loss {

    public:
		static void MSE(Matrix & dst, Matrix & y, Matrix & yhat);
		static void dMSE(Matrix & dst, Matrix & y, Matrix & yhat);
		static void CrossEntropy(Matrix & dst, Matrix & y, Matrix & yhat);
		static void dCrossEntropy_2_softmax(Matrix & dst, Matrix & y, Matrix & yhat);
};


#endif