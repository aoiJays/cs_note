#ifndef _LOSS_H_
#define _LOSS__H_

#include <algorithm>

class Loss {

    public:
    	static double MSE(double y, double yhat) {
    	    return (y-yhat)*(y-yhat) * 0.5;
    	}
	
        static double dMSE(double y, double yhat) {
    	    return yhat - y;
    	}

};


#endif