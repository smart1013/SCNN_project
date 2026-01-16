#ifndef CONV_LAYER_H_
#define CONV_LAYER_H_

#include "common.h"
#include "tensor.h"
#include "loader.h"

namespace Scnn {

class ConvLayer {
public:

    Scnn::Tensor IA;
    std::vector<Scnn::Tensor*> FW;
    Scnn::Tensor OA;

    ConvLayer();
    ~ConvLayer();

    void initialize();

};


}





#endif // CONV_LAYER_H_