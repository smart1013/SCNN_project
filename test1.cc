#include <iostream>
#include "tensor.h"

int main() {
    Scnn::TensorDims input_dims;
    input_dims.n = 1;
    input_dims.c = 3;
    input_dims.h = 224;
    input_dims.w = 224;
    
    Scnn::Tensor input_activation(input_dims);
    input_activation.set_random(0.0, 1.0, 0.1); 
    float val1 = input_activation.get_value(0, 2, 223, 223);

    std::cout << "val1: " << val1 << std::endl;
    
}