#include <iostream>
#include "tensor.h"

int main() {
    Scnn::TensorDims input_dims, filter_dims;
    input_dims.n = 1;
    input_dims.c = 3;
    input_dims.h = 224;
    input_dims.w = 224;

    filter_dims.n = 1;
    filter_dims.c = 3;
    filter_dims.h = 11;
    filter_dims.w = 11;
    
    Scnn::Tensor input_activation(input_dims);
    Scnn::Tensor filter_weight(filter_dims);
    input_activation.set_random(0.0, 1.0, 0.1); 
    filter_weight.set_random(0.0, 1.0, 0.1); 

    // float val1 = input_activation.get_value(0, 30, 223, 223);
    // float val2 = filter_weight.get_value(0, 30, 5, 5);

    // std::cout << "val1: " << val1 << std::endl;
    // std::cout << "val2: " << val2 << std::endl;
    input_activation.print();
}