#include <iostream>
#include "tensor.h"
#include "loader.h"

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
    input_activation.set_random(0.0, 1.0, 0.5); 
    filter_weight.set_random(0.0, 1.0, 0.5); 

    input_activation.print();
    filter_weight.print();

    Scnn::Input_Buffer weight_buffer;
    weight_buffer.load_input_buffer(filter_weight);

    // for (int i = 0; i < filter_weight.get_size(); i++) {
    //     std::tuple<int, int, int, int> addr = filter_weight.get_addr(i);
    //     int n, c, h, w;
    //     std::tie(n, c, h, w) = addr;
    //     float val = filter_weight.get_value(n, c, h, w);
    //     std::cout << "value:" << val << "\t" << "address:" << n << " " << c << " " << h << " " << w << std::endl;
    // }

}