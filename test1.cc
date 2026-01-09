#include <iostream>
#include "tensor.h"
#include "loader.h"

int main() {
    Scnn::TensorDims input_dims, filter_dims;
    input_dims.n = 1;
    input_dims.c = 100;
    input_dims.h = 224;
    input_dims.w = 224;

    filter_dims.n = 1;
    filter_dims.c = 100;
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

    Scnn::Loader loader;
    loader.load_IA(input_activation);

    int a = 0;
    for (auto& buffer : loader.pe_buffers) {
        std::cout << "size: " << buffer.size << std::endl;
        a += buffer.size;
    }
    std::cout << "a: " << a << std::endl;
    std::cout << "input_activation.size: " << input_activation.non_zero_count << std::endl;





    // for (int i = 0; i < filter_weight.get_size(); i++) {
    //     std::tuple<int, int, int, int> addr = filter_weight.get_addr(i);
    //     int n, c, h, w;
    //     std::tie(n, c, h, w) = addr;
    //     float val = filter_weight.get_value(n, c, h, w);
    //     std::cout << "value:" << val << "\t" << "address:" << n << " " << c << " " << h << " " << w << std::endl;
    // }

    // for (int i = 0; i < weight_buffer.size; i++) {
    //     std::tuple<int, int, int, int> addr = weight_buffer.buffer[i].addr;
    //     int n, c, h, w;
    //     std::tie(n, c, h, w) = addr;
    //     float val = weight_buffer.buffer[i].value;
    //     std::cout << "value:" << val << "\t" << "address:" << n << " " << c << " " << h << " " << w << std::endl;
    // }


    // for (int c = 0; c < filter_weight.dims.c; c++) {
    //     std::cout << "c: " << c << std::endl;
    //     int starting_index = c * filter_weight.dims.h * filter_weight.dims.w;
    //     int ending_index = (c + 1) * filter_weight.dims.h * filter_weight.dims.w;
    //     for (int i = starting_index; i < ending_index; i++) {
    //         std::tuple<int, int, int, int> addr = filter_weight.get_addr(i);
    //         int n, c, h, w;
    //         std::tie(n, c, h, w) = addr;
    //         float val = filter_weight.get_value(n, c, h, w);
    //         std::cout << "value:" << val << "\t" << "address:" << n << " " << c << " " << h << " " << w << std::endl;
    //     }
    // }

}