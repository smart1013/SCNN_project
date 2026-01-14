#include <iostream>
#include "tensor.h"
#include "loader.h"
#include "convlayer.h"

int main() {
    // Scnn::TensorDims input_dims, filter_dims;

    // input_dims.n = Scnn::LayerConfig::N;
    // input_dims.c = Scnn::LayerConfig::C;
    // input_dims.h = Scnn::LayerConfig::H;
    // input_dims.w = Scnn::LayerConfig::W;

    // filter_dims.n = Scnn::LayerConfig::N;
    // filter_dims.c = Scnn::LayerConfig::C;
    // filter_dims.h = Scnn::LayerConfig::S;
    // filter_dims.w = Scnn::LayerConfig::R;
    
    // Scnn::Tensor input_activation(input_dims);
    // Scnn::Tensor filter_weight(filter_dims);
    // input_activation.set_random(0.0, 1.0, 0.5); 
    // filter_weight.set_random(0.0, 1.0, 0.5); 

    Scnn::ConvLayer conv_layer;
    conv_layer.initialize();

    conv_layer.IA.print();
    for (int i = 0; i < conv_layer.FW.size(); i++) {
        conv_layer.FW[i]->print();
    }

    Scnn::Loader loader;
    loader.load_IA(conv_layer.IA);

    // input_activation.print();
    // filter_weight.print();

    // Scnn::Input_Buffer weight_buffer;
    // weight_buffer.load_input_buffer(filter_weight);

    // Scnn::Loader loader;
    // loader.load_IA(conv_layer.IA);

    // int a = 0;
    // for (auto& buffer : loader.IA_buffers) {
    //     std::cout << "size: " << buffer->size << std::endl;
    //     a += buffer->size;
    // }
    // std::cout << "total size: " << a << std::endl;
    // std::cout << "non_zeros: " << conv_layer.IA.non_zero_count << std::endl;

    


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