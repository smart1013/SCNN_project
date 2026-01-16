#include <iostream>
#include "tensor.h"
#include "loader.h"
#include "convlayer.h"
#include "pe.h"


int main() {

    Scnn::ConvLayer conv_layer;
    conv_layer.initialize();

    {
        Scnn::Loader loader;
        Scnn::PE pe;

        int filter_size = 0;
        for (int k = 0; k < Scnn::LayerConfig::K; k++) {
            for (int c = 0; c < Scnn::LayerConfig::C; c++) {
                loader.load_IA(conv_layer.IA, c);
                loader.Load_FW(conv_layer.FW, k, k + 1, c);

                /**************************************************************/
                for (int pe_num = 0; pe_num < Scnn::HardwareConfig::NUM_PE; pe_num++) {
                    Scnn::Input_Buffer* input_tile = loader.IA_buffers[pe_num];
                    
                    // std::cout << "Input tile size:" << "\t" << input_tile->size << std::endl;
                    // std::cout << "Filter weight size:" << "\t" << loader.weight_buffer.size << std::endl;

                    pe.cartesian_product(input_tile, &loader.weight_buffer, &conv_layer.OA);

                }
                /**************************************************************/

            }
        }
    }

    conv_layer.OA.print();



    // Scnn::Loader loader;
    // loader.load_IA(conv_layer.IA);


    // for (int pe = 0; pe < Scnn::HardwareConfig::NUM_PE; pe++) {
    //     Scnn::Input_Buffer* input_tile = loader.IA_buffers[pe];

    //     if (input_tile->size == 0) {
    //         continue;
    //     }
        
    //     for (int k = 0; k < Scnn::LayerConfig::K; k++) {



            
    //     }


    // }




}
