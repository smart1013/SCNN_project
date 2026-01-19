#include <iostream>
#include "tensor.h"
#include "loader.h"
#include "convlayer.h"
#include "pe.h"
#include "dispatcher.h"


int main() {

    Scnn::ConvLayer conv_layer;
    conv_layer.initialize();

    {
        Scnn::Loader loader;
        Scnn::PE pe;

        int total_idle_count = 0;
        int total_count = 0;
        int cycle = 0;
        int ia_count = 0;
        int weight_count = 0;

        for (int k = 0; k < Scnn::LayerConfig::K; k+=Scnn::HardwareConfig::FILTERS_PER_GROUP) {
            for (int c = 0; c < Scnn::LayerConfig::C; c++) {
                loader.load_IA(conv_layer.IA, c);
                loader.Load_FW(conv_layer.FW, k, k + Scnn::HardwareConfig::FILTERS_PER_GROUP, c);

                /**************************************************************/
                for (int pe_num = 0; pe_num < Scnn::HardwareConfig::NUM_PE; pe_num++) {

                    Scnn::Input_Buffer* input_tile = loader.IA_buffers[pe_num];
                    Scnn::Weight_Buffer* weight_buf = &loader.weight_buffer;

                    Scnn::Dispatcher dispatcher;
                    dispatcher.set_buffers(input_tile, weight_buf);

                    while (!dispatcher.finished || dispatcher.output_valid) {
                        dispatcher.Cycle();
                        cycle++;

                        if (dispatcher.output_valid) {

                            if (dispatcher.latched_w_vec.size() < 4) {
                                weight_count++;
                            }
                            dispatcher.output_valid = false;   
                            
                        }
                    }
                }
                /**************************************************************/

            }
        }
        
        std::cout << "Total cycles:" << "\t" << cycle << std::endl;
        // std::cout << "IA count:" << "\t" << ia_count << std::endl;
        std::cout << "Weight count:" << "\t" << weight_count << std::endl;
    }
}
