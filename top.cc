#include <iostream>
#include "tensor.h"
#include "loader.h"
#include "convlayer.h"
#include "pe.h"
#include "dispatcher.h"
#include "mult_array.h"


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
                    Scnn::MultArray mult_array;

                    dispatcher.set_buffers(input_tile, weight_buf);

                    while (!dispatcher.finished || dispatcher.output_valid) {

                        cycle++;

                        mult_array.Cycle(&dispatcher, &conv_layer.OA);
                        dispatcher.Cycle();
                    }

                    total_idle_count += mult_array.idle_count;
                    total_count += mult_array.total_mults_count;
                }
                /**************************************************************/

            }
        }
        
        std::cout << "Total cycles:" << "\t" << cycle << std::endl;
        // std::cout << "IA count:" << "\t" << ia_count << std::endl;
        // std::cout << "Weight count:" << "\t" << weight_count << std::endl;
        std::cout << "Idle counts:" << "\t" << total_idle_count << std::endl;
        std::cout << "Total count:" << "\t" << total_count << std::endl;
        std::cout << "Multiplier Utilization:" << "\t" << 1.0 - (float)total_idle_count / (total_count) << std::endl;
    }
}
