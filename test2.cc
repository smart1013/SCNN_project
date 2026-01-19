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

        int total_idle_count = 0;
        int total_count = 0;

        for (int k = 0; k < Scnn::LayerConfig::K; k+=Scnn::HardwareConfig::FILTERS_PER_GROUP) {
            for (int c = 0; c < Scnn::LayerConfig::C; c++) {
                loader.load_IA(conv_layer.IA, c);
                loader.Load_FW(conv_layer.FW, k, k + Scnn::HardwareConfig::FILTERS_PER_GROUP, c);

                /**************************************************************/
                for (int pe_num = 0; pe_num < Scnn::HardwareConfig::NUM_PE; pe_num++) {
                    Scnn::Input_Buffer* input_tile = loader.IA_buffers[pe_num];

                    auto [idle_count, count] = pe.cartesian_product(input_tile, &loader.weight_buffer, &conv_layer.OA);
                    total_idle_count += idle_count;
                    total_count += count;

                }
                /**************************************************************/

            }
        }
        conv_layer.OA.print();

        std::cout << "Idle counts:" << "\t" << total_idle_count << std::endl;
        std::cout << "Total counts:" << "\t" << total_count << std::endl;
        std::cout << "Multiplier Utilization:" << "\t" << 1.0 - (float)total_idle_count / (total_count) << std::endl;
    }
}
