#include <iostream>
#include "tensor.h"
#include "loader.h"
#include "convlayer.h"
#include "pe.h"
#include "dispatcher.h"
#include "mult_array.h"
#include "buffer_queue.h"
#include "accumulator.h"

int main() {
    Scnn::TensorDims input_dims;
    input_dims.c = 3;
    input_dims.h = 10;
    input_dims.w = 10;

    Scnn::Tensor IA(input_dims);
    IA.load_from_file("ia.csv");
    // IA.print();

    Scnn::TensorDims filter_dims;
    filter_dims.c = 3;
    filter_dims.h = 3;
    filter_dims.w = 3;
    
    // We treat this single FW tensor as 1 filter (K=1)
    Scnn::Tensor* filter = new Scnn::Tensor(filter_dims);
    filter->load_from_file("fw1.csv");
    Scnn::Tensor* filter2 = new Scnn::Tensor(filter_dims);
    filter2->load_from_file("fw2.csv");
    Scnn::Tensor* filter3 = new Scnn::Tensor(filter_dims);
    filter3->load_from_file("fw3.csv");

    std::vector<Scnn::Tensor*> FW;
    FW.push_back(filter);
    FW.push_back(filter2);
    FW.push_back(filter3);

    // Initialize Output Tensor
    Scnn::TensorDims output_dims;
    output_dims.c = 3; 
    output_dims.h = 10; 
    output_dims.w = 10; 
    
    Scnn::Tensor OA(output_dims);

    // Convolution Simulation
    {
        Scnn::Loader loader;

        int total_idle_count = 0;
        int total_count = 0;
        int cycle = 0;
        int total_idle_cycle = 0;
        
        // Loop structure (K=3, C=3 based on our CSVs)
        for (int k = 0; k < 3; k+=Scnn::HardwareConfig::FILTERS_PER_GROUP) {
            for (int c = 0; c < 3; c++) {
                loader.load_IA(IA, c);
                loader.Load_FW(FW, k, k + Scnn::HardwareConfig::FILTERS_PER_GROUP, c);
                
                for (int pe_num = 0; pe_num < Scnn::HardwareConfig::NUM_PE; pe_num++) {
                    Scnn::Input_Buffer* input_tile = loader.IA_buffers[pe_num];
                    
                    Scnn::Dispatcher dispatcher;
                    Scnn::MultArray mult_array;
                    Scnn::BufferQueue buffer_queue;
                    Scnn::Accumulator accumulator;

                    dispatcher.set_buffers(input_tile, &loader.weight_buffer);

                    while (!dispatcher.finished || dispatcher.output_valid || mult_array.has_output() || !buffer_queue.is_empty()) {

                        cycle++;

                        accumulator.Cycle(&buffer_queue, &OA);
                        mult_array.Cycle(&dispatcher, &buffer_queue, &OA);
                        dispatcher.Cycle();
                    }

                    total_idle_count += mult_array.idle_count;
                    total_count += mult_array.total_mults_count;
                    total_idle_cycle += mult_array.idle_cycle;
                }
            }
        }

        std::cout << "Total cycles:" << "\t" << cycle << std::endl;
        std::cout << "Idle counts:" << "\t" << total_idle_count << std::endl;
        std::cout << "Total count:" << "\t" << total_count << std::endl;
        std::cout << "Multiplier Utilization:" << "\t" << 1.0 - (float)total_idle_count / (total_count) << std::endl;
        std::cout << "Idle cycles:" << "\t" << total_idle_cycle << std::endl;

    }

    std::cout << "Output Tensor (OA):" << std::endl;
    OA.print();
    
    // Print non-zero values to check correctness
    std::cout << "Values:" << std::endl;
    for(int i=0; i<OA.data.size(); ++i) {
        auto [c, h, w] = OA.get_addr(i);
        std::cout << "(" << c << "," << h << "," << w << "): " << OA.data[i] << std::endl;
    }

    delete filter;
    delete filter2;
    delete filter3;

}
