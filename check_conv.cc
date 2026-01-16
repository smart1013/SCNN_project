#include <iostream>
#include "tensor.h"
#include "loader.h"
#include "convlayer.h"
#include "pe.h"

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
    output_dims.h = 10; // Assuming same as input for now (10x10 with padding)
    output_dims.w = 10; 
    
    Scnn::Tensor OA(output_dims);

    // Convolution Simulation
    {
        Scnn::Loader loader;
        Scnn::PE pe;
        
        // Loop structure (K=3, C=3 based on our CSVs)
        for (int k = 0; k < 3; k++) {
            for (int c = 0; c < 3; c++) {
                loader.load_IA(IA, c);
                loader.Load_FW(FW, k, k + 1, c);
                
                for (int pe_num = 0; pe_num < Scnn::HardwareConfig::NUM_PE; pe_num++) {
                    Scnn::Input_Buffer* input_tile = loader.IA_buffers[pe_num];
                    pe.cartesian_product(input_tile, &loader.weight_buffer, &OA);
                }
            }
        }
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
