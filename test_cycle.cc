
#include "scnn_cycle_model.h"
#include <iostream>
#include "tensor.h"
#include "loader.h"

int main() {
    // 1. Setup Data
    Scnn::TensorDims dim; dim.c=1; dim.h=4; dim.w=4;
    Scnn::Tensor input(dim);
    input.set_random(0,1,0.2, 42); 
    
    Scnn::Input_Buffer ia_buf;
    ia_buf.load_input_buffer(input);
    
    Scnn::Weight_Buffer w_buf;
    // Dummy weight
    w_buf.add_element(1.0, {0,0,0,0});
    w_buf.add_element(0.5, {0,0,0,1});
    w_buf.add_element(0.5, {0,0,1,0});
    w_buf.add_element(0.5, {0,0,1,1});
    
    Scnn::Tensor output(dim); // Output tensor
    
    // 2. Run Cycle Accurate PE
    std::cout << "Starting Cycle-Accurate PE..." << std::endl;
    Scnn::CycleAccuratePE pe;
    unsigned long long cycles = pe.Run(&ia_buf, &w_buf, &output);
    
    std::cout << "Simulation Complete." << std::endl;
    std::cout << "Total Cycles: " << cycles << std::endl;
    
    // output.print();
    return 0;
}
