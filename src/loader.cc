#include "common.h"
#include "tensor.h"
#include "loader.h"
#include <iostream>


namespace Scnn {
    
    Input_Buffer::Input_Buffer() {
    }

    Input_Buffer::~Input_Buffer() {
        buffer.clear();
    }

    void Input_Buffer::load_input_buffer(Scnn::Tensor& tensor) {
        // buffer.clear();
        buffer.reserve(tensor.non_zero_count);
        // buffer.resize(tensor.non_zero_count);
        this->size = tensor.non_zero_count;

        for (int i = 0; i < tensor.data.size(); i++) {
            std::tuple<int, int, int> addr = tensor.get_addr(i);
            int c, h, w;
            std::tie(c, h, w) = addr;
            float value = tensor.get_value(c, h, w);
            if (value != 0.0) {
                Element element;
                element.value = value;
                element.addr = addr;
                buffer.push_back(element);
            }
        }
    }

    void Input_Buffer::print() {
        for (int i = 0; i < buffer.size(); i++) {
            std::tuple<int, int, int> addr = buffer[i].addr;
            int c, h, w;
            std::tie(c, h, w) = addr;
            float val = buffer[i].value;
            std::cout << "value:" << val << "\t" << "address:" << c << " " << h << " " << w << std::endl;
        }
    }

    void Input_Buffer::add_element(float value, std::tuple<int, int, int> addr) {
        Element element;
        element.value = value;
        element.addr = addr;
        buffer.push_back(element);
        size++;
    }

    Loader::Loader() {
    }

    Loader::~Loader() {
        pe_buffers.clear();
    }

    void Loader::load_IA(Scnn::Tensor& tensor) {
        // 1. Initialize buffers for 64 PEs
        pe_buffers.clear();
        pe_buffers.resize(HardwareConfig::NUM_PE); // 64
        
        // 2. Calculate Tile Dimensions
        int grid_dim = 8; // sqrt(64)
        int h_chunk = (tensor.dims.h + grid_dim - 1) / grid_dim;
        int w_chunk = (tensor.dims.w + grid_dim - 1) / grid_dim;
        
        // 3. Iterate through every pixel in global tensor
        for (int i = 0; i < tensor.data.size(); ++i) {
            float val = tensor.data[i];
            if (val == 0.0) continue; // Skip zeros (SCNN optimization)
            
            // Decode address
            auto [c, h, w] = tensor.get_addr(i);
            
            // 4. Find which PE owns this pixel
            int pe_r = h / h_chunk; 
            int pe_c = w / w_chunk;
            
            // Safety clamp
            if (pe_r >= grid_dim) pe_r = grid_dim - 1;
            if (pe_c >= grid_dim) pe_c = grid_dim - 1;
            
            int pe_index = pe_r * grid_dim + pe_c;
            
            // 5. Add to that PE's specific buffer
            pe_buffers[pe_index].add_element(val, tensor.get_addr(i));
        }

        max_size = 0;
        for (auto& buffer : pe_buffers) {
            if (buffer.size > max_size) {
                max_size = buffer.size;
            }
        }
    }
    

    



}
