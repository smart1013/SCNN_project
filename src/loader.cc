#include "common.h"
#include "tensor.h"
#include "loader.h"
#include <iostream>


namespace Scnn {
    

    Input_Buffer::Input_Buffer() {
        this->size = 0;
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
                Input_Element element;
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
        Input_Element element;
        element.value = value;
        element.addr = addr;
        buffer.push_back(element);
        size++;
    }


    Weight_Buffer::Weight_Buffer() {
        this->size = 0;
    }


    Weight_Buffer::~Weight_Buffer() {
        buffer.clear();
    }


    void Weight_Buffer::add_element(float value, std::tuple<int, int, int, int> addr) {
        Filter_Element element;
        element.value = value;
        element.addr = addr;
        buffer.push_back(element);
        size++;
    }


    void Weight_Buffer::print() {
        for (int i = 0; i < buffer.size(); i++) {
            std::tuple<int, int, int, int> addr = buffer[i].addr;
            int k, c, r, s;
            std::tie(k, c, r, s) = addr;
            float val = buffer[i].value;
            std::cout << "value:" << val << "\t" << "address (k,c,r,s):" << k << " " << c << " " << r << " " << s << std::endl;
        }
    }


    Loader::Loader() {
    }


    Loader::~Loader() {
        for (auto buffer : IA_buffers) {
            delete buffer;
        }
    }


    void Loader::load_IA(Scnn::Tensor& tensor){
        // 1. Initialize buffers for 64 PEs
        for (auto buffer : IA_buffers) {
            delete buffer;
        }
        IA_buffers.clear();
        IA_buffers.reserve(HardwareConfig::NUM_PE);
        for (int i = 0; i < HardwareConfig::NUM_PE; i++) {
            IA_buffers.push_back(new Input_Buffer());
        }
        
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
            IA_buffers[pe_index]->add_element(val, tensor.get_addr(i));
        }

        max_size = 0;
        for (auto& buffer : IA_buffers) {
            if (buffer->size > max_size) {
                max_size = buffer->size;
            }
        }
    }


    void Loader::load_IA(Scnn::Tensor& tensor, int target_channel) {
        // 1. Initialize buffers for 64 PEs
        for (auto buffer : IA_buffers) {
            delete buffer;
        }
        IA_buffers.clear();
        IA_buffers.reserve(HardwareConfig::NUM_PE);
        for (int i = 0; i < HardwareConfig::NUM_PE; i++) {
            IA_buffers.push_back(new Input_Buffer());
        }
        
        // 2. Calculate Tile Dimensions
        int grid_dim = 8; // sqrt(64)
        int h_chunk = (tensor.dims.h + grid_dim - 1) / grid_dim;
        int w_chunk = (tensor.dims.w + grid_dim - 1) / grid_dim;
        
        // 3. Iterate through pixels in the specific channel
        int channel_size = tensor.dims.h * tensor.dims.w;
        int start_index = target_channel * channel_size;
        int end_index = start_index + channel_size;

        for (int i = start_index; i < end_index; ++i) {
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
            IA_buffers[pe_index]->add_element(val, tensor.get_addr(i));
        }

        max_size = 0;
        for (auto& buffer : IA_buffers) {
            if (buffer->size > max_size) {
                max_size = buffer->size;
            }
        }
    }
    

    void Loader::Load_FW(std::vector<Scnn::Tensor*>& fw, int k_start, int k_end, int target_channel) {
        
        // 1. Clear the single weight buffer
        weight_buffer.buffer.clear();
        weight_buffer.size = 0;

        // 2. Iterate through the range of Filters (k)
        // assert(Scnn::LayerConfig::K == fw.size());
        k_end = std::min(k_end, (int)fw.size());
        for (int k = k_start; k < k_end; k++) {
            Scnn::Tensor* filter = fw[k];
            
            // Calculate range for this channel (S * R)
            // Note: filter dims are (C, H[=S], W[=R]) in 3D mode
            int channel_size = filter->dims.h * filter->dims.w; 
            int start_index = target_channel * channel_size;
            int end_index = start_index + channel_size;

            // 3. Iterate through pixels in the specific channel
            for (int i = start_index; i < end_index; ++i) {
                float val = filter->data[i];
                if (val == 0.0) continue; 

                // Get (c, r, s) from 3D tensor
                auto [c, r, s] = filter->get_addr(i);
                
                // Construct 4D address (k, c, r, s)
                std::tuple<int, int, int, int> w_addr = std::make_tuple(k, c, r, s);
                
                // Add to the SINGLE weight buffer
                weight_buffer.add_element(val, w_addr);
            }
        }
    }



}
