#include "dispatcher.h"
#include "common.h"

namespace Scnn {
    
    Dispatcher::Dispatcher() {
        input_buffer_ptr = nullptr;
        weight_buffer_ptr = nullptr;
        reset();
    }

    Dispatcher::~Dispatcher() {
    }

    void Dispatcher::reset() {
        curr_ia_idx = 0;
        curr_w_idx = 0;
        finished = false;
        output_valid = false;
        latched_ia_vec.clear();
        latched_w_vec.clear();
    }


    void Dispatcher::set_buffers(Scnn::Input_Buffer* ia_buff, Scnn::Weight_Buffer* w_buff) {
        input_buffer_ptr = ia_buff;
        weight_buffer_ptr = w_buff;
        reset();
    }


    void Dispatcher::Cycle() {
        // stall condition
        if (output_valid) {
            return;
        }

        // end condition
        if (finished || !input_buffer_ptr || !weight_buffer_ptr) {
            return;
        }

        latched_ia_vec.clear();
        latched_w_vec.clear();

        int ia_size = input_buffer_ptr->size;
        int w_size = weight_buffer_ptr->size;

        // fetch input vector
        for (int k = 0; k < Scnn::HardwareConfig::IA_VECTOR_SIZE; ++k) {
            if (curr_ia_idx + k < ia_size) {
                latched_ia_vec.push_back(input_buffer_ptr->buffer[curr_ia_idx + k]);
            }
        }

        // fetch weight vector
        for (int k = 0; k < Scnn::HardwareConfig::W_VECTOR_SIZE; ++k) {
            if (curr_w_idx + k < w_size) {
                latched_w_vec.push_back(weight_buffer_ptr->buffer[curr_w_idx + k]);
            }
        }

        // output valid
        output_valid = true;

        // update index
        curr_w_idx += Scnn::HardwareConfig::W_VECTOR_SIZE;

        if (curr_w_idx >= w_size) {
            curr_ia_idx += Scnn::HardwareConfig::IA_VECTOR_SIZE;
            curr_w_idx = 0;

            if (curr_ia_idx >= ia_size) {
                finished = true;
            }
        }
        
    }

    bool Dispatcher::is_output_valid() {
        return output_valid;
    }

    std::pair< std::vector<Scnn::Input_Element>, std::vector<Scnn::Filter_Element> > Dispatcher::pop_data() {
        output_valid = false;
        return {latched_ia_vec, latched_w_vec};
    }


}

