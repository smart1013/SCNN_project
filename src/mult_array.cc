#include "mult_array.h"
#include "dispatcher.h"
#include "buffer_queue.h"
#include "common.h"
#include <iostream>

namespace Scnn {
    MultArray::MultArray() {
        reset();
    }

    MultArray::~MultArray() {
    }

    void MultArray::reset() {
        int_latch.clear();
        idle_count = 0;
        total_mults_count = 0;
        idle_cycle = 0;
    }

    void MultArray::print_output_queue() {
        for (const auto& psum : int_latch) {
            int k_out = std::get<0>(psum.addr);
            int y_out = std::get<1>(psum.addr);
            int x_out = std::get<2>(psum.addr);
            std::cout << "Value: " << psum.value << ", Addr: " << "(" << k_out << ", " << y_out << ", " << x_out << ")" << std::endl;
        }
    }

    void MultArray::cartesian_product(const std::vector<Input_Element>& ia_vector, const std::vector<Filter_Element>& w_vector) {
        for (const auto& ia : ia_vector) {
            for (const auto& w : w_vector) {

                int c_in = std::get<0>(ia.addr);
                int y_in = std::get<1>(ia.addr);
                int x_in = std::get<2>(ia.addr);

                int k_out = std::get<0>(w.addr);
                int c_weight = std::get<1>(w.addr);
                int y_weight = std::get<2>(w.addr);
                int x_weight = std::get<3>(w.addr);

                assert(c_in == c_weight);

                int U = Scnn::LayerConfig::STRIDE;
                int D = Scnn::LayerConfig::DILATION;
                int P = Scnn::LayerConfig::PADDING;

                int y_numerator = (y_in - y_weight * D + P);
                int x_numerator = (x_in - x_weight * D + P);

                if (y_numerator % U != 0 || x_numerator % U != 0) {
                    continue;
                }

                int y_out = y_numerator / U;
                int x_out = x_numerator / U;

                PartialSum psum;
                psum.value = ia.value * w.value;
                psum.addr = std::make_tuple(k_out, y_out, x_out);
                
                int_latch.push_back(psum);
            }
        }
    }


    int MultArray::cartesian_product(const std::vector<Input_Element>& ia_vector, const std::vector<Filter_Element>& w_vector, Scnn::Tensor* output_tensor) {

        int_latch.clear();
        int idle_count = 0;

        for (const auto& ia : ia_vector) {
            for (const auto& w : w_vector) {

                int c_in = std::get<0>(ia.addr);
                int y_in = std::get<1>(ia.addr);
                int x_in = std::get<2>(ia.addr);

                int k_out = std::get<0>(w.addr);
                int c_weight = std::get<1>(w.addr);
                int y_weight = std::get<2>(w.addr);
                int x_weight = std::get<3>(w.addr);

                assert(c_in == c_weight);

                int U = Scnn::LayerConfig::STRIDE;
                int D = Scnn::LayerConfig::DILATION;
                int P = Scnn::LayerConfig::PADDING;

                int y_numerator = (y_in - y_weight * D + P);
                int x_numerator = (x_in - x_weight * D + P);

                if (y_numerator % U != 0 || x_numerator % U != 0) {
                   idle_count++;
                   continue;
                }

                int y_out = y_numerator / U;
                int x_out = x_numerator / U;

                int y_out_limit = output_tensor->dims.h;
                int x_out_limit = output_tensor->dims.w;

                if (y_out < 0 || x_out < 0 || y_out >= y_out_limit || x_out >= x_out_limit) {
                    idle_count++;
                    continue;
                }

                PartialSum psum;
                psum.value = ia.value * w.value;
                psum.addr = std::make_tuple(k_out, y_out, x_out);
                
                // std::cout << "Value: " << psum.value << ", Addr: " << "(" << k_out << ", " << y_out << ", " << x_out << ")" << std::endl;
                int_latch.push_back(psum);
            }
        }

        return idle_count;
    }



    bool MultArray::has_output() {
        return !int_latch.empty();
    }

    std::vector<PartialSum> MultArray::pop_outputs() {
        std::vector<PartialSum> outputs;
        for (int i = 0; i < Scnn::HardwareConfig::OUTPUT_PORT; i++) {
            if (int_latch.empty()) {
                break;
            }
            int_latch.pop_front();
        }
        return outputs;
    }





    void MultArray::Cycle(Scnn::Dispatcher* dispatcher, Scnn::BufferQueue* buffer_queue, Scnn::Tensor* output_tensor) {
        
        // check if there are outputs to be dispatched
        if (!int_latch.empty()) {
            buffer_queue->push_outputs(int_latch);

            if (!int_latch.empty()) {
                // stall if there are still outputs to be pushed
                idle_cycle++;
                return;
            }
        }
        
        // stall if there are no inputs to be processed
        if (!dispatcher->is_output_valid()) {
            idle_cycle++;
            return;
        } 

        auto [ia_vector, w_vector] = dispatcher->pop_data();
        idle_count += cartesian_product(ia_vector, w_vector, output_tensor);
        total_mults_count += Scnn::HardwareConfig::NUM_MULTIPLIERS;

        if (!int_latch.empty()) {
            buffer_queue->push_outputs(int_latch);
        }

    }

}
