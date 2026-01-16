#include "mult_array.h"
#include "common.h"
#include <iostream>

namespace Scnn {
    MultArray::MultArray() {
    }

    MultArray::~MultArray() {
    }

    void MultArray::reset() {
        output_queue.clear();
    }

    void MultArray::print_output_queue() {
        for (const auto& psum : output_queue) {
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

                int y_out = (y_in - y_weight * D + P) / U;
                int x_out = (x_in - x_weight * D + P) / U;

                if (y_out < 0 || x_out < 0) {
                    continue;
                }

                PartialSum psum;
                psum.value = ia.value * w.value;
                psum.addr = std::make_tuple(k_out, y_out, x_out);
                
                // std::cout << "Value: " << psum.value << ", Addr: " << "(" << k_out << ", " << y_out << ", " << x_out << ")" << std::endl;
                output_queue.push_back(psum);
            }
        }
    }

    bool MultArray::has_output() {
        return !output_queue.empty();
    }

    std::vector<PartialSum> MultArray::pop_outputs() {
        std::vector<PartialSum> outputs;
        for (int i = 0; i < Scnn::HardwareConfig::OUTPUT_PORT; i++) {
            if (output_queue.empty()) {
                break;
            }
            output_queue.pop_front();
        }
        return outputs;
    }

}
