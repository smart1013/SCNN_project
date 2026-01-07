#include "mult_array.h"
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
            std::cout << "Value: " << psum.value << ", Addr: " << psum.addr << std::endl;
        }
    }

    void MultArray::cartesian_product(const std::vector<Element>& ia_vector, const std::vector<Element>& w_vector) {
        for (const auto& ia : ia_vector) {
            if (!ia.valid) {
                continue;
            }
            for (const auto& w : w_vector) {
                if (!w.valid) {
                    continue;
                }
                
                PartialSum psum;
                psum.valid = true;
                psum.value = ia.value * w.value;
                psum.addr = ia.addr + w.addr;

                output_queue.push_back(psum);
            }
        }
    }

    bool MultArray::has_output() {
        return !output_queue.empty();
    }

    std::vector<PartialSum> MultArray::pop_outputs() {
        std::vector<PartialSum> outputs;
        for (int i = 0; i < OUTPUT_PORTS; i++) {
            if (output_queue.empty()) {
                break;
            }
            output_queue.pop_front();
        }
        return outputs;
    }

}
