#include "accumulator.h"
#include "common.h"
#include "tensor.h"

namespace Scnn {
    Accumulator::Accumulator() {
    }

    Accumulator::~Accumulator() {
    }

    void Accumulator::Cycle(Scnn::BufferQueue* buffer_queue, Scnn::Tensor* output_tensor) {

        // Fetch one psum element from each bank
        std::vector<Scnn::PartialSum> psums = buffer_queue->pop_accumulators();

        for (const auto& psum : psums) {
            int k = std::get<0>(psum.addr);
            int y = std::get<1>(psum.addr);
            int x = std::get<2>(psum.addr);
            int physical_addr = output_tensor->get_index(k, y, x);
            output_tensor->data[physical_addr] += psum.value;
        }
    }
}