#ifndef SCNN_ACCUMULATOR_H
#define SCNN_ACCUMULATOR_H

#include "common.h"
#include "buffer_queue.h"
#include "tensor.h"
#include "mult_array.h"

namespace Scnn {

class Accumulator {
public:
    Accumulator();
    ~Accumulator();

    void Cycle(Scnn::BufferQueue* buffer_queue, Scnn::Tensor* output_tensor);

};

}

#endif // SCNN_ACCUMULATOR_H
