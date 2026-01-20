#ifndef SCNN_MULT_ARRAY_H_
#define SCNN_MULT_ARRAY_H_

#include "common.h"
#include "tensor.h"
#include "loader.h"
#include "dispatcher.h"
#include <deque>

namespace Scnn {
class Dispatcher; // Forward declaration
class BufferQueue; // Forward declaration

struct PartialSum {
    float value;
    std::tuple<int, int, int> addr;
};

class MultArray {

public:
    MultArray();
    ~MultArray();

    void reset();

    void cartesian_product(const std::vector<Input_Element>& ia_vector, const std::vector<Filter_Element>& w_vector);

    int cartesian_product(const std::vector<Input_Element>& ia_vector, const std::vector<Filter_Element>& w_vector, Scnn::Tensor* output_tensor);
    
    std::vector<PartialSum> pop_outputs();

    bool has_output();

    void Cycle(Scnn::Dispatcher* dispatcher, Scnn::BufferQueue* buffer_queue, Scnn::Tensor* output_tensor);

    void print_output_queue();

    std::deque<PartialSum> int_latch;
    int total_mults_count;
    int idle_count;
    int idle_cycle;
};

}

#endif // SCNN_MULT_ARRAY_H_