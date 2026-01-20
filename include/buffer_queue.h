#ifndef SCNN_BUFFER_QUEUE_H_
#define SCNN_BUFFER_QUEUE_H_

#include "common.h"
#include "mult_array.h"
#include <vector>

namespace Scnn {

class BufferQueue {

public:
    std::deque<Scnn::PartialSum> queues[Scnn::HardwareConfig::NUM_BANKS];
    int stall_count;
    int queue_depth;
    int num_banks;

    BufferQueue();
    ~BufferQueue();

    void reset();

    int get_bank_id(int k, int y, int x);

    bool is_empty();

    void push_outputs(std::deque<Scnn::PartialSum>& batch);

    std::vector<Scnn::PartialSum> pop_accumulators();

};

}

#endif // SCNN_BUFFER_QUEUE_H_
