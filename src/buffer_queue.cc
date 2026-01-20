#include "buffer_queue.h"

namespace Scnn {

    BufferQueue::BufferQueue() {
        reset();
    }

    BufferQueue::~BufferQueue() {
    }

    void BufferQueue::reset() {
        for (int i = 0; i < Scnn::HardwareConfig::NUM_BANKS; ++i) {
            queues[i].clear();
        }
        stall_count = 0;
        queue_depth = Scnn::HardwareConfig::BUFFER_QUEUE_DEPTH;
        num_banks = Scnn::HardwareConfig::NUM_BANKS;
    }

    int BufferQueue::get_bank_id(int k, int y, int x) {
        int flat = k + y + x;
        if (flat < 0) {
            flat = -flat;
        }
        return flat % num_banks;
    }

    bool BufferQueue::is_empty() {
        for (int i = 0; i < num_banks; ++i) {
            if (!queues[i].empty()) {
                return false;
            }
        }
        return true;
    }

    void BufferQueue::push_outputs(std::deque<Scnn::PartialSum>& batch) {
        
        auto it = batch.begin();
        while (it != batch.end()) {
            const auto& psum = *it;

            int bank_id = get_bank_id(std::get<0>(psum.addr), std::get<1>(psum.addr), std::get<2>(psum.addr));

            // Move the element into the queue if the queue is not full
            if (queues[bank_id].size() < queue_depth) {
                queues[bank_id].push_back(psum);

                // remove from the input batch
                it = batch.erase(it);
            } else {
                // remain in the batch
                stall_count++;
                ++it;
            }
        }
    }

    std::vector<Scnn::PartialSum> BufferQueue::pop_accumulators() {
        // pop the front item from each bank
        std::vector<Scnn::PartialSum> out_batch;

        for (int i = 0; i < num_banks; ++i) {
            if (!queues[i].empty()) {
                out_batch.push_back(queues[i].front());
                queues[i].pop_front();
            }
        }
        return out_batch;
    }
    

}