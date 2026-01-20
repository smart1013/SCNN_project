#include "tensor.h"
#include "loader.h"
#include "convlayer.h"
#include "pe.h"
#include "dispatcher.h"
#include "mult_array.h"
#include "buffer_queue.h"
#include "common.h"

int main() {
    std::deque<Scnn::PartialSum> batch;
    Scnn::BufferQueue buffer_queue;

    Scnn::PartialSum psum;
    psum.value = 1.0;
    psum.addr = std::make_tuple(1, 1, 1);
    batch.push_back(psum);

    psum.value = 2.0;
    psum.addr = std::make_tuple(1, 1, 1);
    batch.push_back(psum);

    psum.value = 3.0;
    psum.addr = std::make_tuple(1, 1, 1);
    batch.push_back(psum);

    psum.value = 4.0;
    psum.addr = std::make_tuple(1, 1, 1);
    batch.push_back(psum);

    psum.value = 5.0;
    psum.addr = std::make_tuple(2, 1, 1);
    batch.push_back(psum);

    psum.value = 6.0;
    psum.addr = std::make_tuple(2, 1, 1);
    batch.push_back(psum);

    psum.value = 7.0;
    psum.addr = std::make_tuple(2, 1, 1);
    batch.push_back(psum);

    psum.value = 8.0;
    psum.addr = std::make_tuple(2, 1, 1);
    batch.push_back(psum);

    buffer_queue.push_outputs(batch);

    for (auto& element : batch) {
        std::cout << element.value << " " << std::endl;
    }

    int hash1 = buffer_queue.get_bank_id(1, 1, 1);
    int hash2 = buffer_queue.get_bank_id(3, 10, 1);

    std::cout << "hash1: " << hash1 << std::endl;
    std::cout << "hash2: " << hash2 << std::endl;

}
