#include "scnn_cycle_model.h"
#include <iostream>

namespace Scnn {

// ==========================================
// AccumulatorUnit Implementation
// ==========================================
AccumulatorUnit::AccumulatorUnit(Scnn::Tensor* output_tensor, std::deque<PartialSum>* input_queue)
    : output_tensor(output_tensor), input_queue(input_queue) {}

void AccumulatorUnit::Cycle() {
    // Process up to 'OUTPUT_PORT' partial sums per cycle
    for (int i = 0; i < HardwareConfig::OUTPUT_PORT; ++i) {
        if (input_queue->empty()) {
            break;
        }

        PartialSum p = input_queue->front();
        input_queue->pop_front();

        // Write to Tensor ("Accumulate")
            int k_out = std::get<0>(p.addr);
            int y_out = std::get<1>(p.addr);
            int x_out = std::get<2>(p.addr);
            
            // Check bounds (optional if MultArray assumes correctness)
            if (k_out < output_tensor->dims.c && y_out < output_tensor->dims.h && x_out < output_tensor->dims.w) {
                 int phy_addr = output_tensor->get_index(k_out, y_out, x_out);
                 output_tensor->data[phy_addr] += p.value;
            }
    }
}

bool AccumulatorUnit::IsIdle() {
    return input_queue->empty();
}

// ==========================================
// MultiplierArrayUnit Implementation
// ==========================================
MultiplierArrayUnit::MultiplierArrayUnit() {}

void MultiplierArrayUnit::Cycle() {
    // 1. If we have input work to do
    if (!input_queue.empty()) {
        
        // We can only process Input if we have space in Output (Backpressure)
        // Note: For simplicity, assume infinite output queue or large enough
        // Ideally: if (output_queue.size() > threshold) return; 

        VectorPair pair = input_queue.front();
        input_queue.pop_front();

        // 2. Perform Processing (Functional)
        // This generates ALL partial sums for this vector pair. 
        // In real hardware, this might take 1 cycle to Latch, then logic delay.
        functional_unit.output_queue.clear();
        functional_unit.cartesian_product(pair.ia_vec, pair.w_vec);
        
        // 3. Move results to our output latch
        // Note: In strict modeling, we might limit how many we push per cycle?
        // But the multiplier array outputs them largely in parallel to the crossbar.
        for (const auto& p : functional_unit.output_queue) {
            output_queue.push_back(p);
        }
    }
}

bool MultiplierArrayUnit::IsIdle() {
    return input_queue.empty() && output_queue.empty();
}


// ==========================================
// DispatcherUnit Implementation
// ==========================================
DispatcherUnit::DispatcherUnit(Scnn::Input_Buffer* ia_buffer, Scnn::Weight_Buffer* w_buffer, MultiplierArrayUnit* next_stage)
    : ia_buffer(ia_buffer), w_buffer(w_buffer), next_stage(next_stage), i_idx(0), j_idx(0) {}

void DispatcherUnit::Cycle() {
    // Attempt to push work to Next Stage
    if (next_stage->IsInputFull()) {
        return; // Stall
    }
    
    // Check if we are done
    if (i_idx >= ia_buffer->size) {
        return;
    }

    // 1. Fetch IA Vector
    current_ia_vec.clear();
    for (int k = 0; k < HardwareConfig::IA_VECTOR_SIZE; ++k) {
        if (i_idx + k < ia_buffer->size) {
             current_ia_vec.push_back(ia_buffer->buffer[i_idx + k]);
        }
    }

    // 2. Fetch Weight Vector
    current_w_vec.clear();
    for (int k = 0; k < HardwareConfig::W_VECTOR_SIZE; ++k) {
        if (j_idx + k < w_buffer->size) {
             current_w_vec.push_back(w_buffer->buffer[j_idx + k]);
        }
    }

    // 3. Push to Next Stage
    VectorPair pair;
    pair.ia_vec = current_ia_vec;
    pair.w_vec = current_w_vec;
    next_stage->input_queue.push_back(pair);

    // 4. Update Indices
    j_idx += HardwareConfig::W_VECTOR_SIZE;
    if (j_idx >= w_buffer->size) {
        j_idx = 0; // Reset weight ptr
        i_idx += HardwareConfig::IA_VECTOR_SIZE; // Advance IA ptr
    }
}

bool DispatcherUnit::IsIdle() {
    // Idle only if processed everything
    return (i_idx >= ia_buffer->size);
}


// ==========================================
// CycleAccuratePE Implementation
// ==========================================
CycleAccuratePE::CycleAccuratePE() {
    global_cycles = 0;
}
CycleAccuratePE::~CycleAccuratePE() {}

unsigned long long CycleAccuratePE::Run(Scnn::Input_Buffer* input_tile, Scnn::Weight_Buffer* weight_buffer, Scnn::Tensor* output_tensor) {
    if (input_tile->size == 0) return 0;

    // Instantiate Components
    multiplier = std::make_unique<MultiplierArrayUnit>();
    dispatcher = std::make_unique<DispatcherUnit>(input_tile, weight_buffer, multiplier.get());
    accumulator = std::make_unique<AccumulatorUnit>(output_tensor, &multiplier->output_queue);
    
    global_cycles = 0;
    
    // Run Loop
    while (true) {
        global_cycles++;
        
        // Reverse Order Execution
        accumulator->Cycle();
        multiplier->Cycle();
        dispatcher->Cycle();
        
        // Exit Condition: All stages idle and queues empty
        if (dispatcher->IsIdle() && multiplier->IsIdle() && accumulator->IsIdle()) {
            break;
        }
        
        // Safety Break
        if (global_cycles > 10000000) {
            std::cout << "Cycle Limit Reached!" << std::endl;
            break;
        }
    }
    
    return global_cycles;
}

}
