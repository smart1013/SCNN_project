#ifndef SCNN_CYCLE_MODEL_H_
#define SCNN_CYCLE_MODEL_H_

#include "common.h"
#include "tensor.h"
#include "loader.h"
#include "mult_array.h"
#include <deque>
#include <memory>

namespace Scnn {

// Abstract Base Class for a Pipeline Stage
class CycleStage {
public:
    virtual void Cycle() = 0;
    virtual bool IsIdle() = 0;
    virtual ~CycleStage() = default;
};

// 1. Accumulator Stage
// Reads PartialSums from input queue, adds to Output Tensor.
class AccumulatorUnit : public CycleStage {
public:
    AccumulatorUnit(Scnn::Tensor* output_tensor, std::deque<PartialSum>* input_queue);
    void Cycle() override;
    bool IsIdle() override;

private:
    Scnn::Tensor* output_tensor;
    std::deque<PartialSum>* input_queue; // Ptr to MultArray's output queue
    int banks;
    // Add logic for bank conflicts/latency here
};

// 2. Multiplier Array Stage
// Reads Vector Pairs from input queue, computes Cartesian Product, pushes PartialSums to output queue.
struct VectorPair {
    std::vector<Input_Element> ia_vec;
    std::vector<Filter_Element> w_vec;
};

class MultiplierArrayUnit : public CycleStage {
public:
    MultiplierArrayUnit();
    void Cycle() override;
    bool IsIdle() override;
    
    // Connectors
    std::deque<VectorPair> input_queue;  // From Dispatcher
    std::deque<PartialSum> output_queue; // To Accumulator
    
    // Helpers
    bool IsInputFull() const { return input_queue.size() >= input_queue_capacity; }
    
private:
    static const int input_queue_capacity = 2; // Double buffering
    static const int output_queue_capacity = 16; 
    
    // The functional unit that does the math (reused from your existing code)
    Scnn::MultArray functional_unit;
};

// 3. Dispatcher / Input Stage
// Iterates over IA and Weight buffers, forms Vector Pairs, pushes to Multiplier Array.
class DispatcherUnit : public CycleStage {
public:
    DispatcherUnit(Scnn::Input_Buffer* ia_buffer, Scnn::Weight_Buffer* w_buffer, MultiplierArrayUnit* next_stage);
    void Cycle() override;
    bool IsIdle() override;

private:
    Scnn::Input_Buffer* ia_buffer;
    Scnn::Weight_Buffer* w_buffer;
    MultiplierArrayUnit* next_stage;
    
    // Iteration State
    int i_idx; // IA index
    int j_idx; // Weight index
    
    std::vector<Input_Element> current_ia_vec;
    std::vector<Filter_Element> current_w_vec;
};


// 4. Top-Level Cycle-Accurate PE
class CycleAccuratePE {
public:
    CycleAccuratePE();
    ~CycleAccuratePE();
    
    // Run the full simulation for these buffers
    unsigned long long Run(Scnn::Input_Buffer* input_tile, Scnn::Weight_Buffer* weight_buffer, Scnn::Tensor* output_tensor);

private:
    std::unique_ptr<DispatcherUnit> dispatcher;
    std::unique_ptr<MultiplierArrayUnit> multiplier;
    std::unique_ptr<AccumulatorUnit> accumulator;
    
    unsigned long long global_cycles;
};

}

#endif // SCNN_CYCLE_MODEL_H_
