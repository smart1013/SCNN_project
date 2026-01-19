# SCNN Cycle-Level Simulation Refactoring Plan

## 1. Core Concept: Reverse Dataflow
To achieve cycle accuracy, we must abandon the nested `for` loops that run to completion. Instead, we represent hardware units as objects with internal state (registers) that advance one step per `Cycle()` call. We execute these calls in **reverse order** (End -> Start) to simulate proper pipeline register behavior.

## 2. New Component Architecture

We will break the current `PE` responsibilities into distinct sub-modules, each with a `Cycle()` method.

### A. Dispatcher (Data FIFO)
*   **Responsibility:** Replaces the outer loops (`i` for Inputs, `j` for Weights).
*   **State:**
    *   `current_ia_idx`: Index of current input active input vector.
    *   `current_w_idx`: Index of current active weight vector.
    *   `output_buffer`: A latch holding the fetched vectors (IA packet + Weight packet).
*   **Cycle() Logic:**
    1. Check if the **Multiplier Array** is ready to accept new data.
    2. If ready, fetch the next `IA_vector` and `W_vector` from the buffers.
    3. Push these vectors to the pipeline register between Dispatcher and Multiplier.
    4. Increment indices.

### B. Multiplier Array
*   **Responsibility:** Performs the Cartesian Product.
*   **State:**
    *   `input_latch`: Holds one `IA_vector` + `W_vector` pair received from Dispatcher.
    *   `internal_output_queue`: Holds the generated `PartialSum`s.
*   **Cycle() Logic:**
    1. If `internal_output_queue` is not full, read from `input_latch`.
    2. Perform computations (generates 16 partial sums for 4x4 input).
    3. Push results into `internal_output_queue`.
    4. **Wait**: It might take multiple cycles to drain 16 partial sums if the bandwidth to the next stage is limited (e.g., Crossbar bandwidth).

### C. Accumulator (Back-end)
*   **Responsibility:** Reads partial sums and updates the `Output Tensor`.
*   **State:**
    *   `input_latch`: Holds `PartialSum`s arriving from the Multiplier/Crossbar.
*   **Cycle() Logic:**
    1. Read from `input_latch`.
    2. Perform the atomic add to the global `output_tensor`.
    3. Mark the latch as "empty" (ready for new data next cycle).

## 3. The PE `Cycle()` Orchestrator
The `PE` class becomes the top-level container that orchestrates the clock.

```cpp
void PE::RunSimulation() {
    int cycles = 0;
    while (!is_finished()) {
        // Reverse Dataflow Execution
        accumulator.Cycle();       // 3. Drain output
        multiplier_array.Cycle();  // 2. Process data
        dispatcher.Cycle();        // 1. Fetch new data
        
        cycles++;
    }
}
```

## 4. Proposed Steps for Refactoring

1.  **Define Interfaces:** Create a simple `Module` interface or standard structure for `Cycle()`, `IsBusy()`, `Push()`, `Pop()`.
2.  **Refactor `PE.h`**: Split the single monolithic class into `Dispatcher`, `Multiplier`, and `Accumulator` member objects.
3.  **Implement `Dispatcher`**: Move the looping logic (lines 63-82 of `pe.cc`) into this class's state machine.
4.  **Implement `Multiplier`**: Modify `MultArray` to compute one batch and then stall until it can offload all results.
5.  **Implement `Accumulator`**: Move the write-back logic (lines 93-99 of `pe.cc`) here.
6.  **Integrate**: Write the `PE::Cycle()` loop as shown in the slide.

## 5. Key Questions for You
*   **Bandwidth**: How many partial sums can the text `Multiplier` send to the `Accumulator` per cycle? (e.g., 1, 4, or infinite?)
*   **Latency**: Do you want to model a specific latency (e.g., Multiplication takes 2 cycles)? Or just 1 cycle per stage?
