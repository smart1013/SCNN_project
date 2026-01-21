
#include "tensor.h"
#include "loader.h"
#include "convlayer.h"
#include "pe.h"
#include "dispatcher.h"
#include "mult_array.h"
#include "buffer_queue.h"
#include "accumulator.h"
#include "common.h"
#include <iostream>
#include <cmath>
#include <iomanip>

// Golden Reference Convolution (Functional Model)
// Computes the expected result directly without cycle-accurate complexity
void compute_golden_output(Scnn::Tensor& IA, const std::vector<Scnn::Tensor*>& FW, Scnn::Tensor& Golden_OA) {
    int C = Scnn::LayerConfig::C;
    int H = Scnn::LayerConfig::H;
    int W = Scnn::LayerConfig::W;
    int K = Scnn::LayerConfig::K;
    int R = Scnn::LayerConfig::R;
    int S = Scnn::LayerConfig::S;
    
    int STRIDE = Scnn::LayerConfig::STRIDE;
    int PADDING = Scnn::LayerConfig::PADDING;
    int DILATION = Scnn::LayerConfig::DILATION;

    int out_h = Golden_OA.dims.h;
    int out_w = Golden_OA.dims.w;

    std::cout << "Calculating Golden Reference..." << std::endl;

    for (int k = 0; k < K; ++k) {
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                
                float sum = 0.0f;
                
                // Convolve
                // Input coordinate base
                int h_base = y * STRIDE - PADDING;
                int w_base = x * STRIDE - PADDING;

                for (int c = 0; c < C; ++c) {
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            
                            int h_in = h_base + r * DILATION;
                            int w_in = w_base + s * DILATION;

                            // Bounds check (Padding)
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                int ia_idx = c * (H * W) + h_in * W + w_in;
                                float ia_val = IA.data[ia_idx];

                                // Weight Layout: K filters, each is CxRxS tensor
                                int fw_idx = c * (R * S) + r * S + s;
                                float w_val = FW[k]->data[fw_idx];

                                sum += ia_val * w_val;
                            }
                        }
                    }
                }
                
                // Address in Golden Tensor
                int out_idx = k * (out_h * out_w) + y * out_w + x;
                Golden_OA.data[out_idx] = sum;
            }
        }
    }
}

int main() {
    std::cout << "=== Verification Start ===" << std::endl;

    // 1. Initialize Tensors (Same as top.cc)
    Scnn::TensorDims input_dims = {Scnn::LayerConfig::C, Scnn::LayerConfig::H, Scnn::LayerConfig::W};
    Scnn::Tensor IA(input_dims);
    // Random Init
    for (int i = 0; i < IA.data.size(); ++i) {
        IA.data[i] = (rand() % 100 < 30) ? (float)(rand() % 5 + 1) : 0.0f; // 30% density
    }

    Scnn::TensorDims filter_dims = {Scnn::LayerConfig::C, Scnn::LayerConfig::R, Scnn::LayerConfig::S};
    std::vector<Scnn::Tensor*> FW;
    for (int k = 0; k < Scnn::LayerConfig::K; ++k) {
        Scnn::Tensor* filter = new Scnn::Tensor(filter_dims);
        // Random Init
        for (int i = 0; i < filter->data.size(); ++i) {
            filter->data[i] = (rand() % 100 < 30) ? (float)(rand() % 5 + 1) : 0.0f; // 30% density
        }
        FW.push_back(filter);
    }

    // 2. Initialize Sim Output and Golden Output
    int out_h = (Scnn::LayerConfig::H + 2 * Scnn::LayerConfig::PADDING - Scnn::LayerConfig::DILATION * (Scnn::LayerConfig::R - 1) - 1) / Scnn::LayerConfig::STRIDE + 1;
    int out_w = (Scnn::LayerConfig::W + 2 * Scnn::LayerConfig::PADDING - Scnn::LayerConfig::DILATION * (Scnn::LayerConfig::S - 1) - 1) / Scnn::LayerConfig::STRIDE + 1;
    Scnn::TensorDims out_dims = {Scnn::LayerConfig::K, out_h, out_w};
    
    Scnn::Tensor Sim_OA(out_dims);
    Scnn::Tensor Golden_OA(out_dims);

    // 3. Run Golden Model
    compute_golden_output(IA, FW, Golden_OA);

    // 4. Run Cycle-Accurate Simulator (Copied logic from top.cc)
    std::cout << "Running Simulator..." << std::endl;
    {
         Scnn::Loader loader;
         
         // Loop over Filter Groups
         for (int k = 0; k < Scnn::LayerConfig::K; k += Scnn::HardwareConfig::FILTERS_PER_GROUP) {
             
             // Loop over Input Channels
             for (int c = 0; c < Scnn::LayerConfig::C; c++) {
                 
                 // Load Data to Buffers
                 loader.load_IA(IA, c);
                 loader.Load_FW(FW, k, k + Scnn::HardwareConfig::FILTERS_PER_GROUP, c);
                 
                 // Run PEs
                 for (int pe_num = 0; pe_num < Scnn::HardwareConfig::NUM_PE; pe_num++) {
                     Scnn::Input_Buffer* input_tile = loader.IA_buffers[pe_num];
                     Scnn::Weight_Buffer* weight_buf = &loader.weight_buffer;

                     Scnn::Dispatcher dispatcher;
                     Scnn::MultArray mult_array;
                     Scnn::BufferQueue buffer_queue;
                     Scnn::Accumulator accumulator;

                     dispatcher.set_buffers(input_tile, weight_buf);
                     
                     while (!dispatcher.finished || dispatcher.output_valid || mult_array.has_output() || !buffer_queue.is_empty()) {
                         accumulator.Cycle(&buffer_queue, &Sim_OA);
                         mult_array.Cycle(&dispatcher, &buffer_queue, &Sim_OA);
                         dispatcher.Cycle();
                     }
                 }
             }
         }
    }

    // 5. Compare Results
    std::cout << "Comparing Results..." << std::endl;
    int errors = 0;
    float epsilon = 1e-4; // Tolerance for floating point

    for (int i = 0; i < Sim_OA.data.size(); ++i) {
        float sim_val = Sim_OA.data[i];
        float gold_val = Golden_OA.data[i];

        if (std::abs(sim_val - gold_val) > epsilon) {
            if (errors < 10) { // Limit output
                auto [k, y, x] = Sim_OA.get_addr(i);
                std::cout << "Mismatch at (" << k << "," << y << "," << x << "): "
                          << "Sim=" << sim_val << " vs Gold=" << gold_val << std::endl;
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "VERIFICATION PASSED! All values match." << std::endl;
    } else {
        std::cout << "VERIFICATION FAILED! Total Mismatches: " << errors << std::endl;
    }

    // Cleanup
    for(auto t : FW) delete t;

    return 0;
}
