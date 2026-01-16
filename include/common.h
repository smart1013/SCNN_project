#ifndef SCNN_COMMON_H_
#define SCNN_COMMON_H_

#include <iomanip>
#include <deque>
#include <vector>
#include <tuple>
#include <string>
#include <set>
#include <map>
#include <random>   
#include <assert.h>
#include <tuple>
#include <fstream>
#include <iostream>


namespace Scnn {
    using namespace std;

// Hardware Configuration
struct HardwareConfig{
    static const int NUM_PE = 64;
    static const int NUM_MULTIPLIERS = 16;
    static const int OUTPUT_PORT = 16;
    static const int IA_VECTOR_SIZE = 4;
    static const int W_VECTOR_SIZE = 4;
    static const int OUTPUT_PORTS = 16;
};

// Layer (CNN) Configuration
struct LayerConfig {
    static const int C = 3;
    static const int H = 224;
    static const int W = 224;
    static const int S = 11;
    static const int R = 11;
    static const int K = 8;

    static const int STRIDE = 3;
    static const int DILATION = 1;
    static const int PADDING = 1;

    static constexpr float IA_MIN_VAL = 0.0;
    static constexpr float IA_MAX_VAL = 1.0;
    static constexpr float IA_SPARSITY = 0.5;

    static constexpr float FW_MIN_VAL = 0.0;
    static constexpr float FW_MAX_VAL = 1.0;
    static constexpr float FW_SPARSITY = 0.5;
};

}

#endif