#ifndef TENSOR_H_
#define TENSOR_H_

#include "common.h"

namespace Scnn {

struct TensorDims {
    int n;          // Batch size
    int c;          // Channel
    int h;          // Height
    int w;          // Width
};

class Tensor {
public:
    TensorDims dims;
    std::vector<float> data;

    Tensor() {
    }

    ~Tensor() {
        data.clear();
    }

    Tensor(const TensorDims& dims) {
        this->dims = dims;
        data.resize(dims.n * dims.c * dims.h * dims.w);
    }

    int get_index(int n, int c, int h, int w) {
        return ((n * dims.c + c) * dims.h + h) * dims.w + w;
    }

    float get_value(int n, int c, int h, int w) {
        int index = get_index(n, c, h, w);
        return data[index];
    }

    void set_value(int n, int c, int h, int w, float value) {
        int index = get_index(n, c, h, w);
        data[index] = value;
    }

    void set_random(float min_val, float max_val, float sparsity) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min_val, max_val);
        std::bernoulli_distribution is_zero(sparsity);
        
        for (auto& value : data) {
            if (is_zero(gen)) {
                value = 0.0;
            } else {
                value = dis(gen);
            }
        }
    }

};


}
#endif // TENSOR_H_