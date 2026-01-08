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

    Tensor();

    ~Tensor();

    Tensor(const TensorDims& dims);

    int get_size();

    int get_index(int n, int c, int h, int w);

    float get_value(int n, int c, int h, int w);

    void set_value(int n, int c, int h, int w, float value);

    std::tuple<int, int, int, int> get_addr(int phy_addr);

    void set_random(float min_val, float max_val, float sparsity);

    void print();
};

}

#endif // TENSOR_H_