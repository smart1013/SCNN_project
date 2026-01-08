#include "common.h"
#include "tensor.h"


namespace Scnn {
    Tensor::Tensor() {
    }
    
    Tensor::Tensor(const TensorDims& dims) {
        this->dims = dims;
        data.resize(dims.n * dims.c * dims.h * dims.w);
    }

    Tensor::~Tensor() {
        data.clear();
    }

    Tensor::get_index(int n, int c, int h, int w) {
        assert(n < dims.n && c < dims.c && h < dims.h && w < dims.w);
        return ((n * dims.c + c) * dims.h + h) * dims.w + w;
    }

    Tensor::get_value(int n, int c, int h, int w) {
        int index = get_index(n, c, h, w);
        assert(index <= data.size());
        return data[index];
    }

    Tensor::set_value(int n, int c, int h, int w, float value) {
        int index = get_index(n, c, h, w);
        data[index] = value;
    }

    Tensor::get_addr(int phy_addr) {
        int w = phy_addr % dims.w;
        int h = (phy_addr / dims.w) % dims.h;
        int c = (phy_addr / (dims.w * dims.h)) % dims.c;
        int n = (phy_addr / (dims.w * dims.h * dims.c));

        return std::make_tuple(n, c, h, w);
    }

    Tensor::set_random(float min_val, float max_val, float sparsity) {
        // std::random_device rd;
        std::mt19937 gen(0);
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

    Tensor::print() {
        int n = dims.n;
        int c = dims.c;
        int h = dims.h;
        int w = dims.w;
        int size = data.size();
        std::cout << "size: " << size << std::endl;
        std::cout << "n: " << n << std::endl;
        std::cout << "c: " << c << std::endl;
        std::cout << "h: " << h << std::endl;
        std::cout << "w: " << w << std::endl;
    }


}
