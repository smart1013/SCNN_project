#include "common.h"
#include "tensor.h"


namespace Scnn {
    Tensor::Tensor() {
        this->non_zero_count = 0;
        this->size = 0;
        this->sparsity = 0.0;
    }
    
    Tensor::Tensor(const TensorDims& dims) {
        this->dims = dims;
        data.resize(dims.c * dims.h * dims.w);
        this->size = data.size();
        this->non_zero_count = 0;
        this->sparsity = 0.0;
    }

    Tensor::~Tensor() {
        data.clear();
    }

    int Tensor::get_size() {
        return data.size();
    }

    int Tensor::get_index(int c, int h, int w) {
        assert(c < dims.c && h < dims.h && w < dims.w);
        return (c * dims.h + h) * dims.w + w;
    }

    float Tensor::get_value(int c, int h, int w) {
        int index = get_index(c, h, w);
        assert(index <= data.size());
        return data[index];
    }

    void Tensor::set_value(int c, int h, int w, float value) {
        int index = get_index(c, h, w);
        data[index] = value;
    }

    std::tuple<int, int, int> Tensor::get_addr(int phy_addr) {
        int w = phy_addr % dims.w;
        int h = (phy_addr / dims.w) % dims.h;
        int c = (phy_addr / (dims.w * dims.h)) % dims.c;

        return std::make_tuple(c, h, w);
    }

    void Tensor::set_random(float min_val, float max_val, float sparsity, int seed) {
        // std::random_device rd;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(min_val, max_val);
        std::bernoulli_distribution is_zero(sparsity);
        
        this->non_zero_count = 0;
        for (auto& value : data) {
            if (is_zero(gen)) {
                value = 0.0;
            } else {
                value = dis(gen);
                this->non_zero_count++;
            }
        }
        this->sparsity = (float)this->non_zero_count / (float)this->size;
    }

    void Tensor::print() {
        int c = dims.c;
        int h = dims.h;
        int w = dims.w;
        int size = data.size();
        std::cout << "non_zero_count: " << non_zero_count << std::endl;
        std::cout << "size: " << size << std::endl;
        std::cout << "c: " << c << std::endl;
        std::cout << "h: " << h << std::endl;
        std::cout << "w: " << w << std::endl;
    }

    int Tensor::get_non_zero_count() {
        int count = 0;
        for (auto& value : data) {
            if (value != 0.0) {
                count++;
            }
        }
        return count;
    }

}
