#ifndef LOADER_H_
#define LOADER_H_

#include "common.h"
#include "tensor.h"

namespace Scnn {


struct Input_Element {
    float value;
    // (C, H(y), W(x))
    std::tuple<int, int, int> addr;
};


struct Filter_Element {
    float value;
    // (K, C, R(y), S(x))
    std::tuple<int, int, int, int> addr;
};


class Input_Buffer {
public:
    std::vector<Input_Element> buffer;
    int size;

    Input_Buffer();
    ~Input_Buffer();

    void load_input_buffer(Scnn::Tensor& tensor);
    void print();
    void add_element(float value, std::tuple<int, int, int> addr);

private:
    std::string filename;
};


class Weight_Buffer {
public:
    std::vector<Filter_Element> buffer;
    int size;

    Weight_Buffer();
    ~Weight_Buffer();

    void add_element(float value, std::tuple<int, int, int, int> addr);
    void print();

private:
    std::string filename;

};


class Loader {
public:
    Loader();
    ~Loader();

    std::vector<Input_Buffer*> IA_buffers;
    int max_size;

    Scnn::Weight_Buffer weight_buffer;

    void load_IA(Scnn::Tensor& tensor);
    void load_IA(Scnn::Tensor& tensor, int target_channel);
    void Load_FW(std::vector<Scnn::Tensor*>& fw, int k_start, int k_end, int target_channel);

private:
    std::string filename;
};


}
#endif // LOADER_H_