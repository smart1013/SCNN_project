#ifndef LOADER_H_
#define LOADER_H_

#include "common.h"

namespace Scnn {

struct Element {
    float value;
    std::tuple<int, int, int, int> addr;
};


class Input_Buffer {
public:
    std::vector<Element> buffer;
    int size;

    Input_Buffer();
    ~Input_Buffer();

    void load_input_buffer(Scnn::Tensor& tensor);

private:
    std::string filename;
};



class Loader {
public:
    Loader();
    ~Loader();

    std::vector<Input_Buffer>* load_IA(Scnn::Tensor& tensor);
    void load_W(Scnn::Tensor& tensor);

private:
    std::string filename;
};

}
#endif // LOADER_H_