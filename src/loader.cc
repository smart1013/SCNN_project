#include "common.h"
#include "tensor.h"
#include "loader.h"
#include <iostream>


namespace Scnn {
    
    Input_Buffer::Input_Buffer() {
    }
    
    Input_Buffer::~Input_Buffer() {
        buffer.clear();
    }

    void Input_Buffer::load_input_buffer(Scnn::Tensor& tensor) {

        buffer.resize(tensor.non_zero_count);
        this->size = tensor.non_zero_count;

        for (int i = 0; i < tensor.data.size(); i++) {
            std::tuple<int, int, int, int> addr = tensor.get_addr(i);
            int n, c, h, w;
            std::tie(n, c, h, w) = addr;
            float value = tensor.get_value(n, c, h, w);
            if (value == 0.0) {
                continue;
            }
            Element element;
            element.value = value;
            element.addr = addr;
            buffer.push_back(element);
        }
    }
    

}
