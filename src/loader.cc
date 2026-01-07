#include "common.h"
#include "tensor.h"
#include "loader.h"
#include <iostream>


namespace Scnn {
    
    Input_Buffer::Input_Buffer(int size) {
        buffer.resize(size);
    }
    
    Input_Buffer::~Input_Buffer() {
        buffer.clear();
    }

    void Input_Buffer::load_input_buffer(Scnn::Tensor& tensor) {
        for (auto& value : tensor.data) {
            if (value == 0.0) {
                continue;
            }
            Element element;
            element.value = value;
            
        }
    }
    

}
