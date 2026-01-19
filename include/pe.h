#ifndef PE_H_
#define PE_H_

#include "common.h"
#include "tensor.h"
#include "loader.h"
#include "mult_array.h"


namespace Scnn {
    
class PE {

public:
    std::vector<Scnn::Input_Element> ia_vector;
    std::vector<Scnn::Filter_Element> w_vector;
    
    PE();
    ~PE();

    void cartesian_product(Scnn::Input_Buffer* input_tile, Scnn::Weight_Buffer* weight_buffer);
    std::tuple<int, int, int> cartesian_product(Scnn::Input_Buffer* input_tile, Scnn::Weight_Buffer* weight_buffer, Scnn::Tensor* output_tensor);
    
};

}



#endif // PE_H_