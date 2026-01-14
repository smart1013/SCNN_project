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
    std::vector<Scnn::Weight_Element> w_vector;
    
    PE();
    ~PE();

}
}



#endif // PE_H_




