#ifndef SCNN_DISPATCHER_H_
#define SCNN_DISPATCHER_H_

#include "common.h"
#include "loader.h"

namespace Scnn {

class Dispatcher {
public:

    Scnn::Input_Buffer* input_buffer_ptr;
    Scnn::Weight_Buffer* weight_buffer_ptr;

    int curr_ia_idx;
    int curr_w_idx;
    bool finished;

    bool output_valid;
    std::vector<Scnn::Input_Element> latched_ia_vec;
    std::vector<Scnn::Filter_Element> latched_w_vec;
    
    Dispatcher();
    ~Dispatcher();

    void set_buffers(Scnn::Input_Buffer* ia_buff, Scnn::Weight_Buffer* w_buff);

    void Cycle();

    void reset();

    bool is_output_valid();

    std::pair< std::vector<Scnn::Input_Element>, std::vector<Scnn::Filter_Element> > pop_data();

};
}
#endif