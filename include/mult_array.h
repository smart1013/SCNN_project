#ifndef SCNN_MULT_ARRAY_H_
#define SCNN_MULT_ARRAY_H_

#include <vector>
#include <deque>

namespace Scnn {

struct Element {
    bool valid;
    float value;
    int addr;
};

// struct filterWeight {
//     bool valid;
//     float value;
//     int addr;
// };

struct PartialSum {
    bool valid;
    float value;
    int addr;
};

class MultArray {

public:
    MultArray();
    ~MultArray();

    void reset();

    void cartesian_product(const std::vector<Element>& ia_vector, const std::vector<Element>& w_vector);
    
    std::vector<PartialSum> pop_outputs();

    bool has_output();

    void print_output_queue();

private:
    static const int OUTPUT_PORTS = 16;
    std::deque<PartialSum> output_queue;
};

}

#endif // SCNN_MULT_ARRAY_H_