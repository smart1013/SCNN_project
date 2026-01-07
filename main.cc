#include <iostream>
#include "mult_array.h"
#include <vector>

Scnn::Element make_element(float value, int addr) {
    Scnn::Element element;
    element.valid = true;
    element.value = value;
    element.addr = addr;
    return element;
}

int main() {
    std::vector<Scnn::Element> ia_vector;
    std::vector<Scnn::Element> w_vector;

    ia_vector.push_back(make_element(1.0, 0));
    ia_vector.push_back(make_element(2.0, 1));
    ia_vector.push_back(make_element(3.0, 2));

    w_vector.push_back(make_element(4.0, 0));
    w_vector.push_back(make_element(5.0, 1));
    w_vector.push_back(make_element(6.0, 2));
    
    Scnn::MultArray mult_array;
    mult_array.cartesian_product(ia_vector, w_vector);

    mult_array.print_output_queue();

    mult_array.reset();

    return 0;
}
