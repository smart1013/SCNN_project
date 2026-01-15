#include "pe.h"
#include "common.h"

namespace Scnn {
    
    PE::PE() {
        ia_vector.clear();
        w_vector.clear();
    }

    PE::~PE() {
        ia_vector.clear();
        w_vector.clear();
    }

    void PE::cartesian_product(Scnn::Input_Buffer* input_tile, Scnn::Weight_Buffer* weight_buffer) {
        int ia_size = input_tile->size;
        int w_size = weight_buffer->size;
        Scnn::MultArray mult_array;

        for (int i = 0; i < ia_size; i += Scnn::HardwareConfig::IA_VECTOR_SIZE) {
            ia_vector.clear();

            // fetch IA vector
            for (int k = 0; k < Scnn::HardwareConfig::IA_VECTOR_SIZE; ++k) {
                if (i + k < ia_size) {
                    ia_vector.push_back(input_tile->buffer[i + k]);
                }
            }

            // for each IA vector, iterate over all weight vectors
            for (int j = 0; j < w_size; j += Scnn::HardwareConfig::W_VECTOR_SIZE) {
                w_vector.clear();

                // fetch weight vector
                for (int k = 0; k < Scnn::HardwareConfig::W_VECTOR_SIZE; ++k) {
                    if (j + k < w_size) {
                        w_vector.push_back(weight_buffer->buffer[j + k]);
                    }
                }

                /******************************************************/

                // std::cout << "IA vector size:" << "\t" << ia_vector.size() << std::endl;
                // std::cout << "Weight vector size:" << "\t" << w_vector.size() << std::endl;
                
                mult_array.cartesian_product(ia_vector, w_vector);





                /******************************************************/
            }
        }
    }





}
