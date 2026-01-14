#include "pe.h"
#include "common.h"

namespace Scnn {
    
    PE::PE() {
        ia_vector.resize(Scnn::HardwareConfig::IA_VECTOR_SIZE);
        w_vector.resize(Scnn::HardwareConfig::W_VECTOR_SIZE);
    }

    PE::~PE() {
        ia_vector.clear();
        w_vector.clear();
    }







}
