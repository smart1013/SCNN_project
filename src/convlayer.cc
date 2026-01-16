#include "common.h"
#include "tensor.h"
#include "loader.h"
#include "convlayer.h"

namespace Scnn {

    ConvLayer::ConvLayer() {  
    }

    ConvLayer::~ConvLayer() {  
        for (auto filter : FW) {
            delete filter;
        }
    }

    void ConvLayer::initialize() {
        Scnn::TensorDims input_dims, filter_dims, output_dims;
        FW.resize(Scnn::LayerConfig::K);

        input_dims.c = Scnn::LayerConfig::C;
        input_dims.h = Scnn::LayerConfig::H;
        input_dims.w = Scnn::LayerConfig::W;

        filter_dims.c = Scnn::LayerConfig::C;
        filter_dims.h = Scnn::LayerConfig::S;
        filter_dims.w = Scnn::LayerConfig::R;

        output_dims.c = Scnn::LayerConfig::K;
        output_dims.h = ((input_dims.h + Scnn::LayerConfig::PADDING * 2 - Scnn::LayerConfig::DILATION * (filter_dims.h - 1) - 1) / Scnn::LayerConfig::STRIDE) + 1;
        output_dims.w = ((input_dims.w + Scnn::LayerConfig::PADDING * 2 - Scnn::LayerConfig::DILATION * (filter_dims.w - 1) - 1) / Scnn::LayerConfig::STRIDE) + 1;

        IA = Scnn::Tensor(input_dims);
        IA.set_random(Scnn::LayerConfig::IA_MIN_VAL, Scnn::LayerConfig::IA_MAX_VAL, Scnn::LayerConfig::IA_SPARSITY, 42);

        for (int i = 0; i < Scnn::LayerConfig::K; i++) {
            Scnn::Tensor* filter = new Scnn::Tensor(filter_dims);
            filter->set_random(Scnn::LayerConfig::FW_MIN_VAL, Scnn::LayerConfig::FW_MAX_VAL, Scnn::LayerConfig::FW_SPARSITY, i);
            FW[i] = filter;
        }

        OA = Scnn::Tensor(output_dims);
    }
    
    
    
}