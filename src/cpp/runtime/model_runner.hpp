#pragma once

#include <string>
#include <torch/script.h>

namespace ndr {

struct ModelConfig {
    std::string model_path;      // "gru_ctc", etc
    int         input_channels;  // 256
    int         bin_size_ms;     // 20
    int         context_bins;    // 25
    int         stride_bins;     // 5
    std::string vocab;           // "arpabet" 
    int         hidden_size;     // 1024
    int         num_layers;      // 3
    int         vocab_size;      // 40
    float       dropout;         // 0.1
    std::string torchscript_path;// "model.pt"
};

class ModelRunner {
public:
    // loads model_config.json, validates fields, loads model.pt
    // Throws std::runtime_error on any violations
    explicit ModelRunner()
};

}

config = {
    'model_type': 'gru_ctc',
    'input_size': INPUT_SIZE,
    'bin_size_ms': 20,
    'context_bins': 25,
    'stride_bins': 5,
    'vocab': "arpabet",
    'hidden_size': HIDDEN_SIZE,
    'num_layers': NUM_LAYERS,
    'vocab_size': VOCAB_SIZE,
    'dropout': DROPOUT,
    'torchscript_path': TORCHSCRIPT_PATH
}