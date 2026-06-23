#pragma once

#include <string>
#include <torch/script.h>

namespace ndr {

struct ModelConfig {
    std::string model_type;      // "gru_ctc", etc
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
    explicit ModelRunner(const std::string& config_path);

    // forward pass, imnput must be float32, shape [batch T, input_channels]
    // returns a tensor shape [batch T, vocab_size] 
    torch::Tensor forward(const torch::Tensor& input);

    const ModelConfig& config() const { return config_; }
;

private:
    ModelConfig config_;
    torch::jit::script::Module module_; // cpp representation of model.pt

    void load_config(const std::string& config_path);
    void validate_config() const; // read only object
    void load_module();
};

} // namespace ndr

/*
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
*/