#include "model_runner.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

#include <third_party/json.hpp>

using json = nlohmann::json;

namespace ndr {

ModelRunner::ModelRunner(const std::string& config_path) {
    load_config(config_path);
    validate_config();
    load_module();
}

void ModelRunner::load_config(const std::string& config_path) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open config file " + config_path);
    }
    
    json j = json::parse(f);

    config_.model_type          = j.at("model_type").get<std::string>();
    config_.input_channels      = j.at("input_size").get<int>();
    config_.bin_size_ms         = j.at("bin_size_ms").get<int>();
    config_.context_bins        = j.at("context_bins").get<int>();
    config_.stride_bins         = j.at("stride_bins").get<int>();
    config_.vocab               = j.at("vocab").get<std::string>();
    config_.hidden_size         = j.at("hidden_size").get<int>();
    config_.num_layers          = j.at("num_layers").get<int>();
    config_.vocab_size          = j.at("vocab_size").get<int>();
    config_.dropout             = j.at("dropout").get<float>();
    config_.torchscript_path    = j.at("torchscript_path").get<std::string>();
}

void ModelRunner::validate_config() const {
    // invariant: stride bins <= context bins
    if (config_.stride_bins > config_.context_bins) {
        throw std::runtime_error(
            "stride bins (" + std::to_string(config_.stride_bins) + 
            ") must be <= context bins (" +
            std::to_string(config_.context_bins) + ")"    
        );
    }

    // MVP: only CTC supported 
    if (config_.model_type != "gru_ctc") {
        throw std::runtime_error(
            "currently only support gru_ctc, got " + config_.model_type
        );
    }

    // MVP: only arpabet supported
    if (config_.vocab != "arpabet") {
        throw std::runtime_error(
            "currently only support arpabet vocab, got " + config_.vocab
        );
    }
}

void ModelRunner::load_module() {
    try {
        module_ = torch::jit::load(config_.torchscript_path);
        module_.eval();
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            "Filed to load TorchScript model from '" + 
            config_.torchscript_path + "': " + e.what()
        );
    }
}

torch::Tensor ModelRunner::forward(const torch::Tensor& input) {
    // invariant, input must be float32
    if (input.dtype() != torch::kFloat32) {
        throw std::runtime_error(
            "input tensor must be float32, got " + std::string(input.dtype().name())
        );
    }

    // invariant: input channel dim must match config_.input_channels
    if (input.size(1) != config_.input_channels) {
        throw std::runtime_error(
            "input channel dim " + std::to_string(input.size(1)) + 
            " does not match config input_channels " + std::to_string(config_.input_channels)
        );
    }

    // add batch dim: [T, C] -> [1, T, C]
    auto x = input.unsqueeze(0); // adds a dim at position 0

    std::vector<torch::jit::IValue> inputs{x};
    auto output = module_.forward(inputs).toTensor();

    // Remove batch dim: [1, T, vocab_size] -> [T, vocab_size]
    return output.squeeze(0);

}

}