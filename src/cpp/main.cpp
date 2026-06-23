#include <iostream>
#include <runtime/model_runner.hpp>

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: neural_decoderruntime <path/to/model_config.json>\n";
        return 1;
    }

    try {
        std::cout << "[ndr] Loading model from: " << argv[1] << std::endl;
        ndr::ModelRunner runner(argv[1]);

        const auto& cfg = runner.config();
        std::cout << "[ndr] Model loaded.\n"
                  << "      type:           " << cfg.model_type << "\n"
                  << "      input_channels: " << cfg.input_channels << "\n"
                  << "      context_bins:   " << cfg.context_bins << "\n"
                  << "      vocab_size:     " << cfg.vocab_size << "\n";
    
        // Dummy input: [context_bins, input_channels], all zeros
        auto dummy = torch::zeros({cfg.context_bins, cfg.input_channels});
        std::cout << "[ndr] Running forward pass on dummy input "
                  << dummy.sizes() << "...\n";
        
        auto output = runner.forward(dummy);
        std::cout << "[ndr] Output shape: " << output.sizes() << "\n";
        std::cout << "[ndr] Output dtype: " << output.dtype() << "\n";
        std::cout << "[ndr] Day 1 complete. \n";

    } catch (const std::exception& e) {
        std::cerr << "[ndr] ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}