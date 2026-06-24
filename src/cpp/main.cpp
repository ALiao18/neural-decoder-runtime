#include <iostream>
#include <runtime/model_runner.hpp>
#include <runtime/latency_timer.hpp>

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: neural_decoder_runtime <path/to/model_config.json>\n";
        return 1;
    }

    try {
        // --- Load Model ---
        std::cout << "[ndr] Loading model from: " << argv[1] << std::endl;
        ndr::ModelRunner runner(argv[1]);

        const auto& cfg = runner.config();
        std::cout << "[ndr] Model loaded.\n"
                  << "      type:           " << cfg.model_type << "\n"
                  << "      input_channels: " << cfg.input_channels << "\n"
                  << "      context_bins:   " << cfg.context_bins << "\n"
                  << "      vocab_size:     " << cfg.vocab_size << "\n";
        
        // --- Benchmark setup ---
        const int N = 1000;
        auto input = torch::zeros({cfg.context_bins, cfg.input_channels});
        ndr::LatencyTimer timer;

        std::cout << "[ndr] Running " << N << " iterations...\n";
        
        for (int i = 0; i < N; ++i) {

            timer.start("total");
            
            timer.start("preprocessing");
            // stub: z-score normalization will be implemented here
            timer.stop("preprocessing");

            timer.start("inference");
            auto output = runner.forward(input);
            timer.stop("inference");

            timer.start("decoding");
            // stub: CTC beam search will be implemented here
            timer.stop("decoding");

            timer.start("lm");
            // stub: LM scoring will live here
            timer.stop("lm");

            timer.stop("total");
        }

        std::cout << "[ndr] Benchmark complete. Results:\n\n";
        timer.report(N);

    } catch (const std::exception& e) {
        std::cerr << "[ndr] ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}