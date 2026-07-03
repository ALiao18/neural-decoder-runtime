#include <iostream>
#include <fstream>
#include <iomanip>
#include <torch/torch.h>
#include "runtime/model_runner.hpp"
#include "runtime/latency_timer.hpp"
#include "runtime/zscore_constants.hpp"
#include "third_party/json.hpp"

using json = nlohmann::json;

static void run_parity(
    const ndr::ModelRunner& runner,
    ndr::ZScoreConstants& zscore,
    const std::string& fixture_path)
{
    std::ifstream f(fixture_path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open fixture file: " + fixture_path);
    }
    json j = json::parse(f);
    std::string block_id = j.at("block_id").get<std::string>();
    std::vector<float> input_vec = j.at("input").get<std::vector<float>>();

    auto input = torch::tensor(input_vec).reshape({25, 256});

    zscore.set_block(block_id);
    auto normed = zscore.normalize(input);
    auto output = runner.forward(normed);  // [25, 40]

    auto flat = output.reshape({-1}).contiguous();
    auto* data = flat.data_ptr<float>();
    std::cout << "[";
    for (int i = 0; i < flat.numel(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << std::setprecision(8) << data[i];
    }
    std::cout << "]\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  benchmark: neural_decoder_runtime <config>\n"
                  << "  parity:    neural_decoder_runtime <config> --parity <fixture>\n";
        return 1;
    }

    try {
        std::cerr << "[ndr] Loading model from: " << argv[1] << "\n";
        ndr::ModelRunner runner(argv[1]);

        const auto& cfg = runner.config();
        std::cerr << "[ndr] Model loaded.\n"
                  << "      type:           " << cfg.model_type     << "\n"
                  << "      input_channels: " << cfg.input_channels << "\n"
                  << "      context_bins:   " << cfg.context_bins   << "\n"
                  << "      vocab_size:     " << cfg.vocab_size     << "\n";

        ndr::ZScoreConstants zscore("artifacts/constants.json");

        // ── Parity mode ───────────────────────────────────────────────────
        if (argc == 4 && std::string(argv[2]) == "--parity") {
            run_parity(runner, zscore, argv[3]);
            return 0;
        }

        // ── Benchmark mode ────────────────────────────────────────────────
        zscore.set_block("1");

        const int N = 1000;
        auto input = torch::zeros({cfg.context_bins, cfg.input_channels});
        ndr::LatencyTimer timer;

        std::cout << "[ndr] Running " << N << " iterations...\n";

        for (int i = 0; i < N; ++i) {
            timer.start("total");

            timer.start("preprocessing");
            auto normed = zscore.normalize(input);
            timer.stop("preprocessing");

            timer.start("inference");
            auto output = runner.forward(normed);
            timer.stop("inference");

            timer.start("decoding");
            // stub: CTC beam search will live here
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