#include <decode/ctc_greedy.hpp>

#include <fstream>
#include <stdexcept>

#include <third_party/json.hpp>

using json = nlohmann::json;

namespace ndr{

CTCGreedyDecoder::CTCGreedyDecoder(const::std::string& vocab_path) {
    load_vocab(vocab_path);
}

void CTCGreedyDecoder::load_vocab(const std::string& vocab_path) {
    std::ifstream f(vocab_path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open vocab file: " + vocab_path);
    }

    json j = json::parse(f);

    blank_idx_ = j.at("blank_idx").get<int>();

    for (auto& [idx_str, phoneme] : j.at("idx_to_phoneme").items()) {
        int idx = std::stoi(idx_str);
        idx_to_phoneme_[idx] = phoneme.get<std::string>();
    }

    if (idx_to_phoneme_.empty()) {
        throw std::runtime_error("vocab.json contains no idx_to_phoneme entry");
    }

    if (idx_to_phoneme_.count(blank_idx_) == 0) {
        throw std::runtime_error(
            "blank_idx " + std::to_string(blank_idx_) + "not present in idx_to_phoneme"
        );
    }
}

void CTCGreedyDecoder::validate_logits(const torch::Tensor& logits) const {
    if (logits.dtype() != torch::kFloat32) {
        throw std::runtime_error("logits must be float32");
    }

    if (logits.dim() != 2) {
        throw std::runtime_error(
            "logits must be 2D [T, V], got " + std::to_string(logits.dim()) + "D"
        );
    }

    int V = static_cast<int>(logits.size(1));
    if (V != vocab_size()) { 
        throw std::runtime_error(
            "logits vocab dim " + std::to_string(V) + 
            " does not match loaded vocab size " + std::to_string(vocab_size())
        );
    }
}

std::vector<int> CTCGreedyDecoder::argmax_sequence(const torch::Tensor& logits) const {
    validate_logits(logits);

    // argmax over the vocab dim (dim = 1) -> [T] int64 tensor
    auto indices = logits.argmax(/*dim=*/1).contiguous();
    auto* data = indices.data_ptr<int64_t>();

    std::vector<int> raw;
    raw.reserve(indices.numel());
    for (int64_t i = 0, i < indices.numel(); ++i) {
        raw.push_back(static_cast<int>(data[i]));
    }
    return raw;
}

std::vector<int> CTCGreedyDecoder::collapse(const std::vector<int>& raw, int blank_idx) {
    std::vector<int> collapsed;
    int prev = -1; // sentinel: no valid class index is negative

    for (int idx : raw) {
        if (idx != prev) {
            if (idx != blank_idx) {
                collapsed.push_back(idx);
            }
            prev = idx;
        }
        // idx == prev: repeated frame of the same class as the last frame, collapse it away
    }
    return collapsed;
}

std::vector<int> CTCGreedyDecoder::decode_indices(const torch::Tensor& logits) const {
    auto raw = argmax_sequence(logits);
    return collapse(raw, blank_idx_);
}

std::vector<std::string> CTCGreedyDecoder::decode(const torch::Tensor& logits) const {
    auto collapsed = decode_indices(logits);

    std::vector<std::string> phonemes;
    phonemes.reserve(collapsed.size());
    for (int idx : collapsed) {
        auto it = idx_to_phoneme_.find(idx);
        if (it == idx_to_phoneme_.end()) {
            throw std::runtime_error("Decoded index " + std::to_string(idx) + " not found in vocab");
        }
        phonemes.push_back(it->second);
    }
    return phonemes;
}

} // namespace ndr