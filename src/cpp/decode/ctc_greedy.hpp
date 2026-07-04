#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <torch/torch.h>

namespace ndr{

class CTCGreedyDecoder{
public:
    // Loads vocab.json (blank_idx + idx_to_phoneme map), throws if missing/malformed
    explicit CTCGreedyDecoder(const std::string& vocab_path);

    // Per-timestep argmax over the [T, V] logits.
    // Throws if logits is not 2D float32, or V doesn't match loaded vocab size
    std::vector<std::string> decode(const torch::Tensor& logits) const;

    // Collapses consecutive repeated indices, then strips blank_idx
    // Static + pure so it can be unit-tested without loaded vocab
    static std::vector<int> collapse(const std::vector<int>& raw, int blank_idx);

    // Full pipeline: logits -> argmax -> collapse -> phoneme strings
    std::vector<std::string> decode(const torch::Tensor& logits) const;

    // Same pipeline but returns collapsed indices, skipping string lookup
    // used by parity tests to compare
    std::vector<int> decode_indices(const torch::Tensor& logits) const;

    int blank_idx() const { return blank_idx_; }
    int vocab_size() const { return static_cast<int>(idx_to_phoneme_.size()); }

private:
    std::unordered_map<int, std::string> idx_to_phoneme_;
    int blank_idx_ = 0;

    void load_vocab(const std::string& vocab_path);
    void validate_logits(const torch::Tensor& logits) const;
};

} // namespace ndr