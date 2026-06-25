#pragma once

#include <string>
#include <unordered_map>
#include <torch/torch.h>

namespace ndr {

struct BlockStats {
    torch::Tensor means; // [C] float32
    torch::Tensor stds;  // [C] float32
};

class ZScoreConstants {
public:
    // Loads constants.json. Throws if file is missing or malformed.
    explicit ZScoreConstants(const std::string& constants_path);

    // pre-selects a block for the session, called before normalize()
    // throws if block_id not found in constants
    void set_block(const std::string& block_id);
    

    // applies z-scoring channel-wise to input [T,C]
    // returns normalized [T,C] float32 tensor
    // throws if set_block() has not been called, or input shape/dtype is wrong
    torch::Tensor normalize(const torch::Tensor& input) const;

    // returns true if block_id exists in loaded constants
    bool has_block(const std::string& block_id) const;

    int num_blocks()   const { return static_cast<int>(blocks_.size()); }
    int num_channels() const { return num_channels_; }

private: 
    std::unordered_map<std::string, BlockStats> blocks_;
    const BlockStats* active_block_ = nullptr; // points into blocks_, set by set_block()
    int num_channels_ = 0;

    void load(const std::string& constants_path);
    void validate_input(const torch::Tensor& input) const;
};

} // namespace ndr