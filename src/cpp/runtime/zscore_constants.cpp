#include <runtime/zscore_constants.hpp>

#include <fstream>
#include <stdexcept>
#include <vector>

#include <third_party/json.hpp>

using json = nlohmann::json;

namespace ndr{

ZScoreConstants::ZScoreConstants(const std::string& constants_path) {
    load(constants_path);
}
    
void ZScoreConstants::load(const std::string& constants_path) {
    std::ifstream f(constants_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open constants file: " + constants_path);
    }

    json j = json::parse(f);

    for (auto& [block_id, stats] : j.items()) {
        std::vector<float> means_vec = stats.at("means").get<std::vector<float>>();
        std::vector<float> stds_vec  = stats.at("stds").get<std::vector<float>>();

        if (means_vec.size() != stds_vec.size()) {
            throw std::runtime_error(
                "Block " + block_id + ": means and stds length mismatch"
            );
        }

        // validate no zero stds to prevent division by zero
        for (size_t i = 0; i < stds_vec.size(); ++i) {
            if (stds_vec[i] == 0.0f) {
                throw std::runtime_error(
                    "Block " + block_id + ": zero std on channel " + std::to_string(i)
                );
            }
        }

        int C = static_cast<int>(means_vec.size());

        // Set num_channels_ on fist block, validate consistency across blocks
        if (num_channels_ == 0) {
            num_channels_ = C;
        } else if (C != num_channels_) {
            throw std::runtime_error(
                "Block " + block_id + ": channel count " + std::to_string(C) +
                " does not match expected " + std::to_string(num_channels_)
            );
        }

        BlockStats bs;
        bs.means = torch::tensor(means_vec); // [C] float32
        bs.stds  = torch::tensor(stds_vec);  // [C] float32
        blocks_[block_id] = std::move(bs); 
    }

    if (blocks_.empty()) {
        throw std::runtime_error("constants.json contains no blocks");
    }
}

void ZScoreConstants::set_block(const std::string& block_id) {
    auto it = blocks_.find(block_id);
    if (it == blocks_.end()) {
        throw std::runtime_error(
            "Block '" + block_id + "' not found in constants"
        );
    }
    active_block_ = &it->second;
}

bool ZScoreConstants::has_block(const std::string& block_id) const{
    return blocks_.count(block_id) > 0;
}

void ZScoreConstants::validate_input(const torch::Tensor& input) const {
    if (active_block_ == nullptr) {
        throw std::runtime_error(
            "ZScoreConstants::normalize() called before set_block()"
        );
    }
    
    if (input.dtype() != torch::kFloat32) {
        throw std::runtime_error(
            "Input tensor must be float32"
        );
    }

    if (input.dim() != 2) {
        throw std::runtime_error(
            "Input tensor must be 2D [T,C], got " + 
            std::to_string(input.dim()) + "D"
        );
    }

    if (input.size(1) != num_channels_) {
        throw std::runtime_error(
            "Input channel dim " + std::to_string(input.size(1)) + 
            " does not match constants num_channels " +
            std::to_string(num_channels_)
        );
    }

}

torch::Tensor ZScoreConstants::normalize(const torch::Tensor& input) const {
    validate_input(input);

    // broadcasting: means/stds are [C], input is [T,C]
    return (input - active_block_->means) / active_block_->stds;
}

} // namespace ndr