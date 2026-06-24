#include "latency_timer.hpp"

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <iostream>

namespace ndr {

void LatencyTimer::start(const std::string& label) {
    if (active_.count(label) > 0) {
        throw std::runtime_error(
            std::string("LatnecyTimer::start() called twice for label '") + label + 
            std::string("' without stop()")
        );
    }
    active_[label] = Clock::now();
}

void LatencyTimer::stop(const std::string& label) {
    auto it = active_.find(label);
    if (it == active_.end()) {
        throw std::runtime_error(
            std::string("LatencyTimer::stop() called for label '") + label + 
            std::string("' without matching start()")
        );
    }

    auto duration_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - it->second             // now - second value of it, ie start time
    ).count();                                  // stores duration in ms as double   

    samples_[label].push_back(duration_ms);     // append to samples for label
    active_.erase(it);
}

void LatencyTimer::reset() {
    active_.clear();
    samples_.clear();
}

double LatencyTimer::percentile(const std::vector<double>& sorted_samples, double p) {
    if (sorted_samples.empty()) {
        throw std::runtime_error("LatencyTimer::percentile() called with empty vector");
    }

    // nearest-rank method: https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method
    size_t idx = static_cast<size_t>(std::ceil(p * sorted_samples.size())) - 1;
    idx = std::min(idx, sorted_samples.size() - 1); // clamp to last index
    return sorted_samples[idx];
}

void LatencyTimer::report(int n) const {
    // component order for output: total, preprocessing, inference, decoding, LM
    const std::vector<std::string> order = {
        "total", "preprocessing", "inference", "decoding", "lm"
    };

    std::cout << "{\n";
    std::cout << "  \"n\": " << n << ",\n";
    std::cout << "  \"latency_ms\": {\n";

    for (size_t i = 0; i < order.size(); ++i) {
        const auto& label = order[i];
        auto it           = samples_.find(label);

        std::cout << "    \"" << label << "\": ";
        
        if (it == samples_.end() || it->second.empty()){
            std::cout << "null";
        } else {
            auto sorted = it->second;
            std::sort(sorted.begin(), sorted.end());

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "{"
                      << "\"p50\": " << percentile(sorted, 0.50) << ", "
                      << "\"p95\": " << percentile(sorted, 0.95) << ", "
                      << "\"p99\": " << percentile(sorted, 0.99)
                      << "}";
        }

        if (i + 1 < order.size()) std::cout << ",";
        std::cout << "\n";
    }

    std::cout << "  }\n}\n}";
}

} // namespace ndr
