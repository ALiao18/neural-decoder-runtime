#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace ndr {

class LatencyTimer {
public:
    // stores std::chrono::high_resolution_clock::now() keyed by label
    void start(const std::string& label);

    // computes duration since start(label), append to samples_[label]
    void stop(const std::string& label);

    // for each label, sort samples, prints p50, 95, 99
    void report(int n) const;

    // clears all samples for clean re-runs
    void reset();

private:
    using Clock     = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    // Completed duration samples per label, in ms
    std::unordered_map<std::string, std::vector<double>> samples_;

    // Active start times, present between start() and stop()
    std::unordered_map<std::string, TimePoint> active_;

    // Compute percentile (0-1) from a sorted vector
    // vector must be sorted before calling, throws if empty
    static double percentile(const std::vector<double>& sorted_samples, double p);
};
}