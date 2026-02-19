#ifndef FUZZYLOGIC_H
#define FUZZYLOGIC_H

#include <cmath>
#include <algorithm>
#include <initializer_list>

namespace Fuzzy {

// Sigmoid membership: smooth 0->1 transition centered at 'center'.
// Returns exactly 0.5 at center. Steepness controls transition width
// (smaller = wider band, larger = sharper).
inline double sigmoid(double x, double center, double steepness) {
    return 1.0 / (1.0 + std::exp(-(x - center) / steepness));
}

// Inverse sigmoid: 1.0 when x is low, drops to 0.0 as x exceeds center.
// Useful for cooldown readiness (high readiness when time has passed).
inline double inverseSigmoid(double x, double center, double steepness) {
    return 1.0 - sigmoid(x, center, steepness);
}

// Exponential decay: value decreases over elapsed time.
// Pattern from Generative Agents: factor^elapsed
// Example: decay(0.995, 60) = 0.995^60 ≈ 0.74
inline double decay(double factor, double elapsed) {
    return std::pow(factor, elapsed);
}

// Gaussian membership: bell curve centered at 'center'.
// Returns 1.0 at center, drops off symmetrically by sigma.
inline double gaussian(double x, double center, double sigma) {
    double delta = x - center;
    return std::exp(-(delta * delta) / (sigma * sigma));
}

// Weighted factor for multi-signal combination.
struct WeightedFactor {
    double weight;
    double value; // should be in [0, 1]
};

// Weighted score: normalized weighted average of multiple factors.
// Returns sum(w_i * v_i) / sum(w_i).
inline double weightedScore(std::initializer_list<WeightedFactor> factors) {
    double sum = 0.0;
    double weightSum = 0.0;
    for (const auto &f : factors) {
        sum += f.weight * f.value;
        weightSum += f.weight;
    }
    return weightSum > 0.0 ? sum / weightSum : 0.0;
}

// Convenience: soft threshold (same as sigmoid, reads better at call sites).
inline double softThreshold(double x, double threshold, double width) {
    return sigmoid(x, threshold, width);
}

// Clamp value to [0, 1].
inline double clamp01(double x) {
    return std::max(0.0, std::min(1.0, x));
}

} // namespace Fuzzy

#endif // FUZZYLOGIC_H
