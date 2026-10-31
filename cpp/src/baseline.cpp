#include "netsentinel/baseline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace netsentinel {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}  // namespace

double median(std::vector<double>& buf) {
    if (buf.empty()) return kNaN;
    const std::size_t mid = buf.size() / 2;
    const auto mid_it = buf.begin() + static_cast<std::ptrdiff_t>(mid);
    std::nth_element(buf.begin(), mid_it, buf.end());
    const double upper = *mid_it;
    if (buf.size() % 2 == 1) return upper;
    const double lower = *std::max_element(buf.begin(), mid_it);
    return (lower + upper) / 2.0;
}

std::vector<double> seasonal_baseline(const std::vector<double>& values,
                                      std::size_t n, std::size_t k,
                                      const std::vector<bool>& weekend,
                                      const BaselineConfig& cfg) {
    if (values.size() != n * k) throw std::invalid_argument("values must have n * k elements");
    if (weekend.size() != n) throw std::invalid_argument("weekend must have n elements");
    if (cfg.readings_per_day < 2 || cfg.baseline_days < 1 || cfg.fallback_window < 1)
        throw std::invalid_argument("invalid baseline config");

    // Lags in readings: same slot and +/- one slot, 1..baseline_days days back.
    std::vector<std::size_t> lags;
    for (int day = 1; day <= cfg.baseline_days; ++day)
        for (int offset = -1; offset <= 1; ++offset)
            lags.push_back(static_cast<std::size_t>(day * cfg.readings_per_day + offset));

    const auto window = static_cast<std::size_t>(cfg.fallback_window);
    std::vector<double> out(n * k, kNaN);
    std::vector<double> same_type, any_type, recent;
    same_type.reserve(lags.size());
    any_type.reserve(lags.size());
    recent.reserve(window);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            same_type.clear();
            any_type.clear();
            for (const std::size_t lag : lags) {
                if (lag > i) continue;
                const double v = values[(i - lag) * k + j];
                if (std::isnan(v)) continue;
                any_type.push_back(v);
                if (weekend[i] == weekend[i - lag]) same_type.push_back(v);
            }

            double base = median(same_type);
            if (std::isnan(base)) base = median(any_type);
            if (std::isnan(base)) {
                recent.clear();
                const std::size_t start = i > window ? i - window : 0;
                for (std::size_t p = start; p < i; ++p) {
                    const double v = values[p * k + j];
                    if (!std::isnan(v)) recent.push_back(v);
                }
                base = median(recent);
            }
            if (std::isnan(base)) base = values[i * k + j];
            out[i * k + j] = base;
        }
    }
    return out;
}

}  // namespace netsentinel
