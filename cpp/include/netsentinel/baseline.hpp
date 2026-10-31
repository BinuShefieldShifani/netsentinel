// baseline.hpp - per-cell time-of-day baseline for KPI readings
//
// C++ implementation of features.seasonal_baseline (Python reference).
// For every reading i and KPI j, the baseline is chosen in this order:
//   1. median of the same 15-min slot (+/- one slot) on earlier days of the
//      same type (weekday or weekend), up to `baseline_days` days back
//   2. if there is no such day: the same slots on any earlier day
//   3. if there is no earlier day at all: median of the previous
//      `fallback_window` readings
//   4. the very first reading: the reading itself
// Only past readings are used. NaN values are skipped.
#pragma once

#include <cstddef>
#include <vector>

namespace netsentinel {

struct BaselineConfig {
    int readings_per_day = 96;
    int baseline_days = 7;
    int fallback_window = 12;
};

// Median of the values in `buf` (reorders it). Returns NaN for an empty buffer.
// For an even count, the mean of the two middle values (same as numpy).
double median(std::vector<double>& buf);

// values:  row-major n x k matrix (n readings, k KPIs), in time order, no gaps
// weekend: n flags, true if the reading falls on a weekend
// returns: row-major n x k baseline
// Throws std::invalid_argument if the sizes don't match or the config is invalid.
std::vector<double> seasonal_baseline(const std::vector<double>& values,
                                      std::size_t n, std::size_t k,
                                      const std::vector<bool>& weekend,
                                      const BaselineConfig& cfg = {});

}  // namespace netsentinel
