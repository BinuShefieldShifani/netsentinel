#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "netsentinel/baseline.hpp"

using netsentinel::BaselineConfig;
using netsentinel::median;
using netsentinel::seasonal_baseline;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Small days so the tests stay readable: 4 readings per day, 3 days back.
const BaselineConfig kSmall{4, 3, 2};

// One KPI, value = reading index.
std::vector<double> ramp(std::size_t n) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<double>(i);
    return v;
}

}  // namespace

// ── median ───────────────────────────────────────────────────────────────

TEST(Median, OddCount) {
    std::vector<double> v{5, 1, 3};
    EXPECT_DOUBLE_EQ(median(v), 3.0);
}

TEST(Median, EvenCountAveragesTheMiddleTwo) {
    std::vector<double> v{4, 1, 3, 2};
    EXPECT_DOUBLE_EQ(median(v), 2.5);
}

TEST(Median, EmptyIsNaN) {
    std::vector<double> v;
    EXPECT_TRUE(std::isnan(median(v)));
}

// ── seasonal_baseline ────────────────────────────────────────────────────

TEST(Baseline, FirstReadingIsItsOwnBaseline) {
    const auto out = seasonal_baseline({7.0, 9.0}, 2, 1, {false, false}, kSmall);
    EXPECT_DOUBLE_EQ(out[0], 7.0);
}

TEST(Baseline, FirstDayUsesRecentReadings) {
    // No earlier day yet: median of the previous readings (fallback_window = 2).
    // With 4 readings per day the shortest lag is 3 (one day minus one slot),
    // so readings 1 and 2 have no earlier day.
    const auto out = seasonal_baseline(ramp(5), 5, 1, std::vector<bool>(5, false), kSmall);
    EXPECT_DOUBLE_EQ(out[1], 0.0);          // only reading 0 before it
    EXPECT_DOUBLE_EQ(out[2], 0.5);          // readings 0 and 1
    EXPECT_DOUBLE_EQ(out[3], 0.0);          // reading 0 is "one day minus one slot" back
}

TEST(Baseline, UsesSameSlotOnEarlierDays) {
    // Reading 9 (day 2, slot 1). Lags 3,4,5 (day 1) and 7,8,9 (day 2) -> readings 6,5,4,2,1,0.
    const auto out = seasonal_baseline(ramp(10), 10, 1, std::vector<bool>(10, false), kSmall);
    EXPECT_DOUBLE_EQ(out[9], 3.0);          // median of {0,1,2,4,5,6}
}

TEST(Baseline, PrefersDaysOfTheSameType) {
    // Day 0 weekday, day 1 weekend, day 2 weekday. Reading 9 (weekday) must
    // ignore the weekend day even though it is more recent.
    std::vector<bool> weekend(12, false);
    for (std::size_t i = 4; i < 8; ++i) weekend[i] = true;
    std::vector<double> v(12, 10.0);
    for (std::size_t i = 4; i < 8; ++i) v[i] = 100.0;   // weekend values are very different
    const auto out = seasonal_baseline(v, 12, 1, weekend, kSmall);
    EXPECT_DOUBLE_EQ(out[9], 10.0);
}

TEST(Baseline, FallsBackToAnyDayOnFirstWeekend) {
    // Day 0 weekday, day 1 is the first weekend day: no earlier weekend exists,
    // so it uses the weekday slots instead of the last readings (which may be a fault).
    std::vector<bool> weekend{false, false, false, false, true, true, true, true};
    std::vector<double> v{10, 10, 10, 10, 50, 50, 50, 50};   // a "fault" starts on the weekend
    const auto out = seasonal_baseline(v, 8, 1, weekend, kSmall);
    // Readings 5 and 6: one day back is weekday 1-3. (Reading 7's "previous slot"
    // lag lands on reading 4, which is already the weekend, so it is not used here.)
    EXPECT_DOUBLE_EQ(out[5], 10.0);   // not 50: the fault does not become its own baseline
    EXPECT_DOUBLE_EQ(out[6], 10.0);
}

TEST(Baseline, NeverLooksAtTheCurrentOrFutureReadings) {
    auto v = ramp(20);
    const auto before = seasonal_baseline(v, 20, 1, std::vector<bool>(20, false), kSmall);
    v[12] = 1e9;
    const auto after = seasonal_baseline(v, 20, 1, std::vector<bool>(20, false), kSmall);
    for (std::size_t i = 0; i <= 12; ++i) EXPECT_DOUBLE_EQ(before[i], after[i]) << "reading " << i;
}

TEST(Baseline, SkipsMissingValues) {
    std::vector<double> v{1, kNaN, 3, 4};
    const auto out = seasonal_baseline(v, 4, 1, std::vector<bool>(4, false), kSmall);
    EXPECT_DOUBLE_EQ(out[2], 1.0);    // previous two readings are 1 and NaN -> 1
}

TEST(Baseline, KpisAreIndependentColumns) {
    // Two KPIs, row-major: column 1 = 10 x column 0.
    std::vector<double> v;
    for (int i = 0; i < 10; ++i) { v.push_back(i); v.push_back(10.0 * i); }
    const auto out = seasonal_baseline(v, 10, 2, std::vector<bool>(10, false), kSmall);
    for (std::size_t i = 0; i < 10; ++i) EXPECT_DOUBLE_EQ(out[i * 2 + 1], 10.0 * out[i * 2]);
}

TEST(Baseline, RejectsMismatchedSizes) {
    EXPECT_THROW(seasonal_baseline({1, 2, 3}, 2, 2, {false, false}, kSmall), std::invalid_argument);
    EXPECT_THROW(seasonal_baseline({1, 2}, 2, 1, {false}, kSmall), std::invalid_argument);
    EXPECT_THROW(seasonal_baseline({1, 2}, 2, 1, {false, false}, BaselineConfig{1, 1, 1}),
                 std::invalid_argument);
}
