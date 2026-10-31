"""
Time the baseline on the full dataset with each implementation.
Run:  python -m experiments.bench_baseline
"""

import time

import numpy as np

import features
from data_generator import generate_dataset


def time_it(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    df = generate_dataset()
    cells = [g for _, g in df.groupby("cell_id", sort=False)]
    print(f"{len(cells)} cells x {len(cells[0])} readings x {len(features.KPI_FEATURES)} KPIs")

    t_py = time_it(lambda: [features.seasonal_baseline_python(c) for c in cells])
    print(f"Python (numpy):  {t_py:6.2f} s")
    if features._cpp is None:
        print("C++ module not built; see README.")
        return
    t_cpp = time_it(lambda: [features.seasonal_baseline(c) for c in cells])
    print(f"C++:             {t_cpp:6.2f} s   ({t_py / t_cpp:.1f}x faster)")

    # The live pipeline scores one cell at a time with about a week of history.
    one = cells[0].iloc[:700]
    t_py1 = time_it(lambda: features.seasonal_baseline_python(one), 20)
    t_cpp1 = time_it(lambda: features.seasonal_baseline(one), 20)
    print(f"One cell, 700 readings: Python {t_py1 * 1000:.1f} ms, C++ {t_cpp1 * 1000:.1f} ms")
    diff = np.abs(features.seasonal_baseline(cells[0]).to_numpy()
                  - features.seasonal_baseline_python(cells[0]).to_numpy()).max()
    print(f"Max difference between implementations: {diff:.1e}")


if __name__ == "__main__":
    main()
