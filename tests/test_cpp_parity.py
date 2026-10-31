"""
The C++ baseline must give the same numbers as the Python reference.
Skipped when the module isn't built, unless NETSENTINEL_REQUIRE_CPP=1 (set in CI).
"""

import os

import numpy as np
import pandas as pd
import pytest

import features
from features import KPI_FEATURES, seasonal_baseline_python

if features._cpp is None:
    if os.getenv("NETSENTINEL_REQUIRE_CPP") == "1":
        raise RuntimeError("NETSENTINEL_REQUIRE_CPP=1 but _netsentinel_cpp is not built")
    pytest.skip("C++ module not built (see README: Building the C++ component)", allow_module_level=True)

cpp = features._cpp


def _cpp_baseline(cell_df):
    values = np.ascontiguousarray(cell_df[KPI_FEATURES].to_numpy(dtype=float))
    out = cpp.seasonal_baseline(values, features._weekend_flags(cell_df))
    return pd.DataFrame(out, index=cell_df.index, columns=KPI_FEATURES)


@pytest.mark.parametrize("cell", ["CELL_0001", "CELL_0007", "CELL_0013", "CELL_0020"])
def test_cpp_matches_python_on_full_cells(dataset, cell):
    c = dataset[dataset["cell_id"] == cell]
    pd.testing.assert_frame_equal(_cpp_baseline(c), seasonal_baseline_python(c), rtol=0, atol=1e-12)


@pytest.mark.parametrize("length", [1, 2, 13, 95, 97, 300])
def test_cpp_matches_python_on_short_histories(dataset, length):
    c = dataset[dataset["cell_id"] == "CELL_0003"].iloc[:length]
    pd.testing.assert_frame_equal(_cpp_baseline(c), seasonal_baseline_python(c), rtol=0, atol=1e-12)


def test_cpp_matches_python_with_missing_values(dataset):
    c = dataset[dataset["cell_id"] == "CELL_0004"].iloc[:500].copy()
    rng = np.random.default_rng(0)
    for kpi in KPI_FEATURES:
        c.loc[c.index[rng.choice(500, 40, replace=False)], kpi] = np.nan
    pd.testing.assert_frame_equal(_cpp_baseline(c), seasonal_baseline_python(c), rtol=0, atol=1e-12)


def test_cpp_matches_python_without_timestamps():
    df = pd.DataFrame({k: np.sin(np.arange(250) / 7.0 + i) for i, k in enumerate(KPI_FEATURES)})
    pd.testing.assert_frame_equal(_cpp_baseline(df), seasonal_baseline_python(df), rtol=0, atol=1e-12)


def test_build_features_is_the_same_with_either_backend(dataset, monkeypatch):
    c = dataset[dataset["cell_id"] == "CELL_0009"]
    monkeypatch.setenv("NETSENTINEL_FEATURES", "python")
    assert features.baseline_backend() == "python"
    py = features.build_features(c)
    monkeypatch.setenv("NETSENTINEL_FEATURES", "auto")
    assert features.baseline_backend() == "cpp"
    pd.testing.assert_frame_equal(features.build_features(c), py, rtol=0, atol=1e-9)


def test_cpp_rejects_bad_input():
    with pytest.raises(ValueError):
        cpp.seasonal_baseline(np.zeros(10), np.zeros(10, dtype=bool))          # 1-D values
    with pytest.raises(ValueError):
        cpp.seasonal_baseline(np.zeros((10, 2)), np.zeros(9, dtype=bool))      # wrong length
