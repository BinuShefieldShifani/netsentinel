"""
features.py - Turn raw KPI readings into "how unusual is this for this cell"
===========================================================================
PRB utilisation of 80% is normal at 19:00 in a busy urban cell and very odd
at 04:00 in a rural one. So each reading is compared with its own cell's
history at the same time of day:

    baseline  = median of the same 15-min slot (+/- one slot) on earlier
                days of the same type (weekday or weekend) in the past week;
                if there is no same-type day yet, the same slot on any
                earlier day; only on the very first day, the median of the
                previous 12 readings
    residual  = reading - baseline
    smoothed  = rolling mean of the residual over the last hour

Model input = the raw KPI values plus the smoothed residuals. That pair won
on validation days 8-10 against raw-only, residual-only and per-cell-scaled
variants (see experiments/select_features.py).

Only past readings are used, so the features can be computed live.

The baseline has two implementations that must give identical results:
  cpp/       C++17, used automatically if the _netsentinel_cpp module is built
  python     seasonal_baseline_python() below, the reference
Set NETSENTINEL_FEATURES=python to force the Python version.
"""

import os
import warnings

import numpy as np
import pandas as pd

from config import BASELINE_DAYS, FALLBACK_WINDOW, KPIS, READINGS_PER_DAY, RESIDUAL_SMOOTH

try:
    import _netsentinel_cpp as _cpp
except ImportError:          # not built; the Python reference is used
    _cpp = None

KPI_FEATURES = list(KPIS.keys())
RESIDUAL_COLUMNS = [f"{k}_res" for k in KPI_FEATURES]
SMOOTHED_COLUMNS = [f"{k}_res_mean{RESIDUAL_SMOOTH}" for k in KPI_FEATURES]
FEATURE_COLUMNS = KPI_FEATURES + SMOOTHED_COLUMNS


def baseline_backend() -> str:
    if _cpp is not None and os.getenv("NETSENTINEL_FEATURES", "auto").lower() != "python":
        return "cpp"
    return "python"


def _weekend_flags(cell_df: pd.DataFrame) -> np.ndarray:
    if "timestamp" in cell_df.columns:
        return (pd.to_datetime(cell_df["timestamp"]).dt.dayofweek >= 5).to_numpy()
    return np.zeros(len(cell_df), dtype=bool)


def seasonal_baseline(cell_df: pd.DataFrame) -> pd.DataFrame:
    """Baseline per KPI for one cell, using the C++ version if it is built."""
    if baseline_backend() == "cpp":
        values = np.ascontiguousarray(cell_df[KPI_FEATURES].to_numpy(dtype=float))
        base = _cpp.seasonal_baseline(values, _weekend_flags(cell_df), READINGS_PER_DAY,
                                      BASELINE_DAYS, FALLBACK_WINDOW)
        return pd.DataFrame(base, index=cell_df.index, columns=KPI_FEATURES)
    return seasonal_baseline_python(cell_df)


def seasonal_baseline_python(cell_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reference implementation. Rows must be in time order with no gaps.

    Order of preference for each reading:
      1. same 15-min slot (+/- one) on earlier days of the same type
         (weekday or weekend) within BASELINE_DAYS
      2. same slot on any earlier day, if there is no same-type day yet
         (e.g. the first weekend)
      3. median of the previous FALLBACK_WINDOW readings, only when there is
         no earlier day at all
    Option 3 adapts to whatever is happening right now, including a fault,
    so it is only used during the very first day.
    """
    values = cell_df[KPI_FEATURES].astype(float)
    n = len(values)
    weekend = _weekend_flags(cell_df)

    same_type, any_type = [], []
    for day in range(1, BASELINE_DAYS + 1):
        for offset in (-1, 0, 1):
            lag = day * READINGS_PER_DAY + offset
            past = values.shift(lag).to_numpy(copy=True)
            any_type.append(past.copy())
            match = np.zeros(n, dtype=bool)
            if lag < n:
                match[lag:] = weekend[lag:] == weekend[: n - lag]
            past[~match] = np.nan
            same_type.append(past)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN slices early on
        base = np.nanmedian(np.stack(same_type), axis=0)
        any_day = np.nanmedian(np.stack(any_type), axis=0)
    base = np.where(np.isnan(base), any_day, base)
    fallback = values.shift(1).rolling(FALLBACK_WINDOW, min_periods=1).median().to_numpy()
    base = np.where(np.isnan(base), fallback, base)
    base = np.where(np.isnan(base), values.to_numpy(), base)       # very first reading
    return pd.DataFrame(base, index=cell_df.index, columns=KPI_FEATURES)


def _cell_features(cell_df: pd.DataFrame) -> pd.DataFrame:
    residual = cell_df[KPI_FEATURES].astype(float) - seasonal_baseline(cell_df)
    residual.columns = RESIDUAL_COLUMNS
    smoothed = residual.rolling(RESIDUAL_SMOOTH, min_periods=1).mean()
    smoothed.columns = SMOOTHED_COLUMNS
    raw = cell_df[KPI_FEATURES].astype(float)
    return pd.concat([raw, residual, smoothed], axis=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features for every row of df, returned in the same row order.
    df may hold one cell or many; each cell is handled separately.
    """
    if "cell_id" not in df.columns or df["cell_id"].nunique() <= 1:
        ordered = df.sort_values("timestamp") if "timestamp" in df.columns else df
        return _cell_features(ordered).loc[df.index]
    parts = [_cell_features(g.sort_values("timestamp")) for _, g in df.groupby("cell_id", sort=False)]
    return pd.concat(parts).loc[df.index]
