import numpy as np
import pandas as pd

from config import READINGS_PER_DAY
from features import FEATURE_COLUMNS, RESIDUAL_COLUMNS, build_features, seasonal_baseline


def _one_cell(dataset, cell="CELL_0001"):
    return dataset[dataset["cell_id"] == cell]


def test_feature_columns_and_row_order(dataset, features):
    assert list(features.columns[: len(FEATURE_COLUMNS)]) != []
    for col in FEATURE_COLUMNS + RESIDUAL_COLUMNS:
        assert col in features.columns
    assert features.index.equals(dataset.index)
    assert not features[FEATURE_COLUMNS].isna().any().any()


def test_baseline_uses_only_the_past(dataset):
    cell = _one_cell(dataset).copy()
    before = seasonal_baseline(cell)
    i = cell.index[500]
    cell.loc[i, "prb_dl"] = 999.0                      # change one reading
    after = seasonal_baseline(cell)
    assert before.loc[: i].equals(after.loc[: i])      # nothing up to and including it changes


def test_baseline_removes_daily_pattern(dataset, features):
    """
    Raw PRB swings with the time of day. After subtracting the baseline, the
    average residual should be about the same at every hour.
    """
    quiet = ((dataset["event_id"] == "") & (np.arange(len(dataset)) % 1344 >= 7 * READINGS_PER_DAY)).to_numpy()
    hour = pd.to_datetime(dataset.loc[quiet, "timestamp"]).dt.hour
    raw_by_hour = dataset.loc[quiet, "prb_dl"].groupby(hour).mean()
    res_by_hour = features.loc[quiet, "prb_dl_res"].groupby(hour).mean()
    assert raw_by_hour.max() - raw_by_hour.min() > 20
    assert res_by_hour.abs().max() < 3


def test_monday_is_compared_with_weekdays_not_the_weekend(dataset):
    """Day 8 is a Monday; its baseline must come from earlier weekdays, not Sat/Sun."""
    cell = dataset[dataset["cell_id"] == "CELL_0001"].copy()
    monday_19h = 7 * READINGS_PER_DAY + 76
    weekend_rows = cell.index[5 * READINGS_PER_DAY: 7 * READINGS_PER_DAY]
    before = seasonal_baseline(cell).iloc[monday_19h]
    cell.loc[weekend_rows, "prb_dl"] = 0.0          # wipe out the weekend
    after = seasonal_baseline(cell).iloc[monday_19h]
    assert before["prb_dl"] == after["prb_dl"]


def test_single_cell_and_multi_cell_give_same_features(dataset, features):
    cell = _one_cell(dataset, "CELL_0003")
    alone = build_features(cell)
    pd.testing.assert_frame_equal(alone, features.loc[cell.index])


def test_works_without_timestamp_or_cell_id():
    df = pd.DataFrame({k: np.linspace(1, 2, 30) for k in
                       ["rsrp", "rsrq", "sinr", "prb_dl", "prb_ul", "cqi", "ho_sr", "rlf_rate", "tp_dl", "tp_ul"]})
    out = build_features(df)
    assert len(out) == 30 and not out.isna().any().any()
