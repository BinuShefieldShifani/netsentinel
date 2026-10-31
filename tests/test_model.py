import json

import numpy as np
import pandas as pd
import pytest

from config import ANOMALY_SCORE_CRITICAL, ANOMALY_SCORE_WARNING
from model_trainer import (
    score_cell_window,
    score_readings,
    severity_from_score,
    time_split,
    train,
    train_mask,
)
from tests.conftest import mid_event_rows, quiet_rows, window_ending_at


def test_scores_separate_quiet_from_severe_faults(dataset, features, model_and_scaler):
    model, scaler = model_and_scaler
    scores = score_readings(dataset, model, scaler, features=features)
    held_out = ~train_mask(dataset)
    quiet = held_out & (dataset["event_id"] == "").to_numpy()
    severe = held_out & (dataset["severity"] >= 0.55).to_numpy()
    assert np.median(scores[quiet]) > 0 > np.median(scores[severe])


def test_healthy_cells_are_rarely_flagged(dataset, model_and_scaler):
    """Regression: the old scorer used the wrong score scale and marked every cell CRITICAL."""
    model, scaler = model_and_scaler
    severities = [severity_from_score(score_cell_window(window_ending_at(dataset, i), model, scaler)[0])
                  for i in quiet_rows(dataset)]
    assert severities.count("CRITICAL") / len(severities) < 0.02
    assert severities.count("NORMAL") / len(severities) > 0.95


def test_score_cell_window_matches_full_scoring(dataset, features, model_and_scaler):
    """Scoring the latest reading of a window must equal scoring that row in the full dataset."""
    model, scaler = model_and_scaler
    i = dataset.index[dataset["cell_id"] == "CELL_0002"][700]
    window = window_ending_at(dataset, i)
    score, _ = score_cell_window(window, model, scaler)
    full = score_readings(dataset, model, scaler, features=features)
    assert score == pytest.approx(full[i], abs=1e-9)


def test_window_order_does_not_matter(dataset, model_and_scaler):
    model, scaler = model_and_scaler
    i = mid_event_rows(dataset, "interference")[0]
    w = window_ending_at(dataset, i)
    assert score_cell_window(w, model, scaler)[0] == pytest.approx(
        score_cell_window(w.sample(frac=1, random_state=0), model, scaler)[0])


@pytest.mark.parametrize("score,expected", [
    (0.2, "NORMAL"),
    (ANOMALY_SCORE_WARNING, "NORMAL"),
    (ANOMALY_SCORE_WARNING - 0.001, "WARNING"),
    (ANOMALY_SCORE_CRITICAL, "CRITICAL"),
    (-0.5, "CRITICAL"),
])
def test_severity_thresholds(score, expected):
    assert severity_from_score(score) == expected


def test_time_split_has_no_overlap(dataset):
    train_df, test_df = time_split(dataset)
    assert len(train_df) + len(test_df) == len(dataset)
    assert pd.to_datetime(train_df["timestamp"]).max() < pd.to_datetime(test_df["timestamp"]).min()


def test_train_writes_artifacts_and_comparison(dataset, tmp_path):
    paths = {"model_path": str(tmp_path / "m.pkl"), "scaler_path": str(tmp_path / "s.pkl"),
             "metrics_path": str(tmp_path / "metrics.json")}
    metrics = train(df=dataset, **paths)
    assert (tmp_path / "m.pkl").exists() and (tmp_path / "s.pkl").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved == json.loads(json.dumps(metrics))
    assert set(saved["detectors"]) == {"static_thresholds", "iforest_raw", "iforest_baseline"}


def test_model_beats_static_thresholds_on_held_out_days(dataset, tmp_path):
    """Quality gate: if a change makes the model no better than fixed thresholds, CI fails."""
    m = train(df=dataset, model_path=str(tmp_path / "m.pkl"), scaler_path=str(tmp_path / "s.pkl"),
              metrics_path=str(tmp_path / "x.json"))["detectors"]
    ours, static = m["iforest_baseline"], m["static_thresholds"]
    assert ours["point"]["f1"] > static["point"]["f1"]
    assert ours["events"]["event_recall"] > static["events"]["event_recall"]
    assert ours["events"]["event_recall_severe"] >= 0.85
    assert ours["events"]["false_alarms_per_cell_day"] < 0.5
