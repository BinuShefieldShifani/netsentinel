import json

import numpy as np
import pandas as pd
import pytest

from config import ANOMALY_SCORE_CRITICAL, ANOMALY_SCORE_WARNING, F1_TARGET, PRECISION_TARGET
from model_trainer import (
    evaluate,
    score_cell_window,
    score_readings,
    severity_from_score,
    time_split,
    train,
)


def test_scores_separate_normal_from_anomalous(dataset, model_and_scaler):
    model, scaler = model_and_scaler
    _, test_df = time_split(dataset)
    scores = score_readings(test_df, model, scaler)
    normal = np.median(scores[test_df["is_anomaly"].values == 0])
    faulty = np.median(scores[test_df["is_anomaly"].values == 1])
    assert normal > 0 > faulty


def test_healthy_cell_is_not_critical(dataset, model_and_scaler):
    """Regression: the old scorer used the wrong score scale and marked every cell CRITICAL."""
    model, scaler = model_and_scaler
    _, test_df = time_split(dataset)
    healthy_rows = test_df.index[test_df["is_anomaly"] == 0][:200]
    severities = []
    for i in healthy_rows:
        cell = dataset[dataset["cell_id"] == dataset.loc[i, "cell_id"]].loc[:i].tail(60)
        score, _ = score_cell_window(cell, model, scaler)
        severities.append(severity_from_score(score))
    assert severities.count("CRITICAL") == 0
    assert severities.count("NORMAL") / len(severities) > 0.95


def test_score_cell_window_uses_latest_reading(dataset, model_and_scaler):
    model, scaler = model_and_scaler
    cell = dataset[dataset["cell_id"] == "CELL_0001"].head(30)
    score, _ = score_cell_window(cell, model, scaler)
    assert score == pytest.approx(float(score_readings(cell.tail(1), model, scaler)[0]))


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


def test_held_out_metrics_meet_minimum_targets(dataset, model_and_scaler):
    model, scaler = model_and_scaler
    _, test_df = time_split(dataset)
    m = evaluate(test_df, model, scaler)
    assert m["precision"] >= PRECISION_TARGET
    assert m["f1"] >= F1_TARGET


def test_train_writes_artifacts(dataset, tmp_path):
    paths = {k: str(tmp_path / f) for k, f in
             [("model_path", "m.pkl"), ("scaler_path", "s.pkl"), ("metrics_path", "metrics.json")]}
    metrics = train(df=dataset, **paths)
    for p in paths.values():
        assert (tmp_path / p.split("/")[-1]).exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["f1"] == metrics["f1"]
    assert set(saved["detection_rate_by_scenario"]) == {
        "interference", "capacity_overload", "hardware_fault", "coverage_gap", "handover_storm"}
