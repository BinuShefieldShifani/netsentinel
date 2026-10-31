import numpy as np
import pandas as pd

from evaluation import (
    alarm_onsets,
    alarms_from_flags,
    evaluate_flags,
    fit_static_thresholds,
    static_threshold_flags,
)


def test_alarm_needs_two_of_three_flags():
    flags = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
    cells = np.array(["A"] * 10)
    assert alarms_from_flags(flags, cells).tolist() == [
        False, False, False, False, True, True, True, False, False, False]


def test_alarms_do_not_leak_across_cells():
    flags = np.array([0, 1, 1, 1, 0, 0], dtype=bool)
    cells = np.array(["A", "A", "A", "B", "B", "B"])
    alarms = alarms_from_flags(flags, cells)
    assert alarms.tolist() == [False, False, True, False, False, False]
    assert alarm_onsets(alarms, cells).tolist() == [False, False, True, False, False, False]


def _toy():
    """One cell, 40 readings: a fault at 5-10, a benign surge at 20-24, nothing else."""
    n = 40
    df = pd.DataFrame({
        "cell_id": "A", "timestamp": [f"2024-01-01T{i:03d}" for i in range(n)],
        "is_anomaly": 0, "scenario": "normal", "event_id": "", "severity": 0.0, "benign_event": 0,
    })
    df.loc[5:10, ["is_anomaly", "scenario", "event_id", "severity"]] = [1, "interference", "EV1", 0.9]
    df.loc[20:24, ["event_id", "benign_event"]] = ["EV2", 1]
    return df


def test_event_detection_delay_and_false_alarms():
    df = _toy()
    flags = np.zeros(len(df), dtype=bool)
    flags[[7, 8]] = True            # fault detected: alarm at reading 8 -> 3 readings after start
    flags[[30, 31]] = True          # an alarm far from any event -> false alarm
    out = evaluate_flags(df, flags, np.ones(len(df), dtype=bool))
    ev = out["events"]
    assert ev["n_fault_events"] == 1 and ev["event_recall"] == 1.0
    assert ev["median_delay_min"] == 3 * 15
    assert ev["false_alarms_per_cell_day"] == round(1 / (40 / 96), 3)
    assert ev["benign_surges_alarmed"] == 0.0


def test_alarm_just_after_fault_is_not_false():
    df = _toy()
    flags = np.zeros(len(df), dtype=bool)
    flags[[11, 12]] = True          # within the 1-hour grace period after the fault
    ev = evaluate_flags(df, flags, np.ones(len(df), dtype=bool))["events"]
    assert ev["false_alarms_per_cell_day"] == 0.0
    assert ev["event_recall"] == 0.0     # but it doesn't count as catching the fault either


def test_benign_surge_alarm_is_counted():
    df = _toy()
    flags = np.zeros(len(df), dtype=bool)
    flags[[21, 22]] = True
    ev = evaluate_flags(df, flags, np.ones(len(df), dtype=bool))["events"]
    assert ev["benign_surges_alarmed"] == 1.0


def test_only_events_starting_in_eval_period_are_scored():
    df = _toy()
    eval_mask = np.arange(len(df)) >= 15
    ev = evaluate_flags(df, np.zeros(len(df), dtype=bool), eval_mask)["events"]
    assert ev["n_fault_events"] == 0


def test_static_thresholds_respect_kpi_direction(dataset):
    limits = fit_static_thresholds(dataset)
    assert limits["sinr"][0] == "low" and limits["prb_dl"][0] == "high"
    flags = static_threshold_flags(dataset, limits)
    assert 0 < flags.mean() < 0.2
