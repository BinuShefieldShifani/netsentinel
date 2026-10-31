import numpy as np
import pandas as pd
import pytest

from config import ANOMALY_SCENARIOS, EVENT_MAX_LEN, EVENT_MIN_LEN, KPIS
from data_generator import event_shape, generate_dataset, simulate

QUIET = lambda df: df[(df["is_anomaly"] == 0) & (df["benign_event"] == 0)]   # noqa: E731


def test_shape_and_columns():
    df = simulate(n_cells=3, n_timestamps=200, seed=1)
    assert len(df) == 600
    assert df["cell_id"].nunique() == 3
    for col in ["cell_id", "cell_type", "timestamp", "is_anomaly", "scenario",
                "event_id", "severity", "benign_event", *KPIS]:
        assert col in df.columns


def test_same_seed_gives_same_data():
    a = simulate(n_cells=2, n_timestamps=300, seed=3)
    b = simulate(n_cells=2, n_timestamps=300, seed=3)
    c = simulate(n_cells=2, n_timestamps=300, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_values_stay_within_physical_ranges(dataset):
    assert dataset["rsrp"].between(-130, -44).all()
    assert dataset["rsrq"].between(-19.5, -3).all()
    assert dataset["sinr"].between(-20, 30).all()
    assert dataset["prb_dl"].between(0, 100).all()
    assert dataset["prb_ul"].between(0, 100).all()
    assert dataset["cqi"].between(1, 15).all()
    assert dataset["ho_sr"].between(0, 100).all()
    assert dataset["rlf_rate"].between(0, 10).all()
    assert (dataset["tp_dl"] > 0).all() and (dataset["tp_ul"] > 0).all()


def test_fault_share_close_to_config(dataset):
    assert 0.03 < dataset["is_anomaly"].mean() < 0.10


def test_labels_are_consistent(dataset):
    faults = dataset[dataset["is_anomaly"] == 1]
    assert set(faults["scenario"]) == set(ANOMALY_SCENARIOS)
    assert (faults["event_id"] != "").all() and (faults["severity"] > 0).all()
    assert (QUIET(dataset)["event_id"] == "").all()
    benign = dataset[dataset["benign_event"] == 1]
    assert len(benign) > 0
    assert (benign["is_anomaly"] == 0).all() and (benign["scenario"] == "normal").all()


def test_faults_are_contiguous_events_of_valid_length(dataset):
    for _, g in dataset[dataset["is_anomaly"] == 1].groupby("event_id"):
        assert g["cell_id"].nunique() == 1
        assert np.all(np.diff(g.index) == 1), "event rows must be consecutive"
        assert len(g) <= EVENT_MAX_LEN
        assert g["scenario"].nunique() == 1


def test_event_shape_ramps_up_and_down():
    shape = event_shape(20, None)
    assert len(shape) == 20
    assert shape[0] < shape[10] == 1.0 > shape[-1]
    assert len(event_shape(EVENT_MIN_LEN, None)) == EVENT_MIN_LEN


def test_kpis_move_together_in_normal_operation(dataset):
    """Physically linked KPIs must be correlated (they were independent before)."""
    q = QUIET(dataset)
    assert q["sinr"].corr(q["cqi"]) > 0.8
    assert q["cqi"].corr(q["tp_dl"]) > 0.5
    assert q["prb_dl"].corr(q["tp_dl"]) < -0.5


def test_daily_traffic_pattern(dataset):
    q = QUIET(dataset)
    hour = pd.to_datetime(q["timestamp"]).dt.hour
    assert q.loc[hour.between(18, 20), "prb_dl"].mean() > 2 * q.loc[hour.between(2, 4), "prb_dl"].mean()


def test_benign_surges_raise_load_without_breaking_radio(dataset):
    q, b = QUIET(dataset), dataset[dataset["benign_event"] == 1]
    assert b["prb_dl"].mean() > q["prb_dl"].mean()
    assert abs(b["rsrp"].mean() - q["rsrp"].mean()) < 3


SIGNATURES = [
    ("interference", "sinr", "down"),
    ("capacity_overload", "prb_dl", "up"),
    ("hardware_fault", "rsrp", "down"),
    ("coverage_gap", "ho_sr", "down"),
    ("handover_storm", "ho_sr", "down"),
]


@pytest.mark.parametrize("scenario,kpi,direction", SIGNATURES)
def test_scenario_signatures(dataset, scenario, kpi, direction):
    normal = QUIET(dataset)[kpi].mean()
    faulty = dataset.loc[(dataset["scenario"] == scenario) & (dataset["severity"] >= 0.55), kpi].mean()
    assert (faulty < normal) if direction == "down" else (faulty > normal)


def test_generate_dataset_writes_and_reloads(tmp_path, monkeypatch):
    import data_generator
    small = simulate(n_cells=2, n_timestamps=300, seed=0)
    monkeypatch.setattr(data_generator, "simulate", lambda: small)
    path = tmp_path / "kpi.csv"
    first = generate_dataset(force=True, path=str(path))
    second = generate_dataset(path=str(path))
    assert len(first) == len(second) == 600
    assert np.allclose(first["rsrp"], second["rsrp"])
    assert (second["event_id"] == first["event_id"]).all()   # empty ids survive the CSV round trip
