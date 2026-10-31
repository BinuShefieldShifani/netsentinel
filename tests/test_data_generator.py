import numpy as np
import pytest

from config import ANOMALY_SCENARIOS, KPIS
from data_generator import generate_dataset, simulate


def test_shape_and_columns():
    df = simulate(n_cells=3, n_timestamps=50, seed=1)
    assert len(df) == 150
    assert df["cell_id"].nunique() == 3
    for col in ["cell_id", "cell_type", "timestamp", "is_anomaly", "scenario", *KPIS]:
        assert col in df.columns


def test_same_seed_gives_same_data():
    a = simulate(n_cells=2, n_timestamps=100, seed=3)
    b = simulate(n_cells=2, n_timestamps=100, seed=3)
    c = simulate(n_cells=2, n_timestamps=100, seed=4)
    assert a.equals(b)
    assert not a.equals(c)


def test_values_stay_within_physical_ranges(dataset):
    assert dataset["rsrp"].between(-130, -44).all()
    assert dataset["rsrq"].between(-19.5, -3).all()
    assert dataset["sinr"].between(-20, 30).all()
    assert dataset["prb_dl"].between(0, 100).all()
    assert dataset["prb_ul"].between(0, 100).all()
    assert dataset["cqi"].between(1, 15).all()
    assert dataset["ho_sr"].between(0, 100).all()
    assert (dataset["tp_dl"] >= 0).all() and (dataset["tp_ul"] >= 0).all()


def test_anomaly_rate_close_to_config(dataset):
    assert 0.06 < dataset["is_anomaly"].mean() < 0.10


def test_labels_are_consistent(dataset):
    assert set(dataset.loc[dataset["is_anomaly"] == 0, "scenario"]) == {"normal"}
    assert set(dataset.loc[dataset["is_anomaly"] == 1, "scenario"]) <= set(ANOMALY_SCENARIOS)


# Each scenario must push its signature KPI in the documented direction.
SIGNATURES = [
    ("interference", "sinr", "down"),
    ("capacity_overload", "prb_dl", "up"),
    ("hardware_fault", "rsrp", "down"),
    ("coverage_gap", "rsrp", "down"),
    ("handover_storm", "ho_sr", "down"),
]


@pytest.mark.parametrize("scenario,kpi,direction", SIGNATURES)
def test_scenario_signatures(dataset, scenario, kpi, direction):
    normal = dataset.loc[dataset["scenario"] == "normal", kpi].mean()
    faulty = dataset.loc[dataset["scenario"] == scenario, kpi].mean()
    assert (faulty < normal) if direction == "down" else (faulty > normal)


def test_generate_dataset_writes_and_reloads(tmp_path, monkeypatch):
    import data_generator
    small = simulate(n_cells=2, n_timestamps=20, seed=0)
    monkeypatch.setattr(data_generator, "simulate", lambda: small)
    path = tmp_path / "kpi.csv"
    first = generate_dataset(force=True, path=str(path))
    second = generate_dataset(path=str(path))
    assert path.exists()
    assert len(first) == len(second) == 40
    assert np.allclose(first["rsrp"], second["rsrp"])
