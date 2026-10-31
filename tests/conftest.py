"""Shared fixtures. Everything runs offline: no Ollama, no files in data/ or models/."""

import pandas as pd
import pytest

import pipeline
from data_generator import simulate
from features import FEATURE_COLUMNS, build_features
from model_trainer import fit_model, train_mask

HISTORY = 700          # readings of history passed to the pipeline (just over a week)


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    # 20 cells x 14 days: big enough for every scenario to appear, fast enough for CI
    return simulate(n_cells=20, n_timestamps=1344, seed=7)


@pytest.fixture(scope="session")
def features(dataset) -> pd.DataFrame:
    return build_features(dataset)


@pytest.fixture(scope="session")
def model_and_scaler(dataset, features):
    mask = train_mask(dataset)
    return fit_model(features.loc[mask, FEATURE_COLUMNS].to_numpy())


@pytest.fixture
def offline_pipeline(monkeypatch, model_and_scaler):
    """Pipeline with the test model and the LLM switched off."""
    monkeypatch.setattr(pipeline, "get_model", lambda: model_and_scaler)
    monkeypatch.setattr(pipeline, "call_llm", lambda prompt, system="": None)
    return pipeline


@pytest.fixture
def fake_llm(monkeypatch, model_and_scaler):
    """
    Pipeline with a scripted LLM. Usage:
        fake_llm(classifier='{"failure_mode": ...}', planner='{...}')
    """
    monkeypatch.setattr(pipeline, "get_model", lambda: model_and_scaler)

    def install(classifier=None, planner=None):
        def _call(prompt, system=""):
            return classifier if prompt.startswith("Classify") else planner
        monkeypatch.setattr(pipeline, "call_llm", _call)
        return pipeline

    return install


def window_ending_at(df: pd.DataFrame, row_index: int, n: int = HISTORY) -> pd.DataFrame:
    """The last n readings of a cell, ending at (and including) the given row."""
    cell = df.loc[row_index, "cell_id"]
    return df[df["cell_id"] == cell].loc[:row_index].tail(n)


def mid_event_rows(df: pd.DataFrame, scenario: str, severe: bool = True, limit: int = 40) -> list[int]:
    """Middle reading of each fault event of a scenario that has enough history before it."""
    faults = df[(df["scenario"] == scenario)]
    faults = faults[faults["severity"] >= 0.55] if severe else faults[faults["severity"] < 0.55]
    rows = []
    for _, g in faults.groupby("event_id", sort=False):
        i = g.index[len(g) // 2]
        if len(window_ending_at(df, i)) >= 300:
            rows.append(i)
    return rows[:limit]


def quiet_rows(df: pd.DataFrame, limit: int = 200) -> list[int]:
    """Normal readings on the held-out days, at least 2 hours away from any event."""
    busy = (df["event_id"] != "")
    near = busy.groupby(df["cell_id"]).transform(
        lambda s: s.rolling(17, center=True, min_periods=1).max()).astype(bool)
    held_out = ~train_mask(df)
    return df.index[held_out & ~near].tolist()[:: 7][:limit]
