"""Shared fixtures. Everything runs offline: no Ollama, no files in data/ or models/."""

import pandas as pd
import pytest

import pipeline
from data_generator import KPI_FEATURES, simulate
from model_trainer import fit_model, time_split


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    # 8 cells x 1000 readings: small enough to be fast, large enough for stable stats
    return simulate(n_cells=8, n_timestamps=1000, seed=7)


@pytest.fixture(scope="session")
def model_and_scaler(dataset):
    train_df, _ = time_split(dataset)
    return fit_model(train_df[KPI_FEATURES].values)


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


def window_ending_at(df: pd.DataFrame, row_index: int, n: int = 60) -> pd.DataFrame:
    """The n readings of a cell that end at (and include) the given row."""
    cell = df.loc[row_index, "cell_id"]
    cell_df = df[df["cell_id"] == cell]
    return cell_df.loc[:row_index].tail(n)


def held_out_rows(df: pd.DataFrame, scenario: str) -> list[int]:
    _, test_df = time_split(df)
    return test_df.index[test_df["scenario"] == scenario].tolist()
