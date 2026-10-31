"""
model_trainer.py - Train and evaluate the anomaly detector
==========================================================
Run:  python model_trainer.py

Trains on the first TRAIN_SPLIT_DAYS days and evaluates on the rest, so all
reported numbers come from days the model never saw. Writes the model, the
scaler and models/metrics.json, which compares the model with two simpler
detectors (see evaluation.py).

Model: Isolation Forest on features.py output: the raw KPIs plus how far each
KPI is from its own cell's normal value at that time of day. It is
unsupervised: the labels are only used for evaluation. The feature set was
chosen on validation days 8-10 (experiments/select_features.py), before the
held-out days were looked at.

Scoring convention used everywhere in this repo:
  decision_function() > 0  -> looks normal
  decision_function() < 0  -> flagged
"""

import json
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import (
    ANOMALY_SCORE_CRITICAL,
    ANOMALY_SCORE_WARNING,
    IF_CONTAMINATION,
    IF_MAX_FEATURES,
    IF_MAX_SAMPLES,
    IF_N_ESTIMATORS,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    TRAIN_SPLIT_DAYS,
)
from data_generator import KPI_FEATURES, generate_dataset
from evaluation import evaluate_flags, fit_static_thresholds, format_table, static_threshold_flags
from features import FEATURE_COLUMNS, build_features

# ── Core functions (imported by the pipeline, the app and the tests) ──────

def fit_model(X: np.ndarray) -> tuple[IsolationForest, StandardScaler]:
    scaler = StandardScaler().fit(X)
    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        max_samples=IF_MAX_SAMPLES,
        max_features=IF_MAX_FEATURES,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(scaler.transform(X))
    return model, scaler


def score_readings(df: pd.DataFrame, model, scaler, features: pd.DataFrame | None = None) -> np.ndarray:
    """
    decision_function score for every row of df (same order). Negative = flagged.
    df needs each cell's history, because features compare against earlier days.
    """
    feats = build_features(df) if features is None else features
    return model.decision_function(scaler.transform(feats[FEATURE_COLUMNS].to_numpy()))


def score_cell_window(cell_df: pd.DataFrame, model, scaler) -> tuple[float, bool]:
    """Score the latest reading of one cell, using the rest of cell_df as history."""
    ordered = cell_df.sort_values("timestamp")
    score = float(score_readings(ordered, model, scaler)[-1])
    return score, score < 0


def severity_from_score(score: float) -> str:
    if score <= ANOMALY_SCORE_CRITICAL:
        return "CRITICAL"
    if score < ANOMALY_SCORE_WARNING:
        return "WARNING"
    return "NORMAL"


def train_mask(df: pd.DataFrame, days: int = TRAIN_SPLIT_DAYS) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"])
    return (ts < ts.min() + timedelta(days=days)).to_numpy()


def time_split(df: pd.DataFrame, days: int = TRAIN_SPLIT_DAYS):
    mask = train_mask(df, days)
    return df[mask], df[~mask]


def load_model_and_scaler(model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model at '{model_path}'. Run:  python model_trainer.py")
    return joblib.load(model_path), joblib.load(scaler_path)


# ── Training and comparison ───────────────────────────────────────────────

def compare_detectors(df: pd.DataFrame, features: pd.DataFrame, is_train: np.ndarray):
    """Fit all three detectors on the training days and evaluate them on the rest."""
    is_test = ~is_train
    results = {}

    limits = fit_static_thresholds(df[is_train])
    results["static_thresholds"] = evaluate_flags(df, static_threshold_flags(df, limits), is_test)

    raw_model, raw_scaler = fit_model(df.loc[is_train, KPI_FEATURES].to_numpy())
    raw_flags = raw_model.decision_function(raw_scaler.transform(df[KPI_FEATURES].to_numpy())) < 0
    results["iforest_raw"] = evaluate_flags(df, raw_flags, is_test)

    model, scaler = fit_model(features.loc[is_train, FEATURE_COLUMNS].to_numpy())
    flags = score_readings(df, model, scaler, features=features) < 0
    results["iforest_baseline"] = evaluate_flags(df, flags, is_test)
    return results, model, scaler


def train(df: pd.DataFrame | None = None, model_path: str = MODEL_PATH,
          scaler_path: str = SCALER_PATH, metrics_path: str = METRICS_PATH) -> dict:
    df = generate_dataset() if df is None else df
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    is_train = train_mask(df)
    print(f"Train rows: {is_train.sum():,}   Held-out rows: {(~is_train).sum():,}")

    print("Building features ...")
    features = build_features(df)

    print("Fitting and evaluating detectors ...")
    results, model, scaler = compare_detectors(df, features, is_train)

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "split": f"train first {TRAIN_SPLIT_DAYS} days, evaluate on the rest",
        "model": "iforest_baseline",
        "detectors": results,
    }
    print("\nHeld-out comparison\n" + format_table(results))
    print("\nEvent recall by scenario (iforest_baseline):")
    for k, v in results["iforest_baseline"]["events"]["recall_by_scenario"].items():
        print(f"  {k:<20} {v:.0%}")

    for p in (model_path, scaler_path, metrics_path):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model, scaler and metrics to {os.path.dirname(model_path) or '.'}/")
    return metrics


if __name__ == "__main__":
    train()
