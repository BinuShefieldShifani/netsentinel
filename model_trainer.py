"""
model_trainer.py - Isolation Forest anomaly detector
====================================================
Run:  python model_trainer.py

Generates the dataset (if missing), trains on the first TRAIN_SPLIT_DAYS of
data and evaluates on the remaining days, so the reported numbers come from
readings the model never saw. Metrics are written to models/metrics.json.

Why Isolation Forest?
  - Unsupervised: live networks rarely have clean anomaly labels
  - Fast to train and score
  - Gives a continuous anomaly score, not just 0/1

Scoring convention used everywhere in this repo:
  decision_function() > 0  -> looks normal
  decision_function() < 0  -> predicted anomaly
"""

import json
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from config import (
    ANOMALY_SCORE_CRITICAL,
    ANOMALY_SCORE_WARNING,
    F1_TARGET,
    IF_CONTAMINATION,
    IF_MAX_FEATURES,
    IF_MAX_SAMPLES,
    IF_N_ESTIMATORS,
    METRICS_PATH,
    MODEL_PATH,
    PRECISION_TARGET,
    RANDOM_SEED,
    RECALL_TARGET,
    SCALER_PATH,
    TRAIN_SPLIT_DAYS,
)
from data_generator import KPI_FEATURES, generate_dataset

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


def score_readings(df: pd.DataFrame, model, scaler) -> np.ndarray:
    """decision_function score per reading. Negative = predicted anomaly."""
    return model.decision_function(scaler.transform(df[KPI_FEATURES].values))


def score_cell_window(cell_df: pd.DataFrame, model, scaler) -> tuple[float, bool]:
    """
    Score the LATEST reading of a cell.
    (The earlier version averaged score_samples() over the whole history,
    which used a different scale from the thresholds and marked every cell
    CRITICAL.)
    """
    latest = cell_df.tail(1)
    score = float(score_readings(latest, model, scaler)[0])
    return score, score < 0


def severity_from_score(score: float) -> str:
    if score <= ANOMALY_SCORE_CRITICAL:
        return "CRITICAL"
    if score < ANOMALY_SCORE_WARNING:
        return "WARNING"
    return "NORMAL"


def time_split(df: pd.DataFrame, days: int = TRAIN_SPLIT_DAYS):
    ts = pd.to_datetime(df["timestamp"])
    cutoff = ts.min() + timedelta(days=days)
    return df[ts < cutoff], df[ts >= cutoff]


def evaluate(df: pd.DataFrame, model, scaler) -> dict:
    y = df["is_anomaly"].values
    y_pred = (score_readings(df, model, scaler) < 0).astype(int)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    per_scenario = {}
    for scenario in sorted(df["scenario"].unique()):
        if scenario == "normal":
            continue
        mask = (df["scenario"] == scenario).values
        per_scenario[scenario] = round(float(y_pred[mask].mean()), 3)
    return {
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 3),
        "recall":    round(float(recall_score(y, y_pred, zero_division=0)), 3),
        "f1":        round(float(f1_score(y, y_pred, zero_division=0)), 3),
        "false_alarm_rate": round(float(cm[0, 1] / max(cm[0, 0] + cm[0, 1], 1)), 4),
        "confusion": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                      "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "detection_rate_by_scenario": per_scenario,
        "n_rows": int(len(df)),
    }


def load_model_and_scaler(model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model at '{model_path}'. Run:  python model_trainer.py")
    return joblib.load(model_path), joblib.load(scaler_path)


# ── Training entry point ──────────────────────────────────────────────────

def train(df: pd.DataFrame | None = None, model_path: str = MODEL_PATH,
          scaler_path: str = SCALER_PATH, metrics_path: str = METRICS_PATH) -> dict:
    df = generate_dataset() if df is None else df
    train_df, test_df = time_split(df)

    print(f"Train rows: {len(train_df):,}   Held-out rows: {len(test_df):,}")
    model, scaler = fit_model(train_df[KPI_FEATURES].values)
    metrics = evaluate(test_df, model, scaler)
    metrics["trained_at"] = datetime.now().isoformat(timespec="seconds")
    metrics["split"] = f"train first {TRAIN_SPLIT_DAYS} days, evaluate on the rest"

    print("\nHeld-out results")
    print(f"  Precision : {metrics['precision']:.3f}  (target > {PRECISION_TARGET})")
    print(f"  Recall    : {metrics['recall']:.3f}  (target > {RECALL_TARGET})")
    print(f"  F1        : {metrics['f1']:.3f}  (target > {F1_TARGET})")
    print(f"  False alarm rate: {metrics['false_alarm_rate']:.2%}")
    print("  Detection rate by scenario:")
    for k, v in metrics["detection_rate_by_scenario"].items():
        print(f"    {k:<20} {v:.1%}")

    for p in (model_path, scaler_path, metrics_path):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model, scaler and metrics to {os.path.dirname(model_path)}/")
    return metrics


if __name__ == "__main__":
    train()
