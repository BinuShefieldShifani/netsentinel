"""
model_trainer.py — Isolation Forest Anomaly Detector
======================================================
Why Isolation Forest?
  - Unsupervised: in real O-RAN deployments you rarely have clean anomaly labels
  - Fast to train and predict — suitable for near-real-time RAN monitoring
  - Standard in telecoms research for KPI anomaly detection
  - Interpretable: anomaly score is a continuous value, not just 0/1

What counts as a GOOD result on this synthetic dataset:
  ┌─────────────────────────────────────────────┐
  │ Precision > 0.80 → GOOD  (few false alarms) │
  │ Recall    > 0.70 → GOOD  (catches anomalies)│
  │ F1        > 0.75 → GOOD  (balanced)         │
  │ F1        > 0.85 → EXCELLENT                │
  └─────────────────────────────────────────────┘

  For a NOC team: Precision matters MORE than Recall.
  A false alarm (paging an engineer for a normal cell) is expensive.
  Missing an anomaly is also bad, but slightly more tolerable.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from config import (
    MODEL_PATH, SCALER_PATH,
    IF_CONTAMINATION, IF_N_ESTIMATORS,
    PRECISION_TARGET, F1_TARGET
)
from data_generator import generate_dataset, engineer_features, get_feature_columns, KPI_FEATURES


def train():
    print("=" * 58)
    print("  NetSentinel — Isolation Forest Training")
    print("  Domain: 5G RAN KPI Anomaly Detection")
    print("=" * 58)

    # ── Load and engineer features ────────────────────────────────────────
    df = generate_dataset()
    print("\nEngineering rolling features ...")
    df = engineer_features(df)

    feat_cols = KPI_FEATURES   # 10 base KPIs only — rolling features dilute signal in IF
    X = df[feat_cols].values
    y = df["is_anomaly"].values

    print(f"Feature matrix : {X.shape}")
    print(f"Anomaly rate   : {y.mean()*100:.1f}%  ({y.sum():,} anomalous rows)")

    # ── Scale ─────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train Isolation Forest ────────────────────────────────────────────
    print(f"\nTraining Isolation Forest "
          f"(n_estimators={IF_N_ESTIMATORS}, contamination={IF_CONTAMINATION}) ...")

    model = IsolationForest(
        n_estimators=300,
        contamination=IF_CONTAMINATION,
        max_samples=256,          # smaller samples → more diverse trees → better anomaly separation
        max_features=0.7,         # use 70% of features per tree → reduces noise feature dominance
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # ── Evaluate ──────────────────────────────────────────────────────────
    # IsolationForest: predict() returns -1 (anomaly) or 1 (normal)
    raw_preds = model.predict(X_scaled)
    y_pred    = (raw_preds == -1).astype(int)

    precision = precision_score(y, y_pred, zero_division=0)
    recall    = recall_score(y, y_pred, zero_division=0)
    f1        = f1_score(y, y_pred, zero_division=0)

    print("\n" + "─" * 44)
    print(f"  Precision : {precision:.3f}  (target > {PRECISION_TARGET})")
    print(f"  Recall    : {recall:.3f}  (target > 0.70)")
    print(f"  F1 Score  : {f1:.3f}  (target > {F1_TARGET})")
    print("─" * 44)

    if f1 >= 0.85:
        verdict = "EXCELLENT"
    elif f1 >= F1_TARGET:
        verdict = "GOOD — within target"
    elif f1 >= 0.60:
        verdict = "ACCEPTABLE"
    else:
        verdict = "BELOW TARGET"
    print(f"  Verdict   : {verdict}")
    print("─" * 44)

    # Per-scenario breakdown
    print("\n  Anomaly detection by scenario:")
    print(f"  {'Scenario':<22} | {'Count':>6} | {'Detected':>8} | {'Rate':>6}")
    print("  " + "-" * 50)
    for scenario in df["scenario"].unique():
        if scenario == "normal":
            continue
        mask     = df["scenario"] == scenario
        n_total  = mask.sum()
        n_caught = (y_pred[mask] == 1).sum()
        rate     = n_caught / n_total if n_total > 0 else 0
        print(f"  {scenario:<22} | {n_total:>6} | {n_caught:>8} | {rate:>5.1%}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\n  Confusion matrix:")
    print(f"  TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"  FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
    print(f"\n  False alarm rate: {cm[0,1]/(cm[0,0]+cm[0,1]):.1%}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel  saved → {MODEL_PATH}")
    print(f"Scaler saved → {SCALER_PATH}")

    return model, scaler, f1


def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No model at '{MODEL_PATH}'. Run:  python model_trainer.py"
        )
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def score_cell_window(cell_df, model, scaler) -> tuple[float, int]:
    from data_generator import KPI_FEATURES
    X          = cell_df[KPI_FEATURES].values
    X_scaled   = scaler.transform(X)
    scores     = model.score_samples(X_scaled)
    mean_score = float(scores.mean())
    is_anom    = int((model.predict(X_scaled[-1:]) == -1).item())
    return mean_score, is_anom


if __name__ == "__main__":
    train()