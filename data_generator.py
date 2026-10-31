"""
data_generator.py - Synthetic 5G RAN KPI data generator
=======================================================
Why synthetic data?
  Operator KPI data is commercially sensitive and not publicly available
  with the full KPI set used here, so the KPIs are simulated using the
  3GPP TS 28.552 definitions and plausible nominal ranges.

Known limitation (current version):
  KPIs are drawn independently and each fault lasts a single reading,
  so faults are much easier to spot than in a live network.

What we simulate:
  50 cells × 1000 timestamps = 50,000 rows
  10 KPIs per row (RSRP, RSRQ, SINR, PRB DL/UL, CQI, HO-SR, RLF, TP DL/UL)
  8% of rows have injected anomalies across 5 real failure scenarios

Each cell has a slightly different baseline (urban vs suburban vs rural)
to make the dataset realistic — not all cells have the same nominal values.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import ANOMALY_RATE, ANOMALY_SCENARIOS, DATA_FILE, KPIS, N_CELLS, N_TIMESTAMPS, RANDOM_SEED


def simulate(n_cells: int = N_CELLS, n_timestamps: int = N_TIMESTAMPS,
             anomaly_rate: float = ANOMALY_RATE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Simulate the KPI dataset in memory. Pure function, no file I/O."""
    rng = np.random.default_rng(seed)
    rows = []

    # Cell type distribution (affects baseline KPI values)
    cell_types = rng.choice(
        ["urban_dense", "urban", "suburban", "rural"],
        size=n_cells,
        p=[0.2, 0.4, 0.3, 0.1]
    )

    # Per-cell baseline offsets (simulate different environments)
    cell_offsets = {
        "urban_dense": {"rsrp": -5,  "prb_dl": +15, "prb_ul": +10, "tp_dl": +30},
        "urban":       {"rsrp":  0,  "prb_dl":  0,  "prb_ul":  0,  "tp_dl":  0},
        "suburban":    {"rsrp": +3,  "prb_dl": -10, "prb_ul": -8,  "tp_dl": -20},
        "rural":       {"rsrp": -8,  "prb_dl": -20, "prb_ul": -15, "tp_dl": -40},
    }

    start_time = datetime(2024, 1, 1, 0, 0, 0)

    for cell_idx in range(n_cells):
        cell_id   = f"CELL_{cell_idx+1:04d}"
        cell_type = cell_types[cell_idx]
        offsets   = cell_offsets[cell_type]

        for t in range(n_timestamps):
            timestamp  = start_time + timedelta(minutes=15 * t)
            is_anomaly = rng.random() < anomaly_rate
            scenario   = None

            if is_anomaly:
                scenario = rng.choice(list(ANOMALY_SCENARIOS.keys()))

            row = _generate_row(
                rng, cell_id, cell_type, timestamp,
                offsets, is_anomaly, scenario
            )
            rows.append(row)

    return pd.DataFrame(rows)


def generate_dataset(force: bool = False, path: str = DATA_FILE) -> pd.DataFrame:
    """
    Load the dataset from disk, or simulate and save it if it does not exist.
    """
    if os.path.exists(path) and not force:
        print(f"Loading existing dataset from {path} ...")
        df = pd.read_csv(path)
        print(f"  {len(df):,} rows, {df['cell_id'].nunique()} cells loaded.")
        return df

    print("Generating synthetic 5G RAN KPI dataset ...")
    print(f"  {N_CELLS} cells x {N_TIMESTAMPS} timestamps = {N_CELLS*N_TIMESTAMPS:,} rows")
    df = simulate()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved to {path}")
    print(f"  Anomaly rate: {df['is_anomaly'].mean()*100:.1f}%")
    print(f"  Scenarios: {df[df['is_anomaly'] == 1]['scenario'].value_counts().to_dict()}")
    return df


def _generate_row(rng, cell_id, cell_type, timestamp, offsets,
                  is_anomaly, scenario):
    """Generate one row of KPI values, optionally injecting an anomaly."""

    # ── Nominal values (with cell-type offset) ───────────────────────────
    vals = {}
    for kpi, (desc, unit, mean, std, direction) in KPIS.items():
        offset = offsets.get(kpi, 0)
        vals[kpi] = float(rng.normal(mean + offset, std))

    # ── Time-of-day effect (peak hours: 8am-6pm have higher PRB load) ────
    hour = timestamp.hour
    if 8 <= hour <= 18:
        vals["prb_dl"] += rng.normal(10, 3)
        vals["prb_ul"] += rng.normal(7, 2)
        vals["tp_dl"]  += rng.normal(20, 5)

    # ── Inject anomaly scenario ───────────────────────────────────────────
    if is_anomaly and scenario:
        vals = _inject_anomaly(rng, vals, scenario)

    # ── Physical constraints (clip to valid ranges) ───────────────────────
    vals["rsrp"]     = np.clip(vals["rsrp"],    -130, -44)
    vals["rsrq"]     = np.clip(vals["rsrq"],    -19.5, -3)
    vals["sinr"]     = np.clip(vals["sinr"],    -20,   30)
    vals["prb_dl"]   = np.clip(vals["prb_dl"],  0,    100)
    vals["prb_ul"]   = np.clip(vals["prb_ul"],  0,    100)
    vals["cqi"]      = np.clip(int(vals["cqi"]), 1,    15)
    vals["ho_sr"]    = np.clip(vals["ho_sr"],   0,    100)
    vals["rlf_rate"] = np.clip(vals["rlf_rate"], 0,    10)
    vals["tp_dl"]    = max(vals["tp_dl"],  0)
    vals["tp_ul"]    = max(vals["tp_ul"],  0)

    return {
        "cell_id":   cell_id,
        "cell_type": cell_type,
        "timestamp": timestamp.isoformat(),
        **{k: round(v, 3) for k, v in vals.items()},
        "is_anomaly": int(is_anomaly),
        "scenario":   str(scenario) if scenario else "normal",
    }


def _inject_anomaly(rng, vals: dict, scenario: str) -> dict:
    """
    Each scenario degrades a specific combination of KPIs,
    matching real-world O-RAN failure signatures.
    """
    if scenario == "interference":
        # High interference: SINR collapses, RSRQ drops, CQI drops, throughput drops
        vals["sinr"]     -= rng.uniform(20, 30)
        vals["rsrq"]     -= rng.uniform(7, 12)
        vals["cqi"]      -= rng.uniform(4, 7)
        vals["tp_dl"]    *= rng.uniform(0.1, 0.35)
        vals["rlf_rate"] += rng.uniform(2.0, 5.0)

    elif scenario == "capacity_overload":
        # PRB maxes out hard — key signature is PRB near 100%
        vals["prb_dl"]   = rng.uniform(95, 100)
        vals["prb_ul"]   = rng.uniform(90, 100)
        vals["tp_dl"]    *= rng.uniform(0.2, 0.45)
        vals["tp_ul"]    *= rng.uniform(0.2, 0.45)
        vals["ho_sr"]    -= rng.uniform(3, 8)
        vals["cqi"]      -= rng.uniform(2, 4)

    elif scenario == "hardware_fault":
        # Severe RSRP drop — all metrics collapse
        vals["rsrp"]     -= rng.uniform(25, 40)
        vals["rsrq"]     -= rng.uniform(6, 10)
        vals["sinr"]     -= rng.uniform(12, 20)
        vals["tp_dl"]    *= rng.uniform(0.05, 0.25)
        vals["tp_ul"]    *= rng.uniform(0.05, 0.25)
        vals["rlf_rate"] += rng.uniform(3.0, 7.0)
        vals["ho_sr"]    -= rng.uniform(5, 15)

    elif scenario == "coverage_gap":
        # Very weak signal at cell edge
        vals["rsrp"]     -= rng.uniform(28, 45)
        vals["sinr"]     -= rng.uniform(15, 25)
        vals["cqi"]      -= rng.uniform(5, 8)
        vals["ho_sr"]    -= rng.uniform(8, 20)
        vals["rlf_rate"] += rng.uniform(2.5, 6.0)
        vals["tp_dl"]    *= rng.uniform(0.1, 0.3)

    elif scenario == "handover_storm":
        # Handover success rate collapses, RLF spikes sharply
        vals["ho_sr"]    -= rng.uniform(15, 30)
        vals["rlf_rate"] += rng.uniform(3.0, 8.0)
        vals["prb_dl"]   += rng.uniform(20, 35)
        vals["tp_dl"]    *= rng.uniform(0.3, 0.6)
        vals["sinr"]     -= rng.uniform(5, 10)

    return vals


# ── Feature engineering (not used by the current model, kept for step 3) ─

KPI_FEATURES = list(KPIS.keys())
ROLLING_WINDOWS = [3, 6, 12]   # 15-min intervals → 45min, 90min, 3hr


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean and std features per cell per KPI.
    Captures temporal degradation trends — a single noisy reading
    is less informative than a sustained degradation over 3 hours.
    """
    dfs = []
    for cell_id, group in df.groupby("cell_id"):
        g = group.sort_values("timestamp").copy()
        for kpi in KPI_FEATURES:
            for w in ROLLING_WINDOWS:
                g[f"{kpi}_mean{w}"] = g[kpi].rolling(w, min_periods=1).mean()
                g[f"{kpi}_std{w}"]  = g[kpi].rolling(w, min_periods=1).std().fillna(0)
        dfs.append(g)
    return pd.concat(dfs, ignore_index=True)


def get_feature_columns() -> list[str]:
    base    = KPI_FEATURES
    rolling = [f"{k}_{s}{w}" for k in KPI_FEATURES
               for w in ROLLING_WINDOWS for s in ["mean", "std"]]
    return base + rolling


if __name__ == "__main__":
    df = generate_dataset(force=True)
    print(f"\nDataset shape: {df.shape}")
    print(df.head(3).to_string())
