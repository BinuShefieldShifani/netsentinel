"""
data_generator.py - Synthetic 5G RAN KPI simulator
==================================================
Operator KPI data is commercially sensitive and not publicly available with
the full KPI set used here, so the KPIs are simulated. The goal is not to
copy a real network but to keep the structure that makes detection hard:

1. KPIs depend on each other.
   Each cell has three hidden drivers per reading: traffic load, interference
   and received signal power. The KPIs are derived from them:
       signal power + interference -> SINR -> CQI
       load and CQI                -> PRB utilisation
       CQI and PRB utilisation     -> user throughput
       load, SINR, signal power    -> RSRQ, handover success, radio link failures
   So in normal data SINR, CQI and throughput rise and fall together.

2. Traffic follows a daily pattern.
   Morning and evening peaks on weekdays, a later and flatter curve at
   weekends, plus slow random variation.

3. Faults are events, not single readings.
   A fault ramps up, lasts 1 to 12 hours, then recovers. About 40% are
   subtle. Faults act on the hidden drivers, so their effect spreads to the
   dependent KPIs the way it would in a real cell.

4. Not every unusual event is a fault.
   Some events are harmless load surges (a concert, a football match).
   They are not labelled as anomalies, and a good detector should not alarm
   on them.

Labels per reading: is_anomaly, scenario, event_id, severity, benign_event.
The model never sees these columns; they are only used for evaluation.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import (
    ANOMALY_RATE,
    ANOMALY_SCENARIOS,
    BENIGN_SHARE,
    DATA_FILE,
    EVENT_MAX_LEN,
    EVENT_MIN_LEN,
    KPIS,
    N_CELLS,
    N_TIMESTAMPS,
    RANDOM_SEED,
    READINGS_PER_DAY,
    SUBTLE_SHARE,
)

KPI_FEATURES = list(KPIS.keys())
START_TIME = datetime(2024, 1, 1)          # a Monday

CELL_TYPES = {
    #               mean RSRP  mean load  interference  probability
    "urban_dense": dict(rsrp=-80.0, load=0.85, interf=4.0, p=0.2),
    "urban":       dict(rsrp=-76.0, load=0.70, interf=3.0, p=0.4),
    "suburban":    dict(rsrp=-72.0, load=0.50, interf=2.0, p=0.3),
    "rural":       dict(rsrp=-86.0, load=0.35, interf=1.0, p=0.1),
}

# Effect of each fault on the hidden drivers at full severity.
FAULT_EFFECTS = {
    "interference":      dict(interf=14.0, rsrq=-3.0),
    "capacity_overload": dict(load=0.7, ho=-2.0),
    "hardware_fault":    dict(rsrp=-14.0, rsrq=-2.5, tp_ul_factor=-0.5, rlf=1.0),
    "coverage_gap":      dict(rsrp=-12.0, ho=-5.0, rlf=1.2),
    "handover_storm":    dict(ho=-10.0, rlf=1.5, prb=10.0, interf=2.0),
}


# ── Building blocks ───────────────────────────────────────────────────────

def _bump(h, centre, width):
    return np.exp(-0.5 * ((h - centre) / width) ** 2)


def daily_profile(hour: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """Relative traffic (about 0.15 at night, about 1.0 at the evening peak)."""
    weekday = 0.15 + 0.55 * _bump(hour, 9.5, 2.0) + 0.45 * _bump(hour, 13, 2.5) + 0.75 * _bump(hour, 19.5, 2.5)
    weekend_ = 0.15 + 0.35 * _bump(hour, 12, 3.0) + 0.70 * _bump(hour, 20, 3.0)
    return np.where(weekend, weekend_, weekday)


def ar1(rng, n: int, phi: float, std: float) -> np.ndarray:
    """Slowly varying noise: each value is phi * previous + fresh noise."""
    out = np.empty(n)
    out[0] = rng.normal(0, std / np.sqrt(1 - phi ** 2))
    eps = rng.normal(0, std, n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def event_shape(length: int, rng) -> np.ndarray:
    """Ramp up, plateau, ramp down. Values in (0, 1]."""
    ramp = int(min(4, max(1, length // 3)))
    up = np.arange(1, ramp + 1) / ramp
    down = up[::-1]
    plateau = np.ones(max(length - 2 * ramp, 0))
    shape = np.concatenate([up, plateau, down])[:length]
    return shape


def schedule_events(rng, n: int, anomaly_rate: float) -> list[dict]:
    """Place fault and benign events along one cell's timeline."""
    mean_len = (EVENT_MIN_LEN + EVENT_MAX_LEN) / 2
    mean_gap = max((1 - BENIGN_SHARE) * mean_len / anomaly_rate - mean_len, 8)
    events, pos = [], int(rng.integers(0, int(mean_gap)))
    while pos < n:
        length = int(rng.integers(EVENT_MIN_LEN, EVENT_MAX_LEN + 1))
        if rng.random() < BENIGN_SHARE:
            ev = dict(kind="benign", scenario="normal", severity=float(rng.uniform(0.2, 0.35)))
        else:
            subtle = rng.random() < SUBTLE_SHARE
            sev = rng.uniform(0.25, 0.5) if subtle else rng.uniform(0.6, 1.0)
            ev = dict(kind="fault", scenario=str(rng.choice(list(ANOMALY_SCENARIOS))), severity=float(sev))
        ev.update(start=pos, end=min(pos + length, n))
        events.append(ev)
        pos = ev["end"] + 8 + int(rng.exponential(mean_gap))
    return events


# ── One cell ──────────────────────────────────────────────────────────────

def simulate_cell(rng, cell_id: str, cell_type: str, n: int, anomaly_rate: float,
                  event_counter: list) -> pd.DataFrame:
    p = CELL_TYPES[cell_type]
    t = np.arange(n)
    hour = (t % READINGS_PER_DAY) / (READINGS_PER_DAY / 24)
    weekend = ((t // READINGS_PER_DAY) % 7) >= 5
    profile = daily_profile(hour, weekend)

    # Hidden drivers
    load = p["load"] * rng.uniform(0.85, 1.15) * profile + ar1(rng, n, 0.9, 0.04)
    neighbour_load = p["load"] * profile + ar1(rng, n, 0.95, 0.03)
    rsrp_base = p["rsrp"] + rng.normal(0, 3)
    rsrp_drift = ar1(rng, n, 0.995, 0.15)

    # Fault and benign event effects on the drivers
    eff = {k: np.zeros(n) for k in ["load", "interf", "rsrp", "rsrq", "ho", "rlf", "prb"]}
    tp_ul_factor = np.ones(n)
    is_anomaly = np.zeros(n, dtype=int)
    benign = np.zeros(n, dtype=int)
    scenario = np.array(["normal"] * n, dtype=object)
    event_id = np.array([""] * n, dtype=object)
    severity = np.zeros(n)

    for ev in schedule_events(rng, n, anomaly_rate):
        s, e = ev["start"], ev["end"]
        a = event_shape(e - s, rng) * ev["severity"]
        event_counter[0] += 1
        event_id[s:e] = f"EV{event_counter[0]:05d}"
        if ev["kind"] == "benign":
            eff["load"][s:e] += a
            benign[s:e] = 1
            continue
        for k, v in FAULT_EFFECTS[ev["scenario"]].items():
            if k == "tp_ul_factor":
                tp_ul_factor[s:e] *= 1 + v * a
            else:
                eff[k][s:e] += v * a
        is_anomaly[s:e] = 1
        scenario[s:e] = ev["scenario"]
        severity[s:e] = ev["severity"]

    # Derived KPIs
    load_eff = np.clip(load + eff["load"], 0.02, 1.5)
    rsrp = rsrp_base + rsrp_drift + rng.normal(0, 1.5, n) + eff["rsrp"]
    interf = p["interf"] + 5 * np.clip(neighbour_load, 0, 1.3) + rng.normal(0, 1.0, n) + eff["interf"]
    sinr = np.clip(18 + 0.9 * (rsrp - rsrp_base) - interf + rng.normal(0, 1.2, n), -20, 30)
    rsrq = np.clip(-5.5 - 4.5 * np.minimum(load_eff, 1) - 0.12 * interf + rng.normal(0, 0.6, n) + eff["rsrq"],
                   -19.5, -3)
    cqi = np.clip(np.round(1 + (sinr + 6) / 2.4 + rng.normal(0, 0.4, n)), 1, 15)
    prb_dl = np.clip(80 * load_eff * np.sqrt(9 / cqi) + eff["prb"] + rng.normal(0, 2.5, n), 0, 100)
    prb_ul = np.clip(0.7 * prb_dl + 0.5 * eff["prb"] + rng.normal(0, 3, n), 0, 100)
    tp_dl = np.maximum(30 * cqi * (1 - 0.75 * prb_dl / 100) * (1 + rng.normal(0, 0.06, n)), 0.5)
    tp_ul = np.maximum(10 * cqi * (1 - 0.7 * prb_ul / 100) * tp_ul_factor * (1 + rng.normal(0, 0.08, n)), 0.2)
    ho_sr = np.clip(99.3 - 0.8 * np.minimum(load_eff, 1) - 0.05 * np.maximum(0, -95 - rsrp)
                    + rng.normal(0, 0.25, n) + eff["ho"], 0, 100)
    rlf = np.clip(0.15 + 0.04 * np.maximum(0, 12 - sinr) + 0.25 * np.minimum(load_eff, 1.2)
                  + np.abs(rng.normal(0, 0.05, n)) + eff["rlf"], 0, 10)

    timestamps = [(START_TIME + timedelta(minutes=15 * int(i))).isoformat() for i in t]
    return pd.DataFrame({
        "cell_id": cell_id, "cell_type": cell_type, "timestamp": timestamps,
        "rsrp": rsrp.round(3), "rsrq": rsrq.round(3), "sinr": sinr.round(3),
        "prb_dl": prb_dl.round(3), "prb_ul": prb_ul.round(3), "cqi": cqi.astype(int),
        "ho_sr": ho_sr.round(3), "rlf_rate": rlf.round(3),
        "tp_dl": tp_dl.round(3), "tp_ul": tp_ul.round(3),
        "is_anomaly": is_anomaly, "scenario": scenario, "event_id": event_id,
        "severity": severity.round(3), "benign_event": benign,
    })


# ── Public API ────────────────────────────────────────────────────────────

def simulate(n_cells: int = N_CELLS, n_timestamps: int = N_TIMESTAMPS,
             anomaly_rate: float = ANOMALY_RATE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Simulate the KPI dataset in memory. Sorted by cell, then time."""
    rng = np.random.default_rng(seed)
    types = list(CELL_TYPES)
    cell_types = rng.choice(types, size=n_cells, p=[CELL_TYPES[t]["p"] for t in types])
    counter = [0]
    frames = [simulate_cell(rng, f"CELL_{i+1:04d}", str(cell_types[i]), n_timestamps, anomaly_rate, counter)
              for i in range(n_cells)]
    return pd.concat(frames, ignore_index=True)


def generate_dataset(force: bool = False, path: str = DATA_FILE) -> pd.DataFrame:
    """Load the dataset from disk, or simulate and save it if it does not exist."""
    if os.path.exists(path) and not force:
        print(f"Loading existing dataset from {path} ...")
        df = pd.read_csv(path, keep_default_na=False)
        print(f"  {len(df):,} rows, {df['cell_id'].nunique()} cells loaded.")
        return df

    print("Generating synthetic 5G RAN KPI dataset ...")
    print(f"  {N_CELLS} cells x {N_TIMESTAMPS} readings = {N_CELLS*N_TIMESTAMPS:,} rows")
    df = simulate()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    faults = df[df["is_anomaly"] == 1]
    print(f"  Saved to {path}")
    print(f"  Readings inside faults: {df['is_anomaly'].mean():.1%}  "
          f"({faults['event_id'].nunique()} fault events)")
    print(f"  Benign load surges: {df.loc[df['benign_event'] == 1, 'event_id'].nunique()}")
    return df


if __name__ == "__main__":
    data = generate_dataset(force=True)
    print(data.head(3).to_string())
