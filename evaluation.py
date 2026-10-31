"""
evaluation.py - Measure detection the way an operator would
===========================================================
Per-reading precision/recall is easy to compute but says little about what
an on-call engineer experiences. So detectors are also judged on:

  alarms             a cell raises an alarm when ALARM_K of its last ALARM_M
                     readings are flagged (one odd reading is not an alarm)
  event recall       share of fault events that raised at least one alarm
  detection delay    readings (x 15 min) from fault start to first alarm
  false alarms       alarms that start outside any fault (or its 1-hour
                     tail), per cell per day
  benign triggered   share of harmless load surges that raised an alarm

Three detectors are compared on the same held-out days:

  static_thresholds  fixed per-KPI limits (0.5 / 99.5 percentile of the
                     training days), roughly what a basic NOC rule does
  iforest_raw        Isolation Forest on the raw KPI values
  iforest_baseline   Isolation Forest on raw KPIs plus the smoothed deviation
                     from each cell's own time-of-day baseline (the model the
                     pipeline uses)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from config import ALARM_GRACE, ALARM_K, ALARM_M, KPIS, READINGS_PER_DAY

KPI_FEATURES = list(KPIS.keys())
SUBTLE_BELOW = 0.55   # severity under this counts as a subtle fault


# ── Flags -> alarms ───────────────────────────────────────────────────────

def alarms_from_flags(flags: np.ndarray, cell_ids: np.ndarray,
                      k: int = ALARM_K, m: int = ALARM_M) -> np.ndarray:
    """Alarm on a reading when at least k of the last m readings of that cell were flagged."""
    s = pd.Series(flags.astype(int))
    counts = s.groupby(cell_ids, sort=False).transform(lambda x: x.rolling(m, min_periods=1).sum())
    return counts.to_numpy() >= k


def alarm_onsets(alarms: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    prev = pd.Series(alarms).groupby(cell_ids, sort=False).shift(1, fill_value=False).to_numpy()
    return alarms & ~prev


# ── Baseline detector: static thresholds ──────────────────────────────────

def fit_static_thresholds(train_df: pd.DataFrame, q: float = 0.005) -> dict:
    limits = {}
    for kpi in KPI_FEATURES:
        if KPIS[kpi][4] == "low":
            limits[kpi] = ("low", float(train_df[kpi].quantile(q)))
        else:
            limits[kpi] = ("high", float(train_df[kpi].quantile(1 - q)))
    return limits


def static_threshold_flags(df: pd.DataFrame, limits: dict) -> np.ndarray:
    flags = np.zeros(len(df), dtype=bool)
    for kpi, (direction, limit) in limits.items():
        flags |= (df[kpi] < limit).to_numpy() if direction == "low" else (df[kpi] > limit).to_numpy()
    return flags


# ── Metrics ───────────────────────────────────────────────────────────────

def evaluate_flags(df: pd.DataFrame, flags: np.ndarray, eval_mask: np.ndarray) -> dict:
    """
    df must be sorted by cell then time with a 0..n-1 index.
    flags: one bool per row of df. eval_mask: rows that belong to the evaluation period.
    Alarms are computed on the full series so the period boundary doesn't reset them.
    """
    cell_ids = df["cell_id"].to_numpy()
    alarms = alarms_from_flags(flags, cell_ids)
    onsets = alarm_onsets(alarms, cell_ids)
    y = df["is_anomaly"].to_numpy()

    # Point metrics on the evaluation rows
    yt, ft = y[eval_mask], flags[eval_mask]
    out = {
        "point": {
            "precision": round(float(precision_score(yt, ft, zero_division=0)), 3),
            "recall": round(float(recall_score(yt, ft, zero_division=0)), 3),
            "f1": round(float(f1_score(yt, ft, zero_division=0)), 3),
        }
    }

    # Fault events that start in the evaluation period
    cell_end = df.groupby("cell_id", sort=False).cumcount(ascending=False).to_numpy()  # rows left in cell
    near_fault = y.astype(bool).copy()
    events = []
    faults = df[df["is_anomaly"] == 1].groupby("event_id", sort=False)
    for ev_id, rows in faults:
        idx = rows.index.to_numpy()
        start, end = idx[0], idx[-1]
        tail_end = end + min(ALARM_GRACE, cell_end[end])
        near_fault[end:tail_end + 1] = True
        if not eval_mask[start]:
            continue
        hit = np.flatnonzero(alarms[start:end + 1])
        events.append({
            "scenario": rows["scenario"].iloc[0],
            "subtle": bool(rows["severity"].iloc[0] < SUBTLE_BELOW),
            "detected": hit.size > 0,
            "delay": int(hit[0]) if hit.size else None,
        })
    ev = pd.DataFrame(events)

    # Benign load surges in the evaluation period
    benign = df[(df["benign_event"] == 1)].groupby("event_id", sort=False)
    benign_hits = [bool(alarms[r.index].any()) for _, r in benign if eval_mask[r.index[0]]]

    false_alarms = int((onsets & ~near_fault & eval_mask).sum())
    cell_days = eval_mask.sum() / READINGS_PER_DAY

    detected = ev[ev["detected"]] if len(ev) else ev
    out["events"] = {
        "n_fault_events": int(len(ev)),
        "event_recall": round(float(ev["detected"].mean()), 3) if len(ev) else None,
        "event_recall_severe": round(float(ev.loc[~ev["subtle"], "detected"].mean()), 3) if len(ev) else None,
        "event_recall_subtle": round(float(ev.loc[ev["subtle"], "detected"].mean()), 3) if len(ev) else None,
        "median_delay_min": float(detected["delay"].median() * 15) if len(detected) else None,
        "false_alarms_per_cell_day": round(float(false_alarms / cell_days), 3),
        "benign_surges_alarmed": round(float(np.mean(benign_hits)), 3) if benign_hits else None,
        "recall_by_scenario": (ev.groupby("scenario")["detected"].mean().round(3).to_dict() if len(ev) else {}),
    }
    return out


def format_table(results: dict) -> str:
    """Plain-text comparison table of several detectors."""
    cols = [("event recall", "event_recall"), ("severe", "event_recall_severe"),
            ("subtle", "event_recall_subtle"), ("delay min", "median_delay_min"),
            ("false/cell/day", "false_alarms_per_cell_day"), ("benign alarmed", "benign_surges_alarmed")]
    header = f"{'detector':<18} {'point F1':>8} " + " ".join(f"{c:>14}" for c, _ in cols)
    lines = [header, "-" * len(header)]
    for name, r in results.items():
        vals = " ".join(f"{'-' if r['events'][k] is None else r['events'][k]:>14}" for _, k in cols)
        lines.append(f"{name:<18} {r['point']['f1']:>8} {vals}")
    return "\n".join(lines)
