"""
config.py - NetSentinel central configuration
=============================================
KPI definitions follow 3GPP TS 28.552 (5G performance measurements).
LLM settings can be overridden with environment variables so the same
code runs locally, in CI (no LLM) and against a remote Ollama host.
"""

import os

# ── LLM (Ollama) ──────────────────────────────────────────────────────────
OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
LLM_TIMEOUT_S  = float(os.getenv("NETSENTINEL_LLM_TIMEOUT", "60"))

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR     = "data"
DATA_FILE    = "data/ran_kpi_dataset.csv"
MODEL_PATH   = "models/isolation_forest.pkl"
SCALER_PATH  = "models/scaler.pkl"
METRICS_PATH = "models/metrics.json"

# ── KPI definitions (3GPP TS 28.552) ──────────────────────────────────────
# kpi: (description, unit, nominal_mean, nominal_std, anomaly_direction)
# anomaly_direction: "high" = high values are bad, "low" = low values are bad
KPIS = {
    "rsrp":     ("Reference Signal Received Power",    "dBm",  -75.0,  8.0, "low"),
    "rsrq":     ("Reference Signal Received Quality",  "dB",    -8.0,  3.0, "low"),
    "sinr":     ("Signal to Interference+Noise Ratio", "dB",    15.0,  5.0, "low"),
    "prb_dl":   ("Downlink PRB Utilisation",           "%",     45.0, 15.0, "high"),
    "prb_ul":   ("Uplink PRB Utilisation",             "%",     35.0, 12.0, "high"),
    "cqi":      ("Channel Quality Indicator",          "",       9.0,  2.0, "low"),
    "ho_sr":    ("Handover Success Rate",              "%",     98.5,  1.0, "low"),
    "rlf_rate": ("Radio Link Failure Rate",            "%",      0.5,  0.3, "high"),
    "tp_dl":    ("Downlink Throughput",                "Mbps", 150.0, 40.0, "low"),
    "tp_ul":    ("Uplink Throughput",                  "Mbps",  50.0, 15.0, "low"),
}

# ── Failure scenarios injected during data generation ─────────────────────
ANOMALY_SCENARIOS = {
    "interference":      "High interference from neighbouring cell or external source",
    "capacity_overload": "Cell capacity overload, too many active UEs",
    "hardware_fault":    "Hardware degradation, antenna or baseband unit issue",
    "coverage_gap":      "Coverage hole, UEs at cell edge with weak signal",
    "handover_storm":    "Handover ping-pong, excessive unnecessary handovers",
}

# ── Severity thresholds ───────────────────────────────────────────────────
# Applied to IsolationForest.decision_function() on the LATEST reading.
# decision_function < 0 means the model predicts an anomaly.
# Normal readings sit around +0.1, injected faults around -0.08.
ANOMALY_SCORE_WARNING  = 0.0     # predicted anomalous
ANOMALY_SCORE_CRITICAL = -0.05   # clearly anomalous

# ── KPI monitor ───────────────────────────────────────────────────────────
ZSCORE_FLAG = 2.0          # flag a KPI when the latest reading is > 2 sigma off baseline

# ── Simulation ────────────────────────────────────────────────────────────
N_CELLS        = 100
N_TIMESTAMPS   = 1344    # readings per cell (15-min intervals, 14 days)
READINGS_PER_DAY = 96
ANOMALY_RATE   = 0.06    # target share of readings inside a fault event
BENIGN_SHARE   = 0.3     # share of events that are harmless load surges (not faults)
SUBTLE_SHARE   = 0.4     # share of faults with low severity
EVENT_MIN_LEN  = 4       # readings (1 hour)
EVENT_MAX_LEN  = 48      # readings (12 hours)
RANDOM_SEED    = 42

# ── Features ──────────────────────────────────────────────────────────────
# Baseline for each reading = median of the same time slot (+/- 15 min) on
# previous days of the same type (weekday or weekend) within the last week.
# Falls back to the median of the previous 12 readings.
BASELINE_DAYS   = 7
FALLBACK_WINDOW = 12
RESIDUAL_SMOOTH = 4      # rolling mean of residuals over 1 hour

# ── Model ─────────────────────────────────────────────────────────────────
# Contamination is an alarm budget (share of readings we accept flagging),
# not something tuned on the labels.
IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS  = 300
IF_MAX_SAMPLES   = 256
IF_MAX_FEATURES  = 0.7
TRAIN_SPLIT_DAYS = 10    # train on the first 10 days, evaluate on the last 4

# ── Alarms (how readings become operator alarms) ──────────────────────────
ALARM_K = 2              # raise an alarm when K of the last M readings are flagged
ALARM_M = 3
ALARM_GRACE = 4          # alarms up to 1 hour after a fault ends are not false alarms

PRECISION_TARGET = 0.80
RECALL_TARGET    = 0.70
F1_TARGET        = 0.75
