"""
config.py — NetSentinel Central Configuration
===============================================
Targets: Ericsson (RAN Cognitive Software team), Nokia, Telenor Sweden
Domain:  5G O-RAN autonomous network operations

KPI definitions follow 3GPP TS 28.552 (Management and orchestration;
5G performance measurements) — the standard Ericsson engineers work with.
"""

# ── LLM (Ollama in Docker) ────────────────────────────────────────────────
OLLAMA_HOST  = "http://localhost:11434"
OLLAMA_MODEL = "gptoss:20b"

# ── Data ──────────────────────────────────────────────────────────────────
DATA_DIR       = "data"
DATA_FILE      = "data/ran_kpi_dataset.csv"
MODEL_PATH     = "models/isolation_forest.pkl"
SCALER_PATH    = "models/scaler.pkl"

# ── KPI Definitions (3GPP TS 28.552) ─────────────────────────────────────
# Each KPI: (description, unit, nominal_mean, nominal_std, anomaly_direction)
# anomaly_direction: "high" = high values are bad, "low" = low values are bad

KPIS = {
    "rsrp":    ("Reference Signal Received Power",  "dBm",  -75.0,  8.0, "low"),
    "rsrq":    ("Reference Signal Received Quality","dB",   -8.0,   3.0, "low"),
    "sinr":    ("Signal to Interference+Noise Ratio","dB",   15.0,  5.0, "low"),
    "prb_dl":  ("Downlink PRB Utilisation",         "%",    45.0,  15.0, "high"),
    "prb_ul":  ("Uplink PRB Utilisation",           "%",    35.0,  12.0, "high"),
    "cqi":     ("Channel Quality Indicator",        "",     9.0,   2.0,  "low"),
    "ho_sr":   ("Handover Success Rate",            "%",    98.5,  1.0,  "low"),
    "rlf_rate":("Radio Link Failure Rate",          "%",    0.5,   0.3,  "high"),
    "tp_dl":   ("Downlink Throughput",              "Mbps", 150.0, 40.0, "low"),
    "tp_ul":   ("Uplink Throughput",                "Mbps", 50.0,  15.0, "low"),
}

# ── Anomaly Scenarios (injected during data generation) ──────────────────
# These map to real failure modes Ericsson's NOC teams deal with daily
ANOMALY_SCENARIOS = {
    "interference":      "High interference from neighbouring cell or external source",
    "capacity_overload": "Cell capacity overload — too many active UEs",
    "hardware_fault":    "Hardware degradation — antenna or baseband unit issue",
    "coverage_gap":      "Coverage hole — UEs at cell edge with weak signal",
    "handover_storm":    "Handover ping-pong — excessive unnecessary handovers",
}

# ── Alert Thresholds ──────────────────────────────────────────────────────
# Anomaly score from Isolation Forest: higher (closer to 0) = more anomalous
# IsolationForest score range: -1 (anomaly) to ~0.5 (normal)
ANOMALY_SCORE_CRITICAL = -0.15   # definitely anomalous
ANOMALY_SCORE_WARNING  = -0.05   # borderline

# ── Simulation ────────────────────────────────────────────────────────────
N_CELLS          = 50     # number of simulated 5G cells
N_TIMESTAMPS     = 1000   # time steps per cell (15-min intervals → ~10 days)
ANOMALY_RATE     = 0.08   # 8% of readings are anomalous (realistic for live network)
RANDOM_SEED      = 42

# ── Model ─────────────────────────────────────────────────────────────────
# Isolation Forest hyperparameters tuned for RAN KPI data
IF_CONTAMINATION = 0.08   # matches ANOMALY_RATE above
IF_N_ESTIMATORS  = 200
IF_MAX_SAMPLES   = "auto"

# ── What counts as a good result ─────────────────────────────────────────
# On synthetic data with known anomaly labels:
# F1 > 0.75 → Good    (typical for unsupervised on network data)
# F1 > 0.85 → Excellent
# Precision > 0.80 → important: low false alarm rate matters for NOC teams
PRECISION_TARGET = 0.80
F1_TARGET        = 0.75
