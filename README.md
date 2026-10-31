# NetSentinel — 5G RAN Anomaly Detection & Self-Healing Agent

**Autonomous multi-agent system for 5G O-RAN network operations.**  
Isolation Forest · LangGraph · Local LLM · Streamlit · Python

> Targets Ericsson Cognitive Software, Nokia SON teams, and telecom operators running O-RAN deployments.

---

## What It Does

A 5G Radio Access Network generates millions of KPI measurements every minute.
When a cell degrades — due to interference, hardware faults, capacity overload,
or coverage issues — engineers need to diagnose and remediate within minutes to
avoid dropped calls and SLA breaches.

**NetSentinel monitors 50 simulated 5G cells in real time, detects anomalies,
classifies the failure mode, and generates a 3GPP-parameter remediation plan autonomously.**

---

## Agent Pipeline

```
[KPI Monitor]              ← z-score analysis across 10 3GPP KPIs
       ↓
[Anomaly Scorer]           ← Isolation Forest anomaly score
       ↓
   (conditional)
       ├── anomalous → [Root Cause Classifier]  ← LLM classifies failure mode
       │                        ↓
       │               [Remediation Planner]    ← LLM + 3GPP KB → action plan
       │
       └── normal    → [Routine Monitor]        ← monitoring summary
```

**What makes it agentic:** the confidence router — HIGH confidence + auto-applicable procedure → action plan flagged for immediate execution. LOW confidence or manual procedure required → escalates to human operator with full diagnostic. Different path, different outcome.

---

## KPIs Monitored (3GPP TS 28.552)

| KPI | Description | Unit | Anomaly Direction |
|-----|-------------|------|-------------------|
| RSRP | Reference Signal Received Power | dBm | Low = bad |
| RSRQ | Reference Signal Received Quality | dB | Low = bad |
| SINR | Signal to Interference+Noise Ratio | dB | Low = bad |
| PRB DL | Downlink PRB Utilisation | % | High = bad |
| PRB UL | Uplink PRB Utilisation | % | High = bad |
| CQI | Channel Quality Indicator | — | Low = bad |
| HO-SR | Handover Success Rate | % | Low = bad |
| RLF Rate | Radio Link Failure Rate | % | High = bad |
| TP DL | Downlink Throughput | Mbps | Low = bad |
| TP UL | Uplink Throughput | Mbps | Low = bad |

---

## Failure Scenarios Detected

| Scenario | Key KPI Signature |
|----------|-------------------|
| Interference | SINR ↓, RSRQ ↓, CQI ↓, RLF ↑ |
| Capacity Overload | PRB DL/UL → 100%, TP ↓ |
| Hardware Fault | RSRP ↓↓, all signal quality ↓ |
| Coverage Gap | RSRP < -100 dBm, HO-SR ↓, RLF ↑ |
| Handover Storm | HO-SR ↓↓, RLF ↑, PRB ↑ |

---

## Data

Synthetic 5G RAN KPI dataset generated from 3GPP TS 28.552 distributions.

| Property | Value |
|----------|-------|
| Cells | 50 (urban dense / urban / suburban / rural) |
| Timestamps | 1,000 per cell (15-min intervals ≈ 10 days) |
| Total rows | 50,000 |
| Anomaly rate | 8% (realistic for live network) |
| Generation | Fully reproducible — `python model_trainer.py` |

Real operator KPI data is proprietary. Synthetic generation based on published
3GPP KPI distributions is standard practice in O-RAN research
(RANGAN, SpotLight, O-RAN SC xApp benchmarks).

---

## Model Performance

| Metric | Result | Target |
|--------|--------|--------|
| Precision | > 0.80 | > 0.80 |
| Recall | > 0.70 | > 0.70 |
| F1 Score | > 0.75 | > 0.75 |

Precision is prioritised over Recall: a false alarm (dispatching an engineer
to a healthy cell) costs more than briefly missing a transient anomaly.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data + train model (creates data/ and models/ folders)
python model_trainer.py

# 3. Launch dashboard
streamlit run app.py
```

**LLM backend (Ollama in Docker):**
```bash
docker run -d --gpus all -p 11434:11434 ollama/ollama
docker exec -it <container> ollama pull <your-model>
# Set OLLAMA_HOST and OLLAMA_MODEL in config.py
```

The Isolation Forest anomaly detection works fully offline.
Only root cause classification and remediation planning require the LLM.

---

## Key Engineering Decisions

**Why Isolation Forest (not a supervised classifier)?**  
Real O-RAN deployments rarely have clean anomaly labels. Isolation Forest is
unsupervised — it learns the structure of normal operation and flags deviations.
This matches how Ericsson's NOC tools actually work.

**Why synthetic data?**  
5G operator KPI data is commercially sensitive and not publicly available.
Synthetic generation from 3GPP-defined distributions is the approach used in
published O-RAN research. The anomaly signatures are physically grounded —
each scenario degrades the specific KPIs that would degrade in a real network.

**Why rolling features (3, 6, 12 readings)?**  
A single noisy reading is uninformative. Sustained degradation over 90+ minutes
is a real fault. Rolling mean and std capture the trend, not just the snapshot.

**The conditional edge in LangGraph**  
`apply_immediately: true` only when confidence is HIGH and the procedure is
auto-applicable. Hardware faults always require a field engineer — the agent
never auto-applies those. This is responsible AI design in a safety-critical domain.

---

## References

- 3GPP TS 28.552 — 5G performance measurements
- 3GPP TS 38.300 — NR overall description
- RANGAN: GAN-empowered Anomaly Detection in 5G Cloud RAN (arXiv 2025)
- O-RAN Alliance: O-RAN Use Cases and Deployment Scenarios
