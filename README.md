# NetSentinel: 5G RAN anomaly detection with a guarded LLM agent

![CI](https://github.com/BinuShefieldShifani/netsentinel/actions/workflows/ci.yml/badge.svg)

NetSentinel watches KPI readings from simulated 5G cells, detects when a cell
starts misbehaving, works out the likely failure mode and proposes a
remediation plan. An LLM helps with the diagnosis, but it never gets the last
word on what happens to the network.

Isolation Forest · LangGraph · Ollama · Streamlit · pytest · GitHub Actions

## Pipeline

```
[kpi_monitor]        how unusual is the latest reading for this cell at this time of day?
      |
[anomaly_scorer]     Isolation Forest score of the latest reading
      |
  (router)
      |-- anomalous -> [root_cause_classifier]   LLM, validated, with rule-based fallback
      |                         |
      |                [remediation_planner]     plan built in code, LLM writes text only
      |
      '-- normal    -> [routine_monitor]
```

## How the pipeline avoids trusting the LLM blindly

The LLM is useful for explaining a fault in plain language. It is not trusted
to make decisions about a live cell. Concretely:

1. **Its output is validated.** The failure mode must be one of five known
   values and the confidence must be HIGH, MEDIUM or LOW. Bad JSON, an
   invented failure mode, or an unreachable LLM all lead to the same place: a
   rule-based classifier takes over and the confidence is set to LOW.
2. **It can't touch the decision fields.** `apply_immediately`,
   `escalate_to_human`, the actions and the parameters are always set by
   code. The LLM may only fill in `technical_rationale` and
   `expected_outcome`. There is a test where the LLM tries to switch on
   auto-apply and replace the actions; both are ignored.
3. **Auto-apply needs two conditions.** A plan is applied without a human
   only if the confidence is HIGH *and* the procedure is marked
   auto-applicable. Hardware faults are never auto-applicable, whatever the
   LLM says. Everything else is escalated.
4. **It works without the LLM.** With Ollama switched off, the pipeline still
   detects, classifies and escalates. CI runs the whole suite this way.

See `tests/test_pipeline.py` for the tests behind each point.

## KPIs (3GPP TS 28.552)

| KPI | Description | Unit | Bad when |
|-----|-------------|------|----------|
| RSRP | Reference Signal Received Power | dBm | low |
| RSRQ | Reference Signal Received Quality | dB | low |
| SINR | Signal to Interference + Noise Ratio | dB | low |
| PRB DL / UL | PRB utilisation | % | high |
| CQI | Channel Quality Indicator | | low |
| HO-SR | Handover Success Rate | % | low |
| RLF rate | Radio Link Failure rate | % | high |
| TP DL / UL | User throughput | Mbps | low |

## Simulated data

Real operator KPI data isn't public, so the data is simulated: 100 cells,
15-minute readings, 14 days. The simulator tries to keep what makes the
problem hard in practice:

- **KPIs depend on each other.** Each cell has three hidden drivers (traffic
  load, interference, received signal power) and the KPIs follow from them.
  Worse SINR means lower CQI, lower CQI means more PRBs and less throughput.
  In normal operation SINR and CQI correlate at about 0.93.
- **Traffic follows the clock.** Morning and evening peaks on weekdays, a
  later and flatter curve at weekends. Evening PRB load is more than twice
  the night-time load.
- **Faults are events.** A fault ramps up, lasts 1 to 12 hours and recovers.
  About 40% are subtle. Faults change the hidden drivers, so the effect
  spreads to the dependent KPIs.
- **Not every surge is a fault.** About 30% of events are harmless load
  surges (think of a football match). They are not labelled as anomalies.

Failure modes: interference, capacity overload, hardware fault, coverage
gap, handover storm. About 6% of readings fall inside a fault.

## Detection

PRB at 80% is normal at 19:00 in a busy city cell and strange at 04:00 in a
rural one. So `features.py` compares each reading with the same time slot on
earlier days of the same type (weekday or weekend) for that cell, and the
model sees both the raw KPIs and the smoothed deviation from that baseline.
Only past readings are used, so this works live.

The feature set was picked on validation days 8 to 10
(`experiments/select_features.py`) before the held-out days were looked at.

## Results

Trained on days 1 to 10, evaluated on days 11 to 14 (91 fault events). An
alarm is raised when 2 of a cell's last 3 readings are flagged.

| Detector | Point F1 | Faults caught | Severe | Subtle | Median delay | False alarms per cell per day | Harmless surges alarmed |
|---|---|---|---|---|---|---|---|
| Static thresholds | 0.47 | 45% | 71% | 13% | 60 min | 0.04 | 12% |
| Isolation Forest, raw KPIs | 0.59 | 77% | 96% | 53% | 45 min | 0.22 | 24% |
| **Isolation Forest + baseline** | **0.69** | **80%** | **96%** | **60%** | 60 min | **0.13** | 24% |

What this shows:

- Fixed thresholds are quiet but miss more than half the faults, almost all
  of the subtle ones.
- Comparing each cell with its own normal behaviour catches as many faults as
  the raw model with roughly 40% fewer false alarms.
- Capacity overload is the hardest fault to catch (68%), because it looks a
  lot like a harmless surge. A quarter of the harmless surges still raise an
  alarm.
- 0.13 false alarms per cell per day sounds small, but across 10,000 cells it
  would be about 1,300 a day. This is a portfolio project, not something an
  operator could deploy as is.
- 91 events is a small sample, so treat differences of a few points as noise.

An earlier version of this project reported F1 0.98. That came from faults
that were huge, lasted a single reading and sat on top of independent noise:
one threshold on RLF rate alone reached F1 0.89. Those numbers were removed
when the simulator was rebuilt.

With the LLM offline, the rule-based fallback names the right failure mode
for about 84% of severe faults and about a third of subtle ones. Most of
the subtle misses come back as "unclassified", which goes to a human.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python model_trainer.py      # generates data/, trains models/, prints the comparison
streamlit run app.py         # dashboard
pytest                       # 71 tests, no LLM needed
```

The LLM is optional. To use one, run Ollama and set:

```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=gpt-oss:20b     # any chat model you have pulled
```

## Tests and CI

`pytest` covers the simulator (ranges, correlations, daily pattern, fault
events), the features (no look-ahead, weekday vs weekend baseline), the
alarm and event metrics (on small hand-made cases), the model (score scale,
severity thresholds, a quality gate that fails CI if the model stops beating
fixed thresholds), the knowledge base, and the agent pipeline end to end with
a mocked LLM: offline, garbage output, invented failure modes, and an LLM
that tries to override the plan.

Several tests are regression tests for bugs found when this repo was first
reviewed: every healthy cell was reported CRITICAL (wrong score scale), every
fault was classified as interference when the LLM was offline, and
single-reading faults were averaged away by the KPI monitor.

GitHub Actions runs lint, tests and a training run on Python 3.11 and 3.12,
and uploads the metrics file from each run.

## Project layout

```
config.py            KPI definitions, thresholds, simulation and model settings
data_generator.py    synthetic KPI simulation
features.py          per-cell time-of-day baseline and deviation features
evaluation.py        alarms, event-level metrics, baseline detectors
model_trainer.py     training, scoring, held-out comparison
experiments/         feature selection on validation days
knowledge_base.py    remediation procedures per failure mode
pipeline.py          LangGraph agent
app.py               Streamlit dashboard
tests/               pytest suite
```

## Known limitations

- Synthetic data only. The simulator's structure is hand-made, so results
  show how detectors compare on this data, not how they would do on a real
  network.
- The model scores single readings (with smoothed history as features); it
  doesn't model sequences.
- Remediation parameter names are illustrative, not from a specific vendor OSS.
- Nothing is actually applied to a network. "Apply" means the plan is
  marked for automatic execution.

## Roadmap

- Measure LLM classification accuracy against the rule-based baseline
- C++ scoring component (CMake, GoogleTest, pybind11)
- Jenkins pipeline alongside GitHub Actions

## References

- 3GPP TS 28.552, 5G performance measurements
- 3GPP TS 38.300, NR overall description
- O-RAN Alliance, use cases and deployment scenarios
