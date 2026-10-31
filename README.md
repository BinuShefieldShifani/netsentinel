# NetSentinel: 5G RAN anomaly detection with a guarded LLM agent

![CI](https://github.com/BinuShefieldShifani/netsentinel/actions/workflows/ci.yml/badge.svg)

NetSentinel watches KPI readings from simulated 5G cells, detects when a cell
starts misbehaving, works out the likely failure mode and proposes a
remediation plan. An LLM helps with the diagnosis, but it never gets the last
word on what happens to the network.

Isolation Forest · LangGraph · Ollama · Streamlit · pytest · GitHub Actions

## Pipeline

```
[kpi_monitor]        robust z-score of the latest reading vs the cell's own history
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
| TP DL / UL | Throughput | Mbps | low |

Failure modes simulated: interference, capacity overload, hardware fault,
coverage gap, handover storm.

## Results, and why you shouldn't be impressed by them yet

The model is trained on the first 7 days and evaluated on the last 3 days of
50 cells (`models/metrics.json`):

| Metric | Held-out |
|--------|----------|
| Precision | 0.969 |
| Recall | 1.000 |
| F1 | 0.984 |
| False alarm rate | 0.29% |

These numbers say more about the data than the model. On the same dataset a
single threshold, `rlf_rate > 2`, already reaches F1 0.89 with no ML at all.
The reasons:

- injected faults are very large (RLF rate moves about 13 standard deviations)
- each fault lasts exactly one 15-minute reading, while real faults build up
  and persist
- KPIs are generated independently, so in normal data SINR, CQI and
  throughput don't move together the way they do in a real cell

Making the simulation harder and more realistic is the next step (see
Roadmap). Expect the scores to drop when that happens.

With the LLM offline, the rule-based fallback picks the right failure mode in
about 87% of held-out faults, and healthy cells are reported as NORMAL in
about 97% of cases (the rest get a WARNING and go to a human).

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python model_trainer.py      # generates data/ and trains models/
streamlit run app.py         # dashboard
pytest                       # 54 tests, no LLM needed
```

The LLM is optional. To use one, run Ollama and set:

```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=gpt-oss:20b     # any chat model you have pulled
```

## Tests and CI

`pytest` covers the data generator (ranges, reproducibility, fault
signatures), the model (score scale, severity thresholds, held-out metrics),
the knowledge base, and the agent pipeline end to end with a mocked LLM:
offline, garbage output, invented failure modes, and an LLM that tries to
override the plan.

Several tests are regression tests for bugs found when this repo was first
reviewed: every healthy cell was reported CRITICAL (wrong score scale), every
fault was classified as interference when the LLM was offline, and
single-reading faults were averaged away by the KPI monitor.

GitHub Actions runs lint, tests and a training run on Python 3.11 and 3.12,
and uploads the metrics file from each run.

## Project layout

```
config.py            KPI definitions, thresholds, model settings
data_generator.py    synthetic KPI simulation
model_trainer.py     Isolation Forest training, scoring, held-out evaluation
knowledge_base.py    remediation procedures per failure mode
pipeline.py          LangGraph agent
app.py               Streamlit dashboard
tests/               pytest suite
```

## Known limitations

- Synthetic data only; see Results.
- Remediation parameter names are illustrative, not from a specific vendor OSS.
- Nothing is actually applied to a network. "Apply" means the plan is
  marked for automatic execution.

## Roadmap

- More realistic simulation: correlated KPIs, faults that build up over
  several readings, subtle degradations, time-split evaluation on that data
- Measure LLM classification accuracy against the rule-based baseline
- C++ scoring component (CMake, GoogleTest, pybind11)
- Jenkins pipeline alongside GitHub Actions

## References

- 3GPP TS 28.552, 5G performance measurements
- 3GPP TS 38.300, NR overall description
- O-RAN Alliance, use cases and deployment scenarios
