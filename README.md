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
3. **Auto-apply needs three conditions.** A plan is applied without a human
   only if the confidence is HIGH, the procedure is marked auto-applicable,
   *and* the rule-based classifier independently names the same failure
   mode. Only load balancing (capacity overload) and handover tuning
   (handover storm) are auto-applicable, since those match SON functions
   operators commonly automate. Interference, hardware faults and coverage
   gaps need their cause confirmed by a person first, whatever the LLM says.

   The cross-check was added after testing a scripted LLM that always
   answers "handover storm, HIGH confidence". Without it, that LLM would
   have triggered 72 wrong automatic fixes on 86 held-out faults. With it,
   13. Those 13 are cases where the rules make the same mistake.
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

If a cell has no earlier day of the same type yet (its first weekend), the
baseline uses the same time slot on any earlier day. An earlier version used
the last 12 readings instead, so a long fault slowly became its own baseline
and stopped being detected after about three hours. There is a regression
test for that.

The feature set was picked on validation days 8 to 10
(`experiments/select_features.py`) before the held-out days were looked at.

## Results

Trained on days 1 to 10, evaluated on days 11 to 14 (91 fault events). An
alarm is raised when 2 of a cell's last 3 readings are flagged.

| Detector | Point F1 | Faults caught | Severe | Subtle | Median delay | False alarms per cell per day | Harmless surges alarmed |
|---|---|---|---|---|---|---|---|
| Static thresholds | 0.47 | 45% | 71% | 13% | 60 min | 0.04 | 12% |
| Isolation Forest, raw KPIs | 0.59 | 77% | 96% | 53% | 45 min | 0.22 | 24% |
| **Isolation Forest + baseline** | **0.68** | **78%** | **94%** | **58%** | 60 min | **0.11** | **15%** |

What this shows:

- Fixed thresholds are quiet but miss more than half the faults, almost all
  of the subtle ones.
- Comparing each cell with its own normal behaviour catches about as many
  faults as the raw model, with half the false alarms and fewer than two
  thirds of the alarms on harmless surges.
- Capacity overload is the hardest fault to catch (64%), because it looks a
  lot like a harmless surge.
- 0.11 false alarms per cell per day sounds small, but across 10,000 cells it
  would be about 1,100 a day. This is a portfolio project, not something an
  operator could deploy as is.
- 91 events is a small sample, so treat differences of a few points as noise.

An earlier version of this project reported F1 0.98. That came from faults
that were huge, lasted a single reading and sat on top of independent noise:
one threshold on RLF rate alone reached F1 0.89. Those numbers were removed
when the simulator was rebuilt.

With the LLM offline, the rule-based fallback names the right failure mode
for 86% of severe faults and 28% of subtle ones (86 held-out faults, see
`llm_eval.py`). Most of the subtle misses come back as "unclassified", which
goes to a human.

## Measuring the LLM against the rules

`llm_eval.py` takes labelled faults, gives the LLM exactly the classifier
prompt the pipeline uses, and gives the rule-based classifier the same
numbers. It reports accuracy (overall, severe, subtle, per fault type),
valid replies versus errors, whether the LLM's confidence means anything,
who is right when they disagree, latency, and how many automatic fixes would
be applied for the wrong fault.

### First result: the LLM made things worse

Qwen2.5 7B through Ollama on a Kaggle T4, original prompt, 86 held-out faults:

| | LLM | Rules | Pipeline (LLM first) |
|---|---|---|---|
| All faults | 58% | 62% | 57% |
| Severe (50) | 66% | 86% | 66% |
| Subtle (36) | 46% | 28% | 44% |

- The LLM said HIGH confidence on 81 of 85 valid replies and was right 56%
  of the time, so its confidence is not a usable safety signal.
- The pipeline would have applied 30 automatic fixes, 13 for the wrong
  fault. The rules cross-check cut this from 22, but it can't catch
  mistakes both make: subtle capacity overloads and coverage gaps look like
  handover storms to both.
- The LLM is better than the rules on subtle faults, where the rules often
  give up and answer "unclassified".

This comparison favours the rules: they were written by someone who knew
how the simulator creates each fault, while the LLM only got a one-line
description of each mode.

### Giving the LLM a fair chance

The next round tests whether a better setup changes the picture:

- **Prompts** (`--prompt`): `basic` (the prompt above), `signatures` (adds
  what each fault does to the KPIs, the same knowledge the rules use),
  `fewshot` (signatures plus 10 solved examples from days 1 to 7)
- **Models:** Qwen2.5 7B and 14B
- **Decision policies:** `llm_first` (the LLM decides, rules as fallback) and
  `rules_first` (the rules decide, the LLM only when they say
  "unclassified"); the LLM's confidence is kept only when it agrees with
  the rules
- **Auto-apply:** the current three conditions, and the same plus "the
  reading must be CRITICAL"

The held-out run above was used to decide to run these experiments, so the
choice between them is made on **validation days 8 to 10** (54 faults). The
held-out days are run once more at the end with the chosen setup. With 54
faults, differences under about 10 points are noise.

**How the winner is chosen, decided before seeing the results:** first, the
fewest automatic fixes for the wrong fault; then the highest pipeline
accuracy; then the simplest setup (rules over LLM, smaller model, shorter
prompt).

Results: to be added after the validation runs.

```bash
python llm_eval.py --prompt signatures                     # validation days (default)
python -m experiments.compare_llm_reports --split validation
python llm_eval.py --split heldout --prompt <chosen>       # once, at the end
```

The pipeline's settings are environment variables, and the defaults are the
original behaviour until a variant wins on validation:
`NETSENTINEL_PROMPT` (basic | signatures | fewshot), `NETSENTINEL_POLICY`
(llm_first | rules_first), `NETSENTINEL_AUTO_CRITICAL` (0 | 1).

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python model_trainer.py      # generates data/, trains models/, prints the comparison
streamlit run app.py         # dashboard
pytest                       # 83 tests (+14 C++ parity tests when built), no LLM needed
```

The LLM is optional. Pick a backend with `NETSENTINEL_LLM`:

```bash
export NETSENTINEL_LLM=ollama        # default; also set OLLAMA_HOST, OLLAMA_MODEL
export NETSENTINEL_LLM=anthropic     # also set ANTHROPIC_API_KEY, optionally ANTHROPIC_MODEL
export NETSENTINEL_LLM=none          # rules only
```

## Using the dashboard

The sidebar works like a time machine. Pick a cell and a **current reading**:
the agent sees every reading up to that one and nothing after it. **Jump into
a labelled event** sets the current reading to about 2 hours into a fault,
which is the quickest way to watch the agent catch one. **Chart history**
only changes how far back the charts reach.

Red bands on the charts are labelled faults and grey bands are harmless
surges. These are ground truth for checking the model; the model never sees
them. The Model view tab lists each labelled event in the window and when
(or whether) the model raised an alarm for it.

Some cases worth replaying (cell, reading):

| Case | Cell | Reading |
|---|---|---|
| Interference | CELL_0002 | 518 |
| Coverage gap | CELL_0001 | 477 |
| Hardware fault | CELL_0001 | 629 |
| Capacity overload | CELL_0002 | 939 |
| Handover storm | CELL_0015 | 902 |
| Harmless surge, no alarm | CELL_0001 | 1262 |

## C++ component

The baseline calculation in `features.py` also exists as a C++17 library
(`cpp/`), the part of the feature pipeline that does the most work. It is
exposed to Python through pybind11, and `features.py` uses it automatically
when it is built. The Python version stays as the reference.

- `cpp/src/baseline.cpp`: the algorithm, no Python dependency
- `cpp/tests/test_baseline.cpp`: 12 GoogleTest cases (median, fallback order,
  weekday/weekend, no look-ahead, missing values, bad input)
- `tests/test_cpp_parity.py`: the C++ and Python versions must agree to
  1e-12 on full cells, short histories, missing values and data without
  timestamps

| | Python (numpy) | C++ |
|---|---|---|
| All 100 cells, 14 days | 3.53 s | 0.45 s (7.9x) |
| One cell, a week of history | 18 ms | 2.6 ms |

Largest difference between the two outputs on the full dataset: 0.

### Building it

On macOS you need the Xcode command line tools (`xcode-select --install`);
on Linux a C++17 compiler. CMake and pybind11 come from `requirements-dev.txt`.

```bash
cmake -S cpp -B build -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build -j
ctest --test-dir build --output-on-failure     # GoogleTest
pytest tests/test_cpp_parity.py                 # C++ vs Python
python -m experiments.bench_baseline            # timing
```

The build writes `_netsentinel_cpp*.so` to the repository root. Without it,
everything still works on the Python version and the parity tests are
skipped. `NETSENTINEL_FEATURES=python` forces the Python version.

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

### CI: GitHub Actions and Jenkins

All checks live in `ci/run_checks.sh` (lint, C++ build with warnings as
errors, GoogleTest, pytest with JUnit output, training run). Both CI systems
call that script, so they can't drift apart:

- **GitHub Actions** runs it on Python 3.11 and 3.12, checks that everything
  still works without the C++ module, and builds the Jenkins agent image
  (`ci/Dockerfile`) and runs the full pipeline inside it.
- **Jenkins** (`Jenkinsfile`) runs the same stages in that image, publishes
  the JUnit results and archives the metrics. It is set up to work with
  Gerrit's change-based review through the Gerrit Trigger plugin; see
  `CONTRIBUTING.md`.

The Jenkinsfile itself has not been run on a Jenkins server. Everything it
calls is run on every push by the agent-image job.

## Project layout

```
config.py            KPI definitions, thresholds, simulation and model settings
data_generator.py    synthetic KPI simulation
features.py          per-cell time-of-day baseline and deviation features
evaluation.py        alarms, event-level metrics, baseline detectors
model_trainer.py     training, scoring, held-out comparison
llm.py               LLM backends (Ollama, Anthropic API, none)
llm_eval.py          LLM vs rules evaluation (validation or held-out faults)
prompts.py           classifier prompt variants
cpp/                 C++17 baseline, pybind11 bindings, GoogleTest
ci/                  shared CI script and the Jenkins agent image
Jenkinsfile          Jenkins pipeline
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

- Finish the fair-chance LLM experiments and set the pipeline defaults from them

## References

- 3GPP TS 28.552, 5G performance measurements
- 3GPP TS 38.300, NR overall description
- 3GPP TS 38.331, NR RRC (measurement events and handover parameters)
- 3GPP TS 38.423, XnAP
- 3GPP TS 28.313, SON for 5G networks
- O-RAN Alliance, use cases and deployment scenarios
