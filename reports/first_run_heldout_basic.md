# First run: LLM vs rules, original setup

ollama / qwen2.5:7b on a Kaggle T4, 2026-10-31. Original prompt (basic),
LLM decides first, held-out days. Produced by the first version of
`llm_eval.py`, so the layout differs from later reports.

86 held-out fault events. Valid LLM replies: 99%.

| | LLM (valid replies) | Rules | Pipeline (LLM, else rules) |
|---|---|---|---|
| All faults | 58% | 62% | 57% |
| Severe (50) | 66% | 86% | 66% |
| Subtle (36) | 46% | 28% | 44% |

**Auto-apply:** the pipeline would auto-apply a fix in 30 cases, **13 of them for the wrong failure mode.** Without the rules cross-check it would have been 22.

**LLM calibration** (accuracy at each confidence level): HIGH 56% (n=81), MEDIUM 100% (n=4), LOW n/a (n=0)

**Disagreements:** 34. LLM right 11, rules right 15, both wrong 8.

**Latency:** median 2.0 s, p90 2.32 s.

**Most common LLM mistakes:** capacity_overload -> handover_storm (11), coverage_gap -> handover_storm (10), hardware_fault -> coverage_gap (6), hardware_fault -> interference (5), capacity_overload -> interference (3)
