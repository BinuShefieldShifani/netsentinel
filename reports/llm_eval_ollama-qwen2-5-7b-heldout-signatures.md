# LLM vs rules: ollama / qwen2.5:7b

Split: **heldout** · prompt: **signatures** · 86 fault events · replies: 86 valid, 0 invalid, 0 errors

| Accuracy | LLM (valid replies) | Rules | Policy: llm_first | Policy: rules_first |
|---|---|---|---|---|
| All faults | 65% | 62% | 65% | 71% |
| Severe (50) | 78% | 86% | 78% | 88% |
| Subtle (36) | 47% | 28% | 47% | 47% |

When the rules say "unclassified" (16 cases), the LLM is right 50%.

| Automatic fixes, if required: | Applied | Wrong fault |
|---|---|---|
| LLM says HIGH, fix is auto-applicable | 32 | 10 |
| ... and the rules agree (current) | 28 | 9 |
| ... and the reading is CRITICAL | 10 | 2 |

**LLM calibration:** HIGH 69% (n=61), MEDIUM 56% (n=25), LOW n/a (n=0)

**Disagreements:** 34. LLM right 14, rules right 11, both wrong 9.

**Latency:** median 1.48 s, p90 1.69 s.

**Most common LLM mistakes:** coverage_gap -> handover_storm (10), hardware_fault -> coverage_gap (8), capacity_overload -> handover_storm (4), interference -> coverage_gap (4), coverage_gap -> hardware_fault (2)
