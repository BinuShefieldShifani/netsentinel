# Validation runs: choosing the prompt, model and policy

Validation days 8-10, 54 fault events per run, Qwen2.5 through Ollama on a
Kaggle T4, 2026-11-01. Copied from the output of
`python -m experiments.compare_llm_reports --split validation`. The per-case
JSON files were lost when the Kaggle session ended; this summary is what
remains.

| Model | Prompt | Valid | LLM | Rules | llm_first | rules_first | LLM on subtle | Acc. when HIGH | Auto fixes / wrong (rules agree) | (+ CRITICAL) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:14b | basic | 100% | 59% | 68% | 59% | 76% | 61% | 62% | 20 / 3 | 6 / 0 |
| qwen2.5:14b | fewshot | 100% | 63% | 68% | 63% | 72% | 48% | 65% | 25 / 2 | 7 / 0 |
| qwen2.5:14b | signatures | 100% | 56% | 68% | 56% | 72% | 44% | 57% | 21 / 2 | 6 / 0 |
| qwen2.5:7b | basic | 100% | 52% | 68% | 52% | 74% | 52% | 54% | 18 / 3 | 6 / 0 |
| qwen2.5:7b | fewshot | 100% | 65% | 68% | 65% | 76% | 48% | 66% | 28 / 6 | 10 / 3 |
| qwen2.5:7b | signatures | 100% | 68% | 68% | 68% | 76% | 56% | 68% | 22 / 3 | 7 / 0 |

## Applying the selection rule (written down before these runs)

1. **Fewest wrong automatic fixes.** Requiring a CRITICAL reading gives 0
   wrong fixes in five of six setups. Chosen: CRITICAL required. Ruled out:
   7B few-shot (3 wrong even with CRITICAL).
2. **Best pipeline accuracy.** rules_first beats llm_first in every setup.
   Best: 76%, a tie between 14B basic and 7B signatures.
3. **Simplest setup.** Smaller model first: **7B, signatures prompt,
   rules_first, CRITICAL required.**

The gain of rules_first over rules alone (76% vs 68%) is about 4 cases out
of 54, below the ~10-point noise level stated in advance. The held-out run
decides whether it is real.
