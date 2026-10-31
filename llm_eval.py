"""
llm_eval.py - Is the LLM better than the rules at naming the fault?
===================================================================
Run (validation days by default; use these to choose prompt and policy):
  NETSENTINEL_LLM=ollama OLLAMA_MODEL=qwen2.5:7b python llm_eval.py --prompt signatures
  NETSENTINEL_LLM=anthropic ANTHROPIC_API_KEY=... python llm_eval.py --prompt fewshot

Final numbers, once everything is chosen (run once):
  python llm_eval.py --split heldout --prompt <chosen>

Splits
  validation  fault events starting on days 8-10
  heldout     fault events starting on days 11-14 (the days the README reports)
  Few-shot examples always come from days 1-7, so they never overlap either split.

For each event it takes the reading about 2 hours in (or the middle of a
short event), gives the LLM the classifier prompt and the rules the same
z-scores, and reports:

  replies             valid, invalid (unusable answer), error (LLM unreachable)
  accuracy            LLM on valid replies, rules, and both decision policies
                        llm_first:   LLM if valid, else rules
                        rules_first: rules, LLM only when the rules say "unclassified"
  by severity         severe vs subtle faults
  calibration         accuracy at each confidence level the LLM gives
  auto-apply          how many automatic fixes, and how many for the WRONG fault, if
                        a) the LLM says HIGH and the fix is auto-applicable
                        b) a) + the rules agree (current pipeline)
                        c) b) + the reading is CRITICAL
  disagreements       when LLM and rules disagree, who is right
  latency             median and p90 seconds per call

A run stops after --max-errors LLM errors in a row, so a broken server can't
produce a report that looks like a result.
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Callable, Optional

import numpy as np
import pandas as pd

import llm
import pipeline
import prompts
from config import ANOMALY_SCENARIOS
from data_generator import generate_dataset
from model_trainer import score_cell_window, severity_from_score, train_mask

REPORT_DIR = "reports"
READINGS_INTO_EVENT = 8      # 2 hours
SUBTLE_BELOW = 0.55
SPLITS = ("validation", "heldout")


class LLMUnavailable(RuntimeError):
    pass


# ── Cases and examples ────────────────────────────────────────────────────

def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)


def split_mask(df: pd.DataFrame, split: str) -> np.ndarray:
    if split == "validation":
        return train_mask(df, 10) & ~train_mask(df, 7)
    if split == "heldout":
        return ~train_mask(df, 10)
    if split == "examples":
        return train_mask(df, 7)
    raise ValueError(f"unknown split {split!r}")


def _events(df: pd.DataFrame, split: str) -> dict[str, list[dict]]:
    """Fault events starting in the split, grouped by true mode."""
    in_split = split_mask(df, split)
    per_cell_pos = df.groupby("cell_id", sort=False).cumcount().to_numpy()
    out = defaultdict(list)
    for ev_id, g in df[df["is_anomaly"] == 1].groupby("event_id", sort=False):
        if not in_split[g.index[0]]:
            continue
        row = g.index[min(READINGS_INTO_EVENT, len(g) // 2)]
        out[g["scenario"].iloc[0]].append({
            "event_id": ev_id, "cell_id": g["cell_id"].iloc[0], "row": int(row),
            "reading": int(per_cell_pos[row]) + 1, "true_mode": g["scenario"].iloc[0],
            "subtle": bool(g["severity"].iloc[0] < SUBTLE_BELOW),
        })
    return out


def pick_cases(df: pd.DataFrame, n: int, split: str = "validation", seed: int = 0) -> list[dict]:
    """Up to n events from the split, spread evenly over the five modes, in mixed order."""
    df = _sorted(df)
    events = _events(df, split)
    rng = np.random.default_rng(seed)
    per_mode = max(1, n // len(ANOMALY_SCENARIOS))
    cases = []
    for mode in ANOMALY_SCENARIOS:
        pool = events.get(mode, [])
        cases += [pool[i] for i in sorted(rng.permutation(len(pool))[:per_mode])]
    return [cases[i] for i in rng.permutation(len(cases))]


def pick_examples(df: pd.DataFrame, per_mode: int = 2, seed: int = 1) -> list[dict]:
    """Few-shot examples from days 1-7: per mode, alternating severe and subtle."""
    df = _sorted(df)
    events = _events(df, "examples")
    rng = np.random.default_rng(seed)
    examples = []
    for mode in ANOMALY_SCENARIOS:
        pool = events.get(mode, [])
        severe = [e for e in pool if not e["subtle"]]
        subtle = [e for e in pool if e["subtle"]]
        chosen = []
        for i in range(per_mode):
            src = severe if i % 2 == 0 else subtle
            src = src or severe or subtle
            if src:
                chosen.append(src.pop(int(rng.integers(len(src)))))
        for e in chosen:
            cell = df[df["cell_id"] == e["cell_id"]].loc[: e["row"]]
            _, z = pipeline.robust_zscores(cell)
            examples.append({"z": z, "mode": mode, "event_id": e["event_id"]})
    return [examples[i] for i in rng.permutation(len(examples))]


# ── One case ──────────────────────────────────────────────────────────────

def run_case(df: pd.DataFrame, case: dict, model, scaler, variant: str,
             examples: Optional[list[dict]]) -> dict:
    cell = df[df["cell_id"] == case["cell_id"]].loc[: case["row"]]
    snapshot, z = pipeline.robust_zscores(cell)
    flagged = pipeline.flag_kpis(z)
    score, model_flag = score_cell_window(cell, model, scaler)
    severity = severity_from_score(score)

    prompt = pipeline.classifier_prompt(case["cell_id"], score, snapshot, z, flagged, variant, examples)
    start = time.perf_counter()
    llm_result, raw = pipeline.diagnose(prompt)
    latency = time.perf_counter() - start
    status = "ok" if llm_result else ("error" if raw is None else "invalid")

    rules_mode = pipeline.rule_based_classify(z)
    llm_mode, llm_conf = (llm_result[0], llm_result[1]) if llm_result else (None, None)
    final = {p: pipeline.decide(llm_result, z, p) for p in pipeline.POLICIES}

    # Auto-apply variants are about the LLM's own call (mode + confidence).
    auto_high = bool(llm_result) and pipeline.would_auto_apply(llm_mode, llm_conf, llm_mode,
                                                               require_critical=False)
    auto_agree = bool(llm_result) and pipeline.would_auto_apply(llm_mode, llm_conf, rules_mode,
                                                                require_critical=False)
    auto_critical = bool(llm_result) and pipeline.would_auto_apply(llm_mode, llm_conf, rules_mode, severity,
                                                                   require_critical=True)
    wrong = llm_mode != case["true_mode"]
    return {
        **case,
        "anomaly_score": round(score, 4),
        "severity_level": severity,
        "detected": bool(model_flag or len(flagged) >= 2),
        "status": status,
        "llm_mode": llm_mode,
        "llm_confidence": llm_conf,
        "llm_raw": (raw or "")[:300],
        "rules_mode": rules_mode,
        "final_llm_first": final["llm_first"][0],
        "final_rules_first": final["rules_first"][0],
        "auto_high": auto_high, "unsafe_high": auto_high and wrong,
        "auto_agree": auto_agree, "unsafe_agree": auto_agree and wrong,
        "auto_critical": auto_critical, "unsafe_critical": auto_critical and wrong,
        "latency_s": round(latency, 3),
    }


# ── Summary ───────────────────────────────────────────────────────────────

def _acc(rows: pd.DataFrame, col: str) -> Optional[float]:
    return round(float((rows[col] == rows["true_mode"]).mean()), 3) if len(rows) else None


def summarise(results: list[dict], meta: dict) -> dict:
    r = pd.DataFrame(results)
    valid = r[r["status"] == "ok"]
    disagree = valid[valid["llm_mode"] != valid["rules_mode"]]
    abstain = r[r["rules_mode"] == pipeline.UNCLASSIFIED]
    cols = {"llm_on_valid": "llm_mode", "rules": "rules_mode",
            "llm_first": "final_llm_first", "rules_first": "final_rules_first"}

    def accs(rows, valid_rows):
        return {k: _acc(valid_rows if k == "llm_on_valid" else rows, c) for k, c in cols.items()}

    return {
        **meta,
        "n_cases": int(len(r)),
        "replies": {s: int((r["status"] == s).sum()) for s in ("ok", "invalid", "error")},
        "llm_valid_rate": round(float((r["status"] == "ok").mean()), 3),
        "accuracy": accs(r, valid),
        "accuracy_by_severity": {
            name: {**accs(r[r["subtle"] == s], valid[valid["subtle"] == s]), "n": int((r["subtle"] == s).sum())}
            for name, s in (("severe", False), ("subtle", True))
        },
        "accuracy_by_true_mode": {m: accs(r[r["true_mode"] == m], valid[valid["true_mode"] == m])
                                  for m in sorted(r["true_mode"].unique())},
        "llm_when_rules_abstain": {"n": int(len(abstain)),
                                   "llm_accuracy": _acc(abstain[abstain["status"] == "ok"], "llm_mode")},
        "llm_calibration": {c: {"n": int((valid["llm_confidence"] == c).sum()),
                                "accuracy": _acc(valid[valid["llm_confidence"] == c], "llm_mode")}
                            for c in ("HIGH", "MEDIUM", "LOW")},
        "auto_apply": {name: {"applied": int(r[f"auto_{name}"].sum()), "wrong": int(r[f"unsafe_{name}"].sum())}
                       for name in ("high", "agree", "critical")},
        "disagreements": {
            "n": int(len(disagree)),
            "llm_right": int((disagree["llm_mode"] == disagree["true_mode"]).sum()),
            "rules_right": int((disagree["rules_mode"] == disagree["true_mode"]).sum()),
            "both_wrong": int(((disagree["llm_mode"] != disagree["true_mode"])
                               & (disagree["rules_mode"] != disagree["true_mode"])).sum()),
        },
        "llm_confusion": {f"{t} -> {p}": c for (t, p), c in
                          Counter(zip(valid["true_mode"], valid["llm_mode"])).most_common() if t != p},
        "latency_s": {"median": round(float(valid["latency_s"].median()), 2) if len(valid) else None,
                      "p90": round(float(valid["latency_s"].quantile(0.9)), 2) if len(valid) else None},
    }


def to_markdown(s: dict) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x:.0%}"

    a, sev, au, d, rp = (s["accuracy"], s["accuracy_by_severity"], s["auto_apply"],
                         s["disagreements"], s["replies"])
    lines = [
        f"# LLM vs rules: {s['label']}",
        "",
        f"Split: **{s['split']}** · prompt: **{s['prompt']}** · {s['n_cases']} fault events · "
        f"replies: {rp['ok']} valid, {rp['invalid']} invalid, {rp['error']} errors",
    ]
    if rp["error"]:
        lines += ["", f"**Warning: {rp['error']} LLM errors (timeouts or unreachable server). "
                      "Those cases count as no reply.**"]
    lines += [
        "",
        "| Accuracy | LLM (valid replies) | Rules | Policy: llm_first | Policy: rules_first |",
        "|---|---|---|---|---|",
    ]
    for name, row in (("All faults", a), (f"Severe ({sev['severe']['n']})", sev["severe"]),
                      (f"Subtle ({sev['subtle']['n']})", sev["subtle"])):
        lines.append(f"| {name} | {pct(row['llm_on_valid'])} | {pct(row['rules'])} | "
                     f"{pct(row['llm_first'])} | {pct(row['rules_first'])} |")
    ab = s["llm_when_rules_abstain"]
    lines += [
        "",
        f"When the rules say \"unclassified\" ({ab['n']} cases), the LLM is right {pct(ab['llm_accuracy'])}.",
        "",
        "| Automatic fixes, if required: | Applied | Wrong fault |",
        "|---|---|---|",
        f"| LLM says HIGH, fix is auto-applicable | {au['high']['applied']} | {au['high']['wrong']} |",
        f"| ... and the rules agree (current) | {au['agree']['applied']} | {au['agree']['wrong']} |",
        f"| ... and the reading is CRITICAL | {au['critical']['applied']} | {au['critical']['wrong']} |",
        "",
        "**LLM calibration:** " + ", ".join(f"{k} {pct(v['accuracy'])} (n={v['n']})"
                                            for k, v in s["llm_calibration"].items()),
        "",
        f"**Disagreements:** {d['n']}. LLM right {d['llm_right']}, rules right {d['rules_right']}, "
        f"both wrong {d['both_wrong']}.",
        "",
        f"**Latency:** median {s['latency_s']['median']} s, p90 {s['latency_s']['p90']} s.",
    ]
    if s["llm_confusion"]:
        lines += ["", "**Most common LLM mistakes:** "
                  + ", ".join(f"{k} ({v})" for k, v in list(s["llm_confusion"].items())[:5])]
    return "\n".join(lines) + "\n"


# ── Entry points ──────────────────────────────────────────────────────────

def evaluate(n: int = 100, split: str = "validation", variant: str = "basic", shots: int = 2,
             complete_fn: Optional[Callable] = None, df: Optional[pd.DataFrame] = None,
             model_and_scaler=None, max_errors: Optional[int] = None,
             progress: bool = True) -> tuple[dict, list[dict]]:
    """complete_fn replaces the real LLM for this run (used by the tests)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    original = pipeline.call_llm
    if complete_fn is not None:
        pipeline.call_llm = complete_fn
    try:
        df = _sorted(generate_dataset() if df is None else df)
        model, scaler = model_and_scaler or pipeline.get_model()
        examples = pick_examples(df, shots) if variant == "fewshot" else None
        cases = pick_cases(df, n, split)
        results, errors_in_a_row = [], 0
        for i, case in enumerate(cases, 1):
            res = run_case(df, case, model, scaler, variant, examples)
            results.append(res)
            errors_in_a_row = errors_in_a_row + 1 if res["status"] == "error" else 0
            if max_errors and errors_in_a_row >= max_errors:
                raise LLMUnavailable(
                    f"{errors_in_a_row} LLM errors in a row after {i} cases. Stopping, because the "
                    "rest of the report would only measure a broken server. Check the LLM and run again.")
            if progress and (i % 10 == 0 or i == len(cases)):
                print(f"  {i}/{len(cases)} cases")
    finally:
        pipeline.call_llm = original
    meta = {"label": f"{llm.backend()} / {llm.model_name()}", "split": split, "prompt": variant,
            "shots_per_mode": shots if variant == "fewshot" else 0}
    return summarise(results, meta), results


def warm_up() -> None:
    """One short request first: loads the model, and fails fast if the LLM isn't reachable."""
    print("Warming up the LLM ...")
    start = time.perf_counter()
    if pipeline.call_llm("Reply with the word OK.") is None:
        raise LLMUnavailable("The LLM did not answer the warm-up request. Check NETSENTINEL_LLM, "
                             "the API key or the Ollama server, and NETSENTINEL_LLM_TIMEOUT.")
    print(f"  ready ({time.perf_counter() - start:.1f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="number of fault events (spread over 5 modes)")
    ap.add_argument("--split", choices=SPLITS, default="validation")
    ap.add_argument("--prompt", choices=prompts.VARIANTS, default="basic")
    ap.add_argument("--shots", type=int, default=2, help="few-shot examples per failure mode")
    ap.add_argument("--max-errors", type=int, default=3, help="stop after this many LLM errors in a row")
    args = ap.parse_args()

    if args.split == "heldout":
        print("NOTE: held-out days. Use these for the final numbers only, not to choose a prompt or policy.")
    label = f"{llm.backend()} / {llm.model_name()}"
    print(f"Evaluating {label} | split {args.split} | prompt {args.prompt}")
    try:
        warm_up()
        summary, results = evaluate(args.n, args.split, args.prompt, args.shots, max_errors=args.max_errors)
    except LLMUnavailable as e:
        raise SystemExit(f"\nSTOPPED: {e}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{llm.backend()}-{llm.model_name()}-{args.split}-{args.prompt}".lower())
    slug = slug.strip("-")
    with open(f"{REPORT_DIR}/llm_eval_{slug}.json", "w") as f:
        json.dump({"summary": summary, "cases": results}, f, indent=2)
    md = to_markdown(summary)
    with open(f"{REPORT_DIR}/llm_eval_{slug}.md", "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"Saved {REPORT_DIR}/llm_eval_{slug}.json and .md")


if __name__ == "__main__":
    main()
