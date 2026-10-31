"""
llm_eval.py - Is the LLM better than the rules at naming the fault?
===================================================================
Run:
  NETSENTINEL_LLM=anthropic ANTHROPIC_API_KEY=... python llm_eval.py --n 100
  NETSENTINEL_LLM=ollama OLLAMA_MODEL=gpt-oss:20b python llm_eval.py --n 100

Takes labelled fault events from the held-out days, picks the reading about
2 hours into each event (or the middle, for short events), and gives the LLM
exactly the prompt the pipeline uses. The rule-based classifier gets the same
z-scores. Then it reports:

  valid output        share of replies that pass validation
  accuracy            LLM (valid replies only), rules, and the final decision
                      the pipeline would make (LLM if valid, else rules)
  by severity         severe vs subtle faults
  calibration         accuracy at each confidence level the LLM gives
  unsafe auto-apply   cases where the pipeline WOULD auto-apply a fix
                      (HIGH confidence + auto-applicable procedure + rules
                      agree) for the WRONG failure mode, and how many there
                      would be without the rules cross-check. This is the
                      number that matters most.
  disagreements       when LLM and rules disagree, who is right
  latency             median and p90 seconds per call

Writes reports/llm_eval_<backend>_<model>.json and .md.
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
from config import ANOMALY_SCENARIOS
from data_generator import generate_dataset
from model_trainer import score_cell_window, train_mask

REPORT_DIR = "reports"
READINGS_INTO_EVENT = 8      # 2 hours


# ── Picking cases ─────────────────────────────────────────────────────────

def pick_cases(df: pd.DataFrame, n: int, seed: int = 0) -> list[dict]:
    """Up to n fault events from the held-out days, spread evenly over the five scenarios."""
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    held_out = ~train_mask(df)
    per_cell_start = df.groupby("cell_id", sort=False).cumcount().to_numpy()
    candidates = defaultdict(list)
    for ev_id, g in df[df["is_anomaly"] == 1].groupby("event_id", sort=False):
        first = g.index[0]
        if not held_out[first]:
            continue
        row = g.index[min(READINGS_INTO_EVENT, len(g) // 2)]
        candidates[g["scenario"].iloc[0]].append({
            "event_id": ev_id, "cell_id": g["cell_id"].iloc[0], "row": int(row),
            "reading": int(per_cell_start[row]) + 1, "true_mode": g["scenario"].iloc[0],
            "subtle": bool(g["severity"].iloc[0] < 0.55),
        })
    rng = np.random.default_rng(seed)
    per_mode = max(1, n // len(ANOMALY_SCENARIOS))
    cases = []
    for mode in ANOMALY_SCENARIOS:
        pool = candidates.get(mode, [])
        idx = rng.permutation(len(pool))[:per_mode]
        cases += [pool[i] for i in sorted(idx)]
    return cases


# ── Running one case ──────────────────────────────────────────────────────

def run_case(df: pd.DataFrame, case: dict, model, scaler) -> dict:
    cell = df[df["cell_id"] == case["cell_id"]].loc[: case["row"]]
    snapshot, z = pipeline.robust_zscores(cell)
    flagged = pipeline.flag_kpis(z)
    score, model_flag = score_cell_window(cell, model, scaler)

    prompt = pipeline.classifier_prompt(case["cell_id"], score, snapshot, z, flagged)
    start = time.perf_counter()
    llm_result, raw = pipeline.diagnose(prompt)
    latency = time.perf_counter() - start

    rules_mode = pipeline.rule_based_classify(z)
    final_mode, final_conf, _, source = pipeline.decide(llm_result, z)
    auto = pipeline.would_auto_apply(final_mode, final_conf, rules_mode)
    auto_without_crosscheck = pipeline.would_auto_apply(final_mode, final_conf, final_mode)
    return {
        **case,
        "anomaly_score": round(score, 4),
        "detected": bool(model_flag or len(flagged) >= 2),
        "llm_valid": llm_result is not None,
        "llm_mode": llm_result[0] if llm_result else None,
        "llm_confidence": llm_result[1] if llm_result else None,
        "llm_raw": (raw or "")[:300],
        "rules_mode": rules_mode,
        "final_mode": final_mode,
        "final_confidence": final_conf,
        "source": source,
        "would_auto_apply": auto,
        "unsafe_auto_apply": auto and final_mode != case["true_mode"],
        "unsafe_without_crosscheck": auto_without_crosscheck and final_mode != case["true_mode"],
        "latency_s": round(latency, 3),
    }


# ── Summary ───────────────────────────────────────────────────────────────

def _acc(rows: pd.DataFrame, col: str) -> Optional[float]:
    return round(float((rows[col] == rows["true_mode"]).mean()), 3) if len(rows) else None


def summarise(results: list[dict]) -> dict:
    r = pd.DataFrame(results)
    valid = r[r["llm_valid"]]
    disagree = valid[valid["llm_mode"] != valid["rules_mode"]]
    out = {
        "n_cases": int(len(r)),
        "llm_valid_rate": round(float(r["llm_valid"].mean()), 3),
        "accuracy": {
            "llm_on_valid": _acc(valid, "llm_mode"),
            "rules": _acc(r, "rules_mode"),
            "final_pipeline": _acc(r, "final_mode"),
        },
        "accuracy_by_severity": {
            sev: {"llm_on_valid": _acc(valid[valid["subtle"] == s], "llm_mode"),
                  "rules": _acc(r[r["subtle"] == s], "rules_mode"),
                  "final_pipeline": _acc(r[r["subtle"] == s], "final_mode"),
                  "n": int((r["subtle"] == s).sum())}
            for sev, s in (("severe", False), ("subtle", True))
        },
        "accuracy_by_true_mode": {
            mode: {"llm_on_valid": _acc(valid[valid["true_mode"] == mode], "llm_mode"),
                   "rules": _acc(r[r["true_mode"] == mode], "rules_mode")}
            for mode in sorted(r["true_mode"].unique())
        },
        "llm_calibration": {
            conf: {"n": int((valid["llm_confidence"] == conf).sum()),
                   "accuracy": _acc(valid[valid["llm_confidence"] == conf], "llm_mode")}
            for conf in ("HIGH", "MEDIUM", "LOW")
        },
        "auto_apply": {
            "would_auto_apply": int(r["would_auto_apply"].sum()),
            "unsafe_auto_apply": int(r["unsafe_auto_apply"].sum()),
            "unsafe_without_crosscheck": int(r["unsafe_without_crosscheck"].sum()),
        },
        "disagreements": {
            "n": int(len(disagree)),
            "llm_right": int((disagree["llm_mode"] == disagree["true_mode"]).sum()),
            "rules_right": int((disagree["rules_mode"] == disagree["true_mode"]).sum()),
            "both_wrong": int(((disagree["llm_mode"] != disagree["true_mode"])
                               & (disagree["rules_mode"] != disagree["true_mode"])).sum()),
        },
        "llm_confusion": {f"{t} -> {p}": c for (t, p), c in
                          Counter(zip(valid["true_mode"], valid["llm_mode"])).most_common() if t != p},
        "latency_s": {"median": round(float(r["latency_s"].median()), 2),
                      "p90": round(float(r["latency_s"].quantile(0.9)), 2)},
    }
    return out


def to_markdown(summary: dict, label: str) -> str:
    a, s, au, d = summary["accuracy"], summary["accuracy_by_severity"], summary["auto_apply"], summary["disagreements"]

    def pct(x):
        return "n/a" if x is None else f"{x:.0%}"

    lines = [
        f"# LLM vs rules: {label}",
        "",
        f"{summary['n_cases']} held-out fault events. Valid LLM replies: {pct(summary['llm_valid_rate'])}.",
        "",
        "| | LLM (valid replies) | Rules | Pipeline (LLM, else rules) |",
        "|---|---|---|---|",
        f"| All faults | {pct(a['llm_on_valid'])} | {pct(a['rules'])} | {pct(a['final_pipeline'])} |",
        f"| Severe ({s['severe']['n']}) | {pct(s['severe']['llm_on_valid'])} | {pct(s['severe']['rules'])} "
        f"| {pct(s['severe']['final_pipeline'])} |",
        f"| Subtle ({s['subtle']['n']}) | {pct(s['subtle']['llm_on_valid'])} | {pct(s['subtle']['rules'])} "
        f"| {pct(s['subtle']['final_pipeline'])} |",
        "",
        f"**Auto-apply:** the pipeline would auto-apply a fix in {au['would_auto_apply']} cases, "
        f"**{au['unsafe_auto_apply']} of them for the wrong failure mode.** "
        f"Without the rules cross-check it would have been {au['unsafe_without_crosscheck']}.",
        "",
        "**LLM calibration** (accuracy at each confidence level): "
        + ", ".join(f"{k} {pct(v['accuracy'])} (n={v['n']})" for k, v in summary["llm_calibration"].items()),
        "",
        f"**Disagreements:** {d['n']}. LLM right {d['llm_right']}, rules right {d['rules_right']}, "
        f"both wrong {d['both_wrong']}.",
        "",
        f"**Latency:** median {summary['latency_s']['median']} s, p90 {summary['latency_s']['p90']} s.",
    ]
    if summary["llm_confusion"]:
        lines += ["", "**Most common LLM mistakes:** "
                  + ", ".join(f"{k} ({v})" for k, v in list(summary["llm_confusion"].items())[:5])]
    return "\n".join(lines) + "\n"


# ── Entry point ───────────────────────────────────────────────────────────

def evaluate(n: int = 100, complete_fn: Optional[Callable] = None, df: Optional[pd.DataFrame] = None,
             model_and_scaler=None, progress: bool = True) -> tuple[dict, list[dict]]:
    """complete_fn replaces the real LLM for this run (used by the tests)."""
    original = pipeline.call_llm
    if complete_fn is not None:
        pipeline.call_llm = complete_fn
    try:
        df = generate_dataset() if df is None else df
        df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
        model, scaler = model_and_scaler or pipeline.get_model()
        cases = pick_cases(df, n)
        results = []
        for i, case in enumerate(cases, 1):
            results.append(run_case(df, case, model, scaler))
            if progress and (i % 10 == 0 or i == len(cases)):
                print(f"  {i}/{len(cases)} cases")
    finally:
        pipeline.call_llm = original
    return summarise(results), results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="number of fault events (spread over 5 scenarios)")
    args = ap.parse_args()

    label = f"{llm.backend()} / {llm.model_name()}"
    print(f"Evaluating {label} on up to {args.n} held-out fault events ...")
    summary, results = evaluate(args.n)
    if summary["llm_valid_rate"] == 0:
        print("\nNo valid LLM replies. Is the LLM configured? (NETSENTINEL_LLM, API key or Ollama server)")

    os.makedirs(REPORT_DIR, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{llm.backend()}-{llm.model_name()}".lower()).strip("-")
    with open(f"{REPORT_DIR}/llm_eval_{slug}.json", "w") as f:
        json.dump({"label": label, "summary": summary, "cases": results}, f, indent=2)
    md = to_markdown(summary, label)
    with open(f"{REPORT_DIR}/llm_eval_{slug}.md", "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"Saved {REPORT_DIR}/llm_eval_{slug}.json and .md")


if __name__ == "__main__":
    main()
