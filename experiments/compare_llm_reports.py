"""
Put several llm_eval runs side by side.
Run:  python -m experiments.compare_llm_reports [--split validation]
"""

import argparse
import glob
import json


def pct(x):
    return "  n/a" if x is None else f"{x:5.0%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, help="only show runs on this split")
    args = ap.parse_args()

    runs = []
    for path in sorted(glob.glob("reports/llm_eval_*.json")):
        s = json.load(open(path)).get("summary", {})
        if "split" not in s or (args.split and s["split"] != args.split):
            continue
        runs.append(s)
    if not runs:
        print("No reports found. Run llm_eval.py first.")
        return

    head = (f"{'model':<26} {'split':<10} {'prompt':<10} {'n':>3} {'valid':>6} "
            f"{'LLM':>6} {'rules':>6} {'llm_1st':>8} {'rules_1st':>9} {'subtle':>7} "
            f"{'HIGH acc':>8} {'auto/wrong (agree)':>19} {'(+crit)':>8}")
    print(head)
    print("-" * len(head))
    for s in runs:
        a, au = s["accuracy"], s["auto_apply"]
        print(f"{s['label']:<26} {s['split']:<10} {s['prompt']:<10} {s['n_cases']:>3} "
              f"{pct(s['llm_valid_rate']):>6} {pct(a['llm_on_valid']):>6} {pct(a['rules']):>6} "
              f"{pct(a['llm_first']):>8} {pct(a['rules_first']):>9} "
              f"{pct(s['accuracy_by_severity']['subtle']['llm_on_valid']):>7} "
              f"{pct(s['llm_calibration']['HIGH']['accuracy']):>8} "
              f"{au['agree']['applied']:>12}/{au['agree']['wrong']:<6} "
              f"{au['critical']['applied']:>4}/{au['critical']['wrong']:<3}")


if __name__ == "__main__":
    main()
