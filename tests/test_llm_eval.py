"""The LLM evaluation harness, run against scripted fake LLMs."""

import json

import pytest

import llm
import llm_eval
import pipeline
import prompts
from config import ANOMALY_SCENARIOS
from model_trainer import train_mask

WRONG_HIGH = json.dumps({"failure_mode": "handover_storm", "confidence": "HIGH"})


@pytest.fixture(scope="module")
def run(dataset, model_and_scaler):
    def _run(fn, split="validation", variant="basic", **kw):
        return llm_eval.evaluate(50, split, variant, complete_fn=fn, df=dataset,
                                 model_and_scaler=model_and_scaler, progress=False, **kw)
    return _run


# ── Splits and examples ───────────────────────────────────────────────────

@pytest.mark.parametrize("split,lo,hi", [("validation", 7, 10), ("heldout", 10, 99)])
def test_cases_come_from_their_split(dataset, split, lo, hi):
    df = llm_eval._sorted(dataset)
    cases = llm_eval.pick_cases(df, 50, split)
    assert cases
    in_split = train_mask(df, hi) & ~train_mask(df, lo) if hi < 99 else ~train_mask(df, lo)
    first_rows = {ev: g.index[0] for ev, g in df[df["is_anomaly"] == 1].groupby("event_id")}
    assert all(in_split[first_rows[c["event_id"]]] for c in cases)
    assert len({c["event_id"] for c in cases}) == len(cases)


def test_cases_are_in_mixed_order(dataset):
    modes = [c["true_mode"] for c in llm_eval.pick_cases(llm_eval._sorted(dataset), 50, "heldout")]
    assert modes != sorted(modes, key=list(ANOMALY_SCENARIOS).index)


def test_fewshot_examples_never_overlap_the_evaluation_splits(dataset):
    df = llm_eval._sorted(dataset)
    examples = llm_eval.pick_examples(df, 2)
    assert {e["mode"] for e in examples} == set(ANOMALY_SCENARIOS)
    used = {e["event_id"] for e in examples}
    for split in llm_eval.SPLITS:
        assert not used & {c["event_id"] for c in llm_eval.pick_cases(df, 500, split)}


# ── Prompts ───────────────────────────────────────────────────────────────

def _prompt(variant, examples=None):
    z = {k: -1.0 for k in prompts.KPIS}
    return pipeline.classifier_prompt("CELL_X", -0.1, {k: 1.0 for k in prompts.KPIS}, z, ["sinr"],
                                      variant, examples)


def test_prompt_variants_add_what_they_promise():
    basic, sig = _prompt("basic"), _prompt("signatures")
    fewshot = _prompt("fewshot", [{"z": {k: 0.5 for k in prompts.KPIS}, "mode": "interference"}])
    assert "Typical pattern" not in basic and "Typical pattern" in sig
    assert "SOLVED EXAMPLES" in fewshot and "SOLVED EXAMPLES" not in sig
    for p in (basic, sig, fewshot):
        assert "Respond with JSON only" in p and "CELL_X" in p


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        _prompt("magic")


# ── Reply handling ────────────────────────────────────────────────────────

def test_unreachable_llm_counts_as_errors_and_pipeline_equals_rules(run):
    summary, _ = run(lambda p, system="": None)
    assert summary["replies"]["error"] == summary["n_cases"] and summary["replies"]["ok"] == 0
    assert summary["accuracy"]["llm_first"] == summary["accuracy"]["rules"]
    assert summary["accuracy"]["rules_first"] == summary["accuracy"]["rules"]
    assert all(v["applied"] == 0 for v in summary["auto_apply"].values())


def test_garbage_counts_as_invalid_not_error(run):
    summary, _ = run(lambda p, system="": "probably interference, not sure")
    assert summary["replies"]["invalid"] == summary["n_cases"]
    assert summary["replies"]["error"] == 0


def test_run_stops_after_errors_in_a_row(run):
    with pytest.raises(llm_eval.LLMUnavailable):
        run(lambda p, system="": None, max_errors=3)


# ── Policies and auto-apply ───────────────────────────────────────────────

def test_rules_first_ignores_a_wrong_llm_when_rules_are_sure(run):
    summary, rows = run(lambda p, system="": WRONG_HIGH, split="heldout")
    for r in rows:
        if r["rules_mode"] != pipeline.UNCLASSIFIED:
            assert r["final_rules_first"] == r["rules_mode"]
    assert summary["accuracy"]["rules_first"] >= summary["accuracy"]["llm_first"]


def test_each_auto_apply_condition_can_only_remove_fixes(run):
    summary, _ = run(lambda p, system="": WRONG_HIGH, split="heldout")
    au = summary["auto_apply"]
    assert au["high"]["applied"] >= au["agree"]["applied"] >= au["critical"]["applied"]
    assert au["high"]["wrong"] > 0
    assert au["agree"]["wrong"] < au["high"]["wrong"]


def test_markdown_report_has_the_key_numbers(run):
    summary, _ = run(lambda p, system="": None)
    md = llm_eval.to_markdown(summary)
    assert "rules_first" in md and "Wrong fault" in md and "Split: **validation**" in md
    assert "LLM errors" in md


def test_harness_restores_the_real_llm_function(run):
    before = pipeline.call_llm
    run(lambda p, system="": None)
    assert pipeline.call_llm is before


def test_llm_backends_fail_quietly_without_config(monkeypatch):
    monkeypatch.setenv("NETSENTINEL_LLM", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm.complete("hi") is None
    monkeypatch.setenv("NETSENTINEL_LLM", "none")
    assert llm.complete("hi") is None
    assert "off" in llm.describe_backend()
