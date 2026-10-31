"""The LLM evaluation harness, run against scripted fake LLMs."""

import json

import pytest

import llm
import llm_eval
import pipeline
from config import ANOMALY_SCENARIOS


@pytest.fixture(scope="module")
def run(dataset, model_and_scaler):
    def _run(fn):
        return llm_eval.evaluate(50, complete_fn=fn, df=dataset, model_and_scaler=model_and_scaler,
                                 progress=False)
    return _run


def test_cases_come_from_held_out_days_and_cover_all_modes(dataset):
    from model_trainer import train_mask
    df = dataset.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    cases = llm_eval.pick_cases(df, 50)
    assert {c["true_mode"] for c in cases} == set(ANOMALY_SCENARIOS)
    held_out = ~train_mask(df)
    assert all(held_out[c["row"]] for c in cases)
    assert len({c["event_id"] for c in cases}) == len(cases)


def test_no_llm_means_pipeline_equals_rules(run):
    summary, _ = run(lambda p, system="": None)
    assert summary["llm_valid_rate"] == 0
    assert summary["accuracy"]["final_pipeline"] == summary["accuracy"]["rules"]
    assert summary["auto_apply"]["would_auto_apply"] == 0


def test_garbage_replies_are_not_counted_as_valid(run):
    summary, rows = run(lambda p, system="": "probably interference, not sure")
    assert summary["llm_valid_rate"] == 0
    assert all(r["source"] == "rules" for r in rows)


def test_confidently_wrong_llm_is_contained_by_the_cross_check(run):
    wrong = json.dumps({"failure_mode": "handover_storm", "confidence": "HIGH"})
    summary, _ = run(lambda p, system="": wrong)
    au = summary["auto_apply"]
    assert summary["llm_valid_rate"] == 1
    assert au["unsafe_without_crosscheck"] > 0
    assert au["unsafe_auto_apply"] < au["unsafe_without_crosscheck"] / 2


def test_markdown_report_mentions_the_key_numbers(run):
    summary, _ = run(lambda p, system="": None)
    md = llm_eval.to_markdown(summary, "test")
    assert "Rules" in md and "Auto-apply" in md and "wrong failure mode" in md


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
