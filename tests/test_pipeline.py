import json

import pandas as pd
import pytest

from config import ANOMALY_SCENARIOS
from pipeline import flag_kpis, parse_json, robust_zscores, rule_based_classify, validate_diagnosis
from tests.conftest import held_out_rows, window_ending_at

# ── parse_json / validate_diagnosis ───────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here it is: {"a": 1} hope that helps', {"a": 1}),
    ("not json at all", None),
    ("[1, 2, 3]", None),
    ("", None),
    (None, None),
])
def test_parse_json(text, expected):
    assert parse_json(text) == expected


def test_validate_diagnosis_accepts_valid_output():
    out = validate_diagnosis({"failure_mode": "Interference", "confidence": "high", "summary": "x"})
    assert out == ("interference", "HIGH", "x")


@pytest.mark.parametrize("diagnosis", [
    None,
    {},
    {"failure_mode": "solar_flare", "confidence": "HIGH"},
    {"failure_mode": "interference", "confidence": "VERY SURE"},
    {"failure_mode": "interference"},
])
def test_validate_diagnosis_rejects_bad_output(diagnosis):
    assert validate_diagnosis(diagnosis) is None


# ── KPI monitor ───────────────────────────────────────────────────────────

def test_single_bad_reading_is_flagged_not_diluted(dataset):
    """Regression: averaging over 12 readings used to hide one-reading faults."""
    flagged_counts = []
    for i in held_out_rows(dataset, "interference")[:30]:
        _, z = robust_zscores(window_ending_at(dataset, i))
        flagged_counts.append(len(flag_kpis(z)))
    assert sum(c > 0 for c in flagged_counts) / len(flagged_counts) > 0.9


def test_past_anomalies_do_not_hide_current_one():
    base = {"rsrp": -75.0, "rsrq": -8.0, "sinr": 15.0, "prb_dl": 45.0, "prb_ul": 35.0,
            "cqi": 9, "ho_sr": 98.5, "rlf_rate": 0.5, "tp_dl": 150.0, "tp_ul": 50.0}
    rows = [dict(base, sinr=15 + (i % 5) - 2) for i in range(40)]
    rows[10]["sinr"] = rows[20]["sinr"] = -20.0       # old faults in the baseline
    rows.append(dict(base, sinr=-18.0))                # current fault
    _, z = robust_zscores(pd.DataFrame(rows))
    assert "sinr" in flag_kpis(z)


# ── Rule-based fallback ───────────────────────────────────────────────────

@pytest.mark.parametrize("scenario", list(ANOMALY_SCENARIOS))
def test_rule_based_classifier_accuracy(dataset, scenario):
    rows = held_out_rows(dataset, scenario)[:40]
    correct = sum(
        rule_based_classify(robust_zscores(window_ending_at(dataset, i))[1]) == scenario
        for i in rows
    )
    assert correct / len(rows) >= 0.7, f"{scenario}: {correct}/{len(rows)}"


# ── End to end, LLM offline ───────────────────────────────────────────────

def test_llm_offline_uses_rules_and_escalates(dataset, offline_pipeline):
    """Regression: with the LLM down, every fault used to come back as 'interference'."""
    modes = set()
    for scenario in ANOMALY_SCENARIOS:
        i = held_out_rows(dataset, scenario)[0]
        out = offline_pipeline.run_analysis(dataset.loc[i, "cell_id"], window_ending_at(dataset, i))
        plan = out["action_plan"]
        assert out["classification_source"] == "rules"
        assert out["failure_confidence"] == "LOW"
        assert plan["apply_immediately"] is False
        assert plan["escalate_to_human"] is True
        modes.add(out["failure_mode"])
    assert len(modes) >= 4


def test_healthy_cell_goes_to_routine_monitor(dataset, offline_pipeline):
    i = held_out_rows(dataset, "normal")[5]
    out = offline_pipeline.run_analysis(dataset.loc[i, "cell_id"], window_ending_at(dataset, i))
    assert out["severity"] == "NORMAL"
    assert out["failure_mode"] == "none"
    assert out["action_plan"]["type"] == "MONITORING"


# ── End to end, scripted LLM ──────────────────────────────────────────────

def _first(dataset, scenario):
    i = held_out_rows(dataset, scenario)[0]
    return dataset.loc[i, "cell_id"], window_ending_at(dataset, i)


def test_high_confidence_auto_applicable_fix_is_applied(dataset, fake_llm):
    p = fake_llm(classifier=json.dumps({"failure_mode": "interference", "confidence": "HIGH", "summary": "s"}))
    out = p.run_analysis(*_first(dataset, "interference"))
    assert out["classification_source"] == "llm"
    assert out["action_plan"]["apply_immediately"] is True
    assert out["action_plan"]["escalate_to_human"] is False


def test_hardware_fault_is_never_auto_applied_even_at_high_confidence(dataset, fake_llm):
    p = fake_llm(classifier=json.dumps({"failure_mode": "hardware_fault", "confidence": "HIGH"}))
    plan = p.run_analysis(*_first(dataset, "hardware_fault"))["action_plan"]
    assert plan["apply_immediately"] is False
    assert plan["escalate_to_human"] is True


def test_medium_confidence_is_escalated(dataset, fake_llm):
    p = fake_llm(classifier=json.dumps({"failure_mode": "interference", "confidence": "MEDIUM"}))
    plan = p.run_analysis(*_first(dataset, "interference"))["action_plan"]
    assert plan["apply_immediately"] is False
    assert plan["escalate_to_human"] is True


def test_llm_cannot_override_decision_fields(dataset, fake_llm):
    """The planner LLM tries to switch on auto-apply and swap the actions. It must be ignored."""
    sneaky = json.dumps({
        "apply_immediately": True, "escalate_to_human": False,
        "actions": ["Delete all neighbour relations"],
        "technical_rationale": "SINR collapsed.", "expected_outcome": "SINR recovers.",
    })
    p = fake_llm(classifier=json.dumps({"failure_mode": "hardware_fault", "confidence": "HIGH"}),
                 planner=sneaky)
    plan = p.run_analysis(*_first(dataset, "hardware_fault"))["action_plan"]
    assert plan["apply_immediately"] is False
    assert plan["escalate_to_human"] is True
    assert "Delete all neighbour relations" not in plan["actions"]
    assert plan["technical_rationale"] == "SINR collapsed."   # text fields are allowed


def test_invalid_llm_mode_falls_back_to_rules(dataset, fake_llm):
    p = fake_llm(classifier=json.dumps({"failure_mode": "solar_flare", "confidence": "HIGH"}))
    out = p.run_analysis(*_first(dataset, "capacity_overload"))
    assert out["classification_source"] == "rules"
    assert out["failure_mode"] in ANOMALY_SCENARIOS or out["failure_mode"] == "unclassified"
    assert out["action_plan"]["apply_immediately"] is False


def test_garbage_llm_output_falls_back_to_rules(dataset, fake_llm):
    p = fake_llm(classifier="I think it might be interference?", planner="no json here")
    out = p.run_analysis(*_first(dataset, "interference"))
    assert out["classification_source"] == "rules"
    assert out["action_plan"]["technical_rationale"]   # code-written default is kept
