import json

import pandas as pd
import pytest

from config import ANOMALY_SCENARIOS
from pipeline import flag_kpis, parse_json, robust_zscores, rule_based_classify, validate_diagnosis
from tests.conftest import mid_event_rows, quiet_rows, window_ending_at

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

def test_fault_reading_is_flagged_not_diluted(dataset):
    """Regression: averaging over 12 readings used to hide short faults."""
    flagged = [len(flag_kpis(robust_zscores(window_ending_at(dataset, i))[1])) > 0
               for s in ANOMALY_SCENARIOS for i in mid_event_rows(dataset, s)[:8]]
    assert sum(flagged) / len(flagged) > 0.9


def test_daily_peak_is_not_flagged(dataset):
    """Evening peak traffic is normal for a cell and must not be flagged on its own."""
    rows = [i for i in quiet_rows(dataset, limit=400)
            if 18 <= pd.Timestamp(dataset.loc[i, "timestamp"]).hour <= 20][:40]
    flagged_prb = ["prb_dl" in flag_kpis(robust_zscores(window_ending_at(dataset, i))[1]) for i in rows]
    assert sum(flagged_prb) / len(flagged_prb) < 0.1


def test_past_anomalies_do_not_hide_current_one():
    base = {"rsrp": -75.0, "rsrq": -8.0, "sinr": 15.0, "prb_dl": 45.0, "prb_ul": 35.0,
            "cqi": 9, "ho_sr": 98.5, "rlf_rate": 0.5, "tp_dl": 150.0, "tp_ul": 50.0}
    rows = [dict(base, sinr=15 + (i % 5) - 2) for i in range(40)]
    rows[10]["sinr"] = rows[20]["sinr"] = -20.0       # old faults in the baseline
    rows.append(dict(base, sinr=-18.0))                # current fault
    _, z = robust_zscores(pd.DataFrame(rows))
    assert "sinr" in flag_kpis(z)


# ── Rule-based fallback ───────────────────────────────────────────────────

def test_rule_based_classifier_on_severe_faults(dataset):
    correct, total = 0, 0
    for scenario in ANOMALY_SCENARIOS:
        rows = mid_event_rows(dataset, scenario)
        hits = sum(rule_based_classify(robust_zscores(window_ending_at(dataset, i))[1]) == scenario for i in rows)
        assert hits / len(rows) >= 0.5, f"{scenario}: {hits}/{len(rows)}"
        correct, total = correct + hits, total + len(rows)
    assert correct / total >= 0.75


def test_rule_based_classifier_says_unclassified_when_nothing_stands_out():
    calm = {k: 0.0 for k in ["rsrp", "rsrq", "sinr", "prb_dl", "prb_ul", "cqi", "ho_sr", "rlf_rate",
                             "tp_dl", "tp_ul"]}
    assert rule_based_classify(calm) == "unclassified"


# ── End to end, LLM offline ───────────────────────────────────────────────

def test_llm_offline_uses_rules_and_escalates(dataset, offline_pipeline):
    """Regression: with the LLM down, every fault used to come back as 'interference'."""
    modes = set()
    for scenario in ANOMALY_SCENARIOS:
        i = mid_event_rows(dataset, scenario)[0]
        out = offline_pipeline.run_analysis(dataset.loc[i, "cell_id"], window_ending_at(dataset, i))
        plan = out["action_plan"]
        assert out["is_anomalous"]
        assert out["classification_source"] == "rules"
        assert out["failure_confidence"] == "LOW"
        assert plan["apply_immediately"] is False
        assert plan["escalate_to_human"] is True
        modes.add(out["failure_mode"])
    assert len(modes) >= 4


def test_healthy_cell_goes_to_routine_monitor(dataset, offline_pipeline):
    i = quiet_rows(dataset)[3]
    out = offline_pipeline.run_analysis(dataset.loc[i, "cell_id"], window_ending_at(dataset, i))
    assert out["severity"] == "NORMAL"
    assert out["failure_mode"] == "none"
    assert out["action_plan"]["type"] == "MONITORING"


# ── End to end, scripted LLM ──────────────────────────────────────────────

def _first(dataset, scenario):
    i = mid_event_rows(dataset, scenario)[0]
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
