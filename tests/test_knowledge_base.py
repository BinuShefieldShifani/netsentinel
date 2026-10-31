from config import ANOMALY_SCENARIOS
from knowledge_base import format_for_llm, retrieve_procedure


def test_every_scenario_has_a_procedure():
    for mode in ANOMALY_SCENARIOS:
        proc = retrieve_procedure(mode)
        assert proc is not None, mode
        assert proc.actions and proc.parameters


def test_only_son_style_procedures_are_auto_applicable():
    auto = {mode for mode in ANOMALY_SCENARIOS if retrieve_procedure(mode).auto_applicable}
    assert auto == {"capacity_overload", "handover_storm"}


def test_no_lte_only_terms():
    """NR uses Xn between base stations; X2 and coverage-enhancement levels are LTE terms."""
    text = " ".join(" ".join(retrieve_procedure(m).actions) for m in ANOMALY_SCENARIOS)
    assert "X2" not in text and "CE level" not in text


def test_unknown_mode_returns_none():
    assert retrieve_procedure("solar_flare") is None


def test_format_for_llm_contains_id_and_actions():
    text = format_for_llm(retrieve_procedure("interference"))
    assert "REM-001" in text and "Actions:" in text
