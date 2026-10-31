from config import ANOMALY_SCENARIOS
from knowledge_base import format_for_llm, retrieve_procedure


def test_every_scenario_has_a_procedure():
    for mode in ANOMALY_SCENARIOS:
        proc = retrieve_procedure(mode)
        assert proc is not None, mode
        assert proc.actions and proc.parameters


def test_hardware_fault_is_never_auto_applicable():
    assert retrieve_procedure("hardware_fault").auto_applicable is False


def test_unknown_mode_returns_none():
    assert retrieve_procedure("solar_flare") is None


def test_format_for_llm_contains_id_and_actions():
    text = format_for_llm(retrieve_procedure("interference"))
    assert "REM-001" in text and "Actions:" in text
