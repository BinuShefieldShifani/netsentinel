"""
prompts.py - Classifier prompt variants
=======================================
  basic       KPI readings and a one-line description of each failure mode
  signatures  basic + what each failure mode does to the KPIs. This is the
              same knowledge the rule-based classifier is built on, so the
              LLM and the rules start from the same information.
  fewshot     signatures + worked examples (readings and the right answer)
              taken from training days only

Which variant the pipeline uses is set by NETSENTINEL_PROMPT (config.py).
Variants are compared on validation days with llm_eval.py, never on the
held-out days.
"""

from config import ANOMALY_SCENARIOS, KPIS

VARIANTS = ("basic", "signatures", "fewshot")

SYSTEM = "You are a 5G RAN engineer. Return only JSON."

# What each failure mode does to the KPIs, relative to the cell's normal.
SIGNATURES = {
    "interference": "SINR and RSRQ clearly down, so CQI and throughput drop and RLF rises. "
                    "RSRP (the cell's own signal power) stays normal.",
    "capacity_overload": "DL and UL PRB utilisation both clearly up, throughput down, handover "
                         "success may dip a little. RSRP and SINR stay normal.",
    "hardware_fault": "RSRP clearly down, RSRQ down, SINR and CQI down, uplink throughput down a lot, "
                      "RLF up. Handover success stays about normal.",
    "coverage_gap": "RSRP clearly down, handover success clearly down, RLF up, SINR down. "
                    "RSRQ stays about normal.",
    "handover_storm": "Handover success clearly down and RLF up while RSRP and SINR stay normal. "
                      "PRB only slightly up.",
}


def _kpi_block(snapshot: dict, z: dict, flagged: list[str]) -> str:
    return "\n".join(
        f"  {k}: {snapshot[k]:.3f} (z={z[k]:.2f})" + (" <- FLAGGED" if k in flagged else "")
        for k in KPIS
    )


def _z_line(z: dict) -> str:
    return ", ".join(f"{k}={z[k]:+.1f}" for k in KPIS)


def _basic_prompt(cell_id: str, score: float, snapshot: dict, z: dict, flagged: list[str]) -> str:
    """The original prompt, unchanged, so earlier results stay comparable."""
    kpi_context = _kpi_block(snapshot, z, flagged)
    modes = "\n".join(f"  {k}: {v}" for k, v in ANOMALY_SCENARIOS.items())
    return (
        "Classify the failure mode of this 5G cell from its KPI readings.\n"
        "z is how far each KPI is from this cell's normal value at this time of day, "
        "in robust standard deviations (negative = lower than normal).\n\n"
        f"Cell: {cell_id} | Anomaly score: {score:.4f} | "
        f"Flagged KPIs: {', '.join(flagged) or 'none'}\n\n"
        f"KPI READINGS:\n{kpi_context}\n\nPOSSIBLE FAILURE MODES:\n{modes}\n\n"
        "Respond with JSON only:\n"
        '{"failure_mode": "<one of the keys above>", "confidence": "<HIGH|MEDIUM|LOW>", '
        '"summary": "<2 sentence technical explanation>"}'
    )


def build_classifier_prompt(cell_id: str, score: float, snapshot: dict, z: dict, flagged: list[str],
                            variant: str = "basic", examples: list[dict] | None = None) -> str:
    """
    examples (fewshot only): [{"z": {...}, "mode": "interference"}, ...]
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; choose from {VARIANTS}")

    if variant == "basic":
        return _basic_prompt(cell_id, score, snapshot, z, flagged)

    modes = "\n".join(f"  {k}: {ANOMALY_SCENARIOS[k]}.\n      Typical pattern: {SIGNATURES[k]}"
                      for k in ANOMALY_SCENARIOS)

    parts = [
        "Classify the failure mode of this 5G cell from its KPI readings.",
        "z is how far each KPI is from this cell's normal value at this time of day, "
        "in robust standard deviations (negative = lower than normal).",
        "",
        f"POSSIBLE FAILURE MODES:\n{modes}",
    ]
    if variant == "fewshot" and examples:
        shots = "\n".join(f"  {_z_line(ex['z'])}\n    -> {ex['mode']}" for ex in examples)
        parts += ["", f"SOLVED EXAMPLES (z-scores, then the correct failure mode):\n{shots}"]
    parts += ["", "Some faults are subtle and only move a few KPIs a little. "
                  "Use LOW or MEDIUM confidence when the pattern is weak or fits more than one mode."]
    parts += [
        "",
        f"Cell: {cell_id} | Anomaly score: {score:.4f} | Flagged KPIs: {', '.join(flagged) or 'none'}",
        "",
        f"KPI READINGS:\n{_kpi_block(snapshot, z, flagged)}",
        "",
        "Respond with JSON only:",
        '{"failure_mode": "<one of the keys above>", "confidence": "<HIGH|MEDIUM|LOW>", '
        '"summary": "<2 sentence technical explanation>"}',
    ]
    return "\n".join(parts)
