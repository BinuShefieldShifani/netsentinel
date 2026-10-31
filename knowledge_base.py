"""
knowledge_base.py - 5G RAN troubleshooting knowledge base
=========================================================
Maps each failure mode to a remediation procedure. In a real deployment
this would come from the operator's OSS runbooks. Parameter names here are
illustrative, not taken from any specific vendor's OSS.

Auto-apply policy: only the two procedures that match standard SON
functions operators commonly automate are auto-applicable:
  capacity overload -> Mobility Load Balancing (MLB)
  handover storm    -> Mobility Robustness Optimisation (MRO)
Everything else needs a human, because the cause has to be confirmed first
(an interference source, a hardware alarm, a neighbour outage).

References:
  3GPP TS 38.300  NR overall description (incl. SON, Xn)
  3GPP TS 38.331  NR RRC (measurement events A3, time-to-trigger, hysteresis)
  3GPP TS 38.423  XnAP (signalling between NR base stations)
  3GPP TS 28.313  SON for 5G networks
"""

from dataclasses import dataclass


@dataclass
class RemediationProcedure:
    id:              str
    failure_mode:    str
    title:           str
    severity:        str          # CRITICAL / HIGH / MEDIUM
    auto_applicable: bool         # can be applied without human approval
    parameters:      list[str]    # illustrative OSS parameter names to tune
    actions:         list[str]
    expected_improvement: str


PROCEDURES: list[RemediationProcedure] = [

    RemediationProcedure(
        id="REM-001",
        failure_mode="interference",
        title="Interference: locate the source before changing the cell",
        severity="HIGH",
        auto_applicable=False,
        parameters=["neighbourTxPower", "neighbourElectricalTilt", "interCellInterferenceCoordination"],
        actions=[
            "Check whether neighbour cells changed power, tilt or load at the same time (internal interference).",
            "Look at uplink noise rise per PRB; a flat rise across the band points to an external source.",
            "If a neighbour is the source: coordinate over Xn, e.g. reduce that neighbour's power or adjust its tilt.",
            "If the source is external: request a spectrum scan; report persistent external interference to the regulator.",
            "Escalate to RF optimisation if SINR stays below 0 dB for more than 30 minutes.",
        ],
        expected_improvement="SINR back to the cell's normal level once the source is removed or coordinated.",
    ),

    RemediationProcedure(
        id="REM-002",
        failure_mode="capacity_overload",
        title="Capacity overload: mobility load balancing (MLB)",
        severity="CRITICAL",
        auto_applicable=True,
        parameters=["cellIndividualOffset", "mlbThreshold", "admissionControlMaxPrb"],
        actions=[
            "Trigger MLB: lower the cell individual offset (CIO) towards less loaded neighbours.",
            "Check that the target neighbours have spare PRB capacity before moving users.",
            "Tighten admission control for new low-priority sessions while PRB stays above 90%.",
            "Use carrier aggregation or another band if the site has a second carrier.",
            "Raise a capacity planning ticket if PRB stays above 90% for 3+ intervals on most days.",
        ],
        expected_improvement="PRB utilisation below 80% within about 3 intervals.",
    ),

    RemediationProcedure(
        id="REM-003",
        failure_mode="hardware_fault",
        title="Hardware fault: confirm with alarms, then dispatch",
        severity="CRITICAL",
        auto_applicable=False,
        parameters=["cellAdminState", "neighbourCoverageCompensation"],
        actions=[
            "Check radio unit alarms: VSWR, power amplifier, temperature, fronthaul link.",
            "Compare all sectors of the site; one sector down points to antenna or feeder, all down to baseband or power.",
            "Create a field dispatch ticket with the alarm details.",
            "If the cell is unusable, lock it and let neighbours compensate coverage.",
        ],
        expected_improvement="No improvement until repaired; neighbour compensation limits the impact.",
    ),

    RemediationProcedure(
        id="REM-004",
        failure_mode="coverage_gap",
        title="Coverage gap: check neighbours, then adjust tilt",
        severity="HIGH",
        auto_applicable=False,
        parameters=["antennaElectricalTilt", "txPower", "neighbourRelations"],
        actions=[
            "Check whether a neighbour cell is down or degraded; that is the most common cause of a sudden gap.",
            "Audit neighbour relations (ANR) so cell-edge users have a handover target.",
            "Reduce electrical downtilt by 1-2 degrees to extend coverage, within interference limits.",
            "Increase transmit power within the licensed EIRP limit if tilt is not enough.",
            "Schedule a drive test if the gap lasts more than 24 hours.",
        ],
        expected_improvement="RSRP at the cell edge back to normal and handover success above 97%.",
    ),

    RemediationProcedure(
        id="REM-005",
        failure_mode="handover_storm",
        title="Handover storm: mobility robustness optimisation (MRO)",
        severity="HIGH",
        auto_applicable=True,
        parameters=["a3Offset", "timeToTrigger", "hysteresis"],
        actions=[
            "Increase A3 offset by 1-2 dB on the affected neighbour pairs.",
            "Increase time-to-trigger, e.g. from 160 ms to 320 ms, to filter short fluctuations.",
            "Increase hysteresis to reduce ping-pong handovers.",
            "Remove neighbour relations that cause unnecessary handovers.",
        ],
        expected_improvement="Handover success above 97% and RLF rate back to normal within about 5 intervals.",
    ),
]


def retrieve_procedure(failure_mode: str) -> RemediationProcedure | None:
    for proc in PROCEDURES:
        if proc.failure_mode == failure_mode:
            return proc
    return None


def format_for_llm(proc: RemediationProcedure) -> str:
    actions_text = "\n".join(proc.actions)
    params_text  = ", ".join(proc.parameters)
    return (
        f"Procedure {proc.id}: {proc.title}\n"
        f"Severity: {proc.severity} | Auto-applicable: {proc.auto_applicable}\n"
        f"Parameters: {params_text}\n"
        f"Actions:\n{actions_text}\n"
        f"Expected improvement: {proc.expected_improvement}"
    )
