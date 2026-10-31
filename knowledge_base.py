"""
knowledge_base.py - 5G RAN troubleshooting knowledge base
=========================================================
Maps each failure mode to a remediation procedure. In a real deployment
this would come from the operator's OSS runbooks. Parameter names here are
illustrative, not taken from any specific vendor's OSS.

References: 3GPP TS 38.300 (NR overall description), TS 36.423 (X2 interface).
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
        title="Interference Mitigation — Power and Tilt Adjustment",
        severity="HIGH",
        auto_applicable=True,
        parameters=["txPowerAdjustment", "antennaElectricalTilt", "icic_enable"],
        actions=[
            "Reduce Tx power by 2 dB on serving cell to limit interference footprint.",
            "Increase electrical antenna downtilt by 2° to reduce side-lobe interference.",
            "Enable ICIC (Inter-Cell Interference Coordination) on X2 interface.",
            "Trigger neighbour cell measurement report for interference source identification.",
            "If SINR remains < 0 dB after 30 min — escalate to RF optimisation team.",
        ],
        expected_improvement="SINR improvement of 3–6 dB within 2 measurement intervals.",
    ),

    RemediationProcedure(
        id="REM-002",
        failure_mode="capacity_overload",
        title="Capacity Overload — Load Balancing and Admission Control",
        severity="CRITICAL",
        auto_applicable=True,
        parameters=["loadBalancingThreshold", "admissionControlMaxPrb", "hoA3Offset"],
        actions=[
            "Trigger MLB (Mobility Load Balancing) — offload UEs to adjacent cells.",
            "Lower A3 handover offset by 2 dB to push border UEs to neighbours earlier.",
            "Tighten admission control: cap new session PRB allocation at 85%.",
            "Activate CA (Carrier Aggregation) if secondary carrier available.",
            "Alert capacity planning team if PRB > 90% persists for > 3 consecutive intervals.",
        ],
        expected_improvement="PRB utilisation reduced to < 80% within 3 measurement intervals.",
    ),

    RemediationProcedure(
        id="REM-003",
        failure_mode="hardware_fault",
        title="Hardware Fault — Field Engineer Dispatch",
        severity="CRITICAL",
        auto_applicable=False,
        parameters=["cellAdminState", "alarmSuppression"],
        actions=[
            "GROUND CELL: set cellAdminState to LOCKED to prevent further UE attachment.",
            "Trigger neighbour cells to increase coverage area (coverage extension).",
            "Create field engineer dispatch ticket — hardware inspection required.",
            "Check RU (Radio Unit) alarm logs: temperature, VSWR, power amplifier status.",
            "Validate antenna connector and feeder cable integrity.",
            "If RSRP < -110 dBm on all sectors — suspect complete antenna failure.",
        ],
        expected_improvement="N/A until hardware repaired. Neighbour coverage extension active.",
    ),

    RemediationProcedure(
        id="REM-004",
        failure_mode="coverage_gap",
        title="Coverage Gap — Remote Electrical Tilt and Power Boost",
        severity="HIGH",
        auto_applicable=True,
        parameters=["antennaElectricalTilt", "txPowerAdjustment", "coverageExtension"],
        actions=[
            "Reduce electrical downtilt by 3° to extend coverage radius.",
            "Increase Tx power by 2 dB (within regulatory EIRP limits).",
            "Enable coverage extension for cell-edge UEs (CE level +5 dB).",
            "Trigger neighbour list audit — add missing neighbours for seamless HO.",
            "Schedule drive test for coverage validation if gap persists > 24h.",
        ],
        expected_improvement="RSRP improvement of 4–8 dBm at cell edge within 1 interval.",
    ),

    RemediationProcedure(
        id="REM-005",
        failure_mode="handover_storm",
        title="Handover Storm — HO Parameter Optimisation",
        severity="HIGH",
        auto_applicable=True,
        parameters=["hoA3Offset", "hoTimeToTrigger", "hoHysteresis", "pingPongTimer"],
        actions=[
            "Increase A3 handover offset by 3 dB to raise HO trigger threshold.",
            "Increase Time-to-Trigger (TTT) from 160ms to 320ms to filter transient fluctuations.",
            "Set hysteresis to 3 dB to prevent ping-pong between cells.",
            "Enable ping-pong prevention timer (minimum 60s between reverse HOs).",
            "Audit neighbour cell list — remove spurious neighbours causing unnecessary HOs.",
        ],
        expected_improvement="Handover success rate > 97% and RLF rate < 0.5% within 5 intervals.",
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
