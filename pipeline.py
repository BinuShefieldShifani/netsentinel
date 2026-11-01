"""
pipeline.py - NetSentinel LangGraph agent pipeline
==================================================

  [kpi_monitor]        robust z-score of the latest reading vs the cell's time-of-day baseline
       |
  [anomaly_scorer]     Isolation Forest on the latest reading (raw KPIs + baseline deviation)
       |
   (router)
       |-- anomalous -> [root_cause_classifier]  LLM, validated, rule-based fallback
       |                        |
       |               [remediation_planner]     plan built in code, LLM writes text only
       |
       '-- normal    -> [routine_monitor]

How the pipeline avoids trusting the LLM blindly:
  - The LLM's failure mode and confidence are checked against a fixed set
    of allowed values. Anything else (bad JSON, unknown mode, LLM offline)
    is treated as no answer.
  - By default the rules decide (policy rules_first). The LLM is only used
    when the rules can't name the fault, and its confidence only counts
    when it agrees with the rules.
  - The safety-relevant fields of the action plan (apply_immediately,
    escalate_to_human, actions, parameters) are always set by code.
    The LLM only fills in two free-text explanation fields.
  - A fix is eligible for automatic application only when the confidence is
    HIGH, the procedure is auto-applicable, the rules independently agree,
    and the anomaly model rates the reading CRITICAL. Automatic application
    itself is switched off by default (AUTO_APPLY_ENABLED); eligible fixes
    still go to a person.
"""

import json
import re
import uuid
from functools import lru_cache
from typing import Optional, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

import llm
import prompts
from config import (
    ANOMALY_SCENARIOS,
    AUTO_APPLY_ENABLED,
    AUTO_APPLY_REQUIRES_CRITICAL,
    DECISION_POLICY,
    KPIS,
    PROMPT_VARIANT,
    ZSCORE_FLAG,
)
from features import RESIDUAL_COLUMNS, build_features
from knowledge_base import format_for_llm, retrieve_procedure
from model_trainer import load_model_and_scaler, score_cell_window, severity_from_score

VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
UNCLASSIFIED = "unclassified"


# ── State ─────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    cell_id: str
    cell_df: object                 # pandas DataFrame, oldest -> newest

    kpi_snapshot: dict              # latest KPI values
    kpi_zscores: dict               # robust z-score of the latest reading
    flagged_kpis: list[str]
    time_window: str

    anomaly_score: float
    is_anomalous: bool
    severity: str                   # CRITICAL / WARNING / NORMAL

    failure_mode: str
    failure_confidence: str
    classification_source: str      # "llm" or "rules"
    rules_mode: str                 # what the rule-based classifier says, kept for the cross-check
    root_cause_summary: str
    procedure: dict
    action_plan: dict

    reasoning_log: list[str]


# ── Model and LLM access (patched in tests) ───────────────────────────────

@lru_cache(maxsize=1)
def get_model():
    return load_model_and_scaler()


def call_llm(prompt: str, system: str = "") -> Optional[str]:
    """Reply from the configured LLM (see llm.py), or None if there isn't one."""
    return llm.complete(prompt, system)


def parse_json(text: Optional[str]) -> Optional[dict]:
    """Extract a JSON object from an LLM reply. None if there isn't one."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1]):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


# ── Node 1: KPI monitor ───────────────────────────────────────────────────

def robust_zscores(df) -> tuple[dict, dict]:
    """
    How unusual the latest reading is for this cell, per KPI.
    Each KPI's deviation from the cell's time-of-day baseline (features.py)
    is divided by the typical size of that deviation in the cell's history.
    Median and MAD are used so past faults in the history don't inflate the
    spread and hide the current one.
    """
    ordered = df.sort_values("timestamp") if "timestamp" in df.columns else df
    residuals = build_features(ordered)[RESIDUAL_COLUMNS].to_numpy()
    latest, history = residuals[-1], (residuals[:-1] if len(residuals) > 1 else residuals)
    med = np.median(history, axis=0)
    mad = np.median(np.abs(history - med), axis=0) * 1.4826
    last_row = ordered.iloc[-1]
    snapshot, zscores = {}, {}
    for j, kpi in enumerate(KPIS):
        spread = max(float(mad[j]), 0.1 * KPIS[kpi][3], 1e-6)   # floor at 10% of nominal std
        snapshot[kpi] = round(float(last_row[kpi]), 3)
        zscores[kpi] = round(float((latest[j] - med[j]) / spread), 2)
    return snapshot, zscores


def flag_kpis(zscores: dict) -> list[str]:
    flagged = []
    for kpi, z in zscores.items():
        direction = KPIS[kpi][4]
        if (direction == "low" and z < -ZSCORE_FLAG) or (direction == "high" and z > ZSCORE_FLAG):
            flagged.append(kpi)
    return flagged


def kpi_monitor_node(state: AgentState) -> AgentState:
    log = list(state.get("reasoning_log", []))
    df = state["cell_df"]
    log.append(f"[KPI Monitor] Analysing cell {state['cell_id']}, {len(df)} readings")

    snapshot, zscores = robust_zscores(df)
    flagged = flag_kpis(zscores)
    time_window = f"latest reading vs previous {len(df) - 1} readings"

    if flagged:
        log.append(f"[KPI Monitor] Flagged KPIs (> {ZSCORE_FLAG} sigma): {', '.join(flagged)}")
    else:
        log.append("[KPI Monitor] All KPIs within normal range")

    return {**state, "kpi_snapshot": snapshot, "kpi_zscores": zscores,
            "flagged_kpis": flagged, "time_window": time_window,
            "reasoning_log": log}


# ── Node 2: Anomaly scorer ────────────────────────────────────────────────

def anomaly_scorer_node(state: AgentState) -> AgentState:
    log = list(state["reasoning_log"])
    model, scaler = get_model()
    score, model_says_anomaly = score_cell_window(state["cell_df"], model, scaler)
    severity = severity_from_score(score)
    is_anomalous = bool(model_says_anomaly or len(state["flagged_kpis"]) >= 2)
    if is_anomalous and severity == "NORMAL":
        severity = "WARNING"   # flagged by the KPI monitor even if the model is unsure

    log.append(f"[Anomaly Scorer] Score: {score:.4f} | Model anomaly: {model_says_anomaly} "
               f"| Severity: {severity}")
    return {**state, "anomaly_score": score, "is_anomalous": is_anomalous,
            "severity": severity, "reasoning_log": log}


def route_by_anomaly(state: AgentState) -> str:
    return "root_cause_classifier" if state["is_anomalous"] else "routine_monitor"


# ── Node 3a: Root cause classifier ────────────────────────────────────────

def rule_based_classify(z: dict) -> str:
    """
    Signature rules over the robust z-scores of the latest reading.
    Used when the LLM is unavailable or returns something invalid.

      signal power down + handovers failing  -> coverage gap
      signal power down, handovers fine      -> hardware fault
      DL and UL PRB both saturating          -> capacity overload
      handovers failing, signal fine         -> handover storm
      SINR and RSRQ down, signal fine        -> interference
    """
    if z["rsrp"] < -3:
        return "coverage_gap" if z["ho_sr"] < -3 else "hardware_fault"
    if z["prb_dl"] > 2.5 and z["prb_ul"] > 2:
        return "capacity_overload"
    if z["ho_sr"] < -4:
        return "handover_storm"
    if z["sinr"] < -2 and z["rsrq"] < -1.5:
        return "interference"
    return UNCLASSIFIED


def validate_diagnosis(diagnosis: Optional[dict]) -> Optional[tuple[str, str, str]]:
    """Return (mode, confidence, summary) if the LLM output is usable, else None."""
    if not diagnosis:
        return None
    mode = str(diagnosis.get("failure_mode", "")).strip().lower()
    conf = str(diagnosis.get("confidence", "")).strip().upper()
    if mode not in ANOMALY_SCENARIOS or conf not in VALID_CONFIDENCE:
        return None
    return mode, conf, str(diagnosis.get("summary", ""))[:500]


CLASSIFIER_SYSTEM = prompts.SYSTEM
POLICIES = ("llm_first", "rules_first")


def classifier_prompt(cell_id: str, score: float, snapshot: dict, z: dict, flagged: list[str],
                      variant: Optional[str] = None, examples: Optional[list[dict]] = None) -> str:
    return prompts.build_classifier_prompt(cell_id, score, snapshot, z, flagged,
                                           variant or PROMPT_VARIANT, examples)


def diagnose(prompt: str) -> tuple[Optional[tuple[str, str, str]], Optional[str]]:
    """Ask the LLM. Returns (validated diagnosis or None, raw reply or None if the LLM was unreachable)."""
    raw = call_llm(prompt, system=CLASSIFIER_SYSTEM)
    return validate_diagnosis(parse_json(raw)), raw


def decide(llm_result: Optional[tuple[str, str, str]], z: dict,
           policy: Optional[str] = None) -> tuple[str, str, str, str]:
    """
    Final (mode, confidence, summary, source).

    llm_first    the LLM if its reply is valid, otherwise the rules at LOW confidence
    rules_first  the rules, unless they say "unclassified"; then the LLM. The LLM's
                 confidence is only kept when it names the same mode as the rules.
    """
    policy = policy or DECISION_POLICY
    if policy not in POLICIES:
        raise ValueError(f"unknown decision policy {policy!r}; choose from {POLICIES}")
    rules_mode = rule_based_classify(z)

    if policy == "llm_first":
        if llm_result:
            mode, confidence, summary = llm_result
            return mode, confidence, summary, "llm"
        return rules_mode, "LOW", "", "rules"

    # rules_first
    if rules_mode != UNCLASSIFIED:
        if llm_result and llm_result[0] == rules_mode:
            return rules_mode, llm_result[1], llm_result[2], "rules+llm"
        return rules_mode, "LOW", llm_result[2] if llm_result else "", "rules"
    if llm_result:
        mode, confidence, summary = llm_result
        return mode, confidence, summary, "llm"
    return UNCLASSIFIED, "LOW", "", "rules"


def root_cause_classifier_node(state: AgentState) -> AgentState:
    log = list(state["reasoning_log"])
    prompt = classifier_prompt(state["cell_id"], state["anomaly_score"], state["kpi_snapshot"],
                               state["kpi_zscores"], state["flagged_kpis"])
    log.append("[Root Cause Classifier] Asking LLM for failure mode ...")
    llm_result, _ = diagnose(prompt)
    mode, confidence, summary, source = decide(llm_result, state["kpi_zscores"])
    if llm_result is None:
        log.append("[Root Cause Classifier] LLM unavailable or output invalid")
    if source.startswith("rules"):
        log.append(f"[Root Cause Classifier] Decided by the rules (policy: {DECISION_POLICY})")
    log.append(f"[Root Cause Classifier] Mode: {mode} | Confidence: {confidence} | Source: {source}")

    proc = retrieve_procedure(mode)
    procedure = {
        "id": proc.id, "title": proc.title, "auto_applicable": proc.auto_applicable,
        "severity": proc.severity, "actions": proc.actions, "parameters": proc.parameters,
    } if proc else {
        "id": "N/A", "title": "Manual investigation required", "auto_applicable": False,
        "severity": "HIGH", "actions": ["Escalate to RAN engineer for manual diagnosis."],
        "parameters": [],
    }
    log.append(f"[Root Cause Classifier] Procedure: {procedure['id']}")

    rules_mode = rule_based_classify(state["kpi_zscores"])
    if rules_mode != mode:
        log.append(f"[Root Cause Classifier] Rules disagree ({rules_mode}), so no automatic action")

    return {**state, "failure_mode": mode, "failure_confidence": confidence,
            "classification_source": source, "rules_mode": rules_mode,
            "root_cause_summary": summary, "procedure": procedure, "reasoning_log": log}


def would_auto_apply(mode: str, confidence: str, rules_mode: str, severity: str = "CRITICAL",
                     require_critical: Optional[bool] = None) -> bool:
    """
    HIGH confidence + auto-applicable procedure + independent agreement from the rules,
    and, if required (NETSENTINEL_AUTO_CRITICAL=1), a CRITICAL reading.
    """
    if require_critical is None:
        require_critical = AUTO_APPLY_REQUIRES_CRITICAL
    proc = retrieve_procedure(mode)
    ok = confidence == "HIGH" and bool(proc and proc.auto_applicable) and rules_mode == mode
    return ok and (severity == "CRITICAL" or not require_critical)


# ── Node 3b: Routine monitor ──────────────────────────────────────────────

def routine_monitor_node(state: AgentState) -> AgentState:
    log = list(state["reasoning_log"])
    log.append("[Routine Monitor] Cell is healthy, no action needed")
    watch_kpi = max(state["kpi_zscores"], key=lambda k: abs(state["kpi_zscores"][k]))
    return {
        **state,
        "failure_mode": "none", "failure_confidence": "HIGH",
        "classification_source": "rules",
        "root_cause_summary": "Cell operating within normal parameters.",
        "procedure": {},
        "action_plan": {
            "type": "MONITORING", "plan_id": f"MON-{state['cell_id']}-{uuid.uuid4().hex[:8]}",
            "cell_id": state["cell_id"],
            "failure_mode": "none", "confidence": "HIGH", "classification_source": "rules",
            "apply_immediately": False, "escalate_to_human": False,
            "anomaly_score": round(state["anomaly_score"], 4), "severity": "NORMAL",
            "recommendation": "Continue scheduled monitoring. No action required.",
            "watch_kpi": watch_kpi, "next_check_in": "15 minutes",
        },
        "reasoning_log": log,
    }


# ── Node 4: Remediation planner ───────────────────────────────────────────

def build_action_plan(state: AgentState) -> dict:
    """All decision fields come from code, never from the LLM."""
    proc = state["procedure"]
    eligible = would_auto_apply(state["failure_mode"], state["failure_confidence"], state["rules_mode"],
                                state["severity"])
    apply_now = eligible and AUTO_APPLY_ENABLED
    return {
        "plan_id": f"ACT-{state['cell_id']}-{uuid.uuid4().hex[:8]}",
        "cell_id": state["cell_id"],
        "severity": state["severity"],
        "failure_mode": state["failure_mode"],
        "confidence": state["failure_confidence"],
        "classification_source": state["classification_source"],
        "rules_agree": state["rules_mode"] == state["failure_mode"],
        "auto_apply_eligible": eligible,
        "apply_immediately": apply_now,
        "escalate_to_human": not apply_now,
        "anomaly_score": round(state["anomaly_score"], 4),
        "actions": list(proc.get("actions", []))[:4],
        "parameters_to_tune": list(proc.get("parameters", [])),
        "technical_rationale": (
            f"Cell {state['cell_id']} matches the {state['failure_mode']} pattern "
            f"(anomaly score {state['anomaly_score']:.4f}). "
            f"Flagged KPIs: {', '.join(state['flagged_kpis']) or 'none'}."
        ),
        "expected_outcome": "",
    }


def remediation_planner_node(state: AgentState) -> AgentState:
    log = list(state["reasoning_log"])
    plan = build_action_plan(state)

    proc_obj = retrieve_procedure(state["failure_mode"])
    if proc_obj:
        plan["expected_outcome"] = proc_obj.expected_improvement

    # The LLM may only improve the two explanation fields.
    prompt = (
        "Write two short sentences for a RAN engineer. Return JSON only: "
        '{"technical_rationale": "...", "expected_outcome": "..."}\n\n'
        f"Failure mode: {state['failure_mode']}\n"
        f"Flagged KPIs: {', '.join(state['flagged_kpis']) or 'none'}\n"
        f"Procedure:\n{format_for_llm(proc_obj)[:400] if proc_obj else 'none'}"
    )
    text = parse_json(call_llm(prompt, system=CLASSIFIER_SYSTEM))
    if text:
        for field in ("technical_rationale", "expected_outcome"):
            value = text.get(field)
            if isinstance(value, str) and value.strip():
                plan[field] = value.strip()[:500]

    log.append(f"[Remediation Planner] Plan {plan['plan_id']} | Apply immediately: "
               f"{plan['apply_immediately']} | Escalate: {plan['escalate_to_human']}")
    return {**state, "action_plan": plan, "reasoning_log": log}


# ── Graph ─────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("kpi_monitor", kpi_monitor_node)
    g.add_node("anomaly_scorer", anomaly_scorer_node)
    g.add_node("root_cause_classifier", root_cause_classifier_node)
    g.add_node("routine_monitor", routine_monitor_node)
    g.add_node("remediation_planner", remediation_planner_node)

    g.add_edge(START, "kpi_monitor")
    g.add_edge("kpi_monitor", "anomaly_scorer")
    g.add_conditional_edges("anomaly_scorer", route_by_anomaly, {
        "root_cause_classifier": "root_cause_classifier",
        "routine_monitor": "routine_monitor",
    })
    g.add_edge("root_cause_classifier", "remediation_planner")
    g.add_edge("remediation_planner", END)
    g.add_edge("routine_monitor", END)
    return g.compile()


def run_analysis(cell_id: str, cell_df) -> AgentState:
    df = cell_df.sort_values("timestamp").reset_index(drop=True)
    return build_graph().invoke({
        "cell_id": cell_id, "cell_df": df,
        "kpi_snapshot": {}, "kpi_zscores": {}, "flagged_kpis": [], "time_window": "",
        "anomaly_score": 0.0, "is_anomalous": False, "severity": "NORMAL",
        "failure_mode": "", "failure_confidence": "", "classification_source": "", "rules_mode": "",
        "root_cause_summary": "", "procedure": {}, "action_plan": {},
        "reasoning_log": [],
    })
