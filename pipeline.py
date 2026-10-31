"""
pipeline.py - NetSentinel LangGraph agent pipeline
==================================================

  [kpi_monitor]        robust z-score of the latest reading vs the cell's baseline
       |
  [anomaly_scorer]     Isolation Forest on the latest reading
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
    falls back to a rule-based classifier with LOW confidence.
  - The safety-relevant fields of the action plan (apply_immediately,
    escalate_to_human, actions, parameters) are always set by code.
    The LLM only fills in two free-text explanation fields.
  - A plan is applied automatically only when confidence is HIGH and the
    procedure is marked auto-applicable. Everything else goes to a human.
"""

import json
import re
import uuid
from functools import lru_cache
from typing import Optional, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from config import (
    ANOMALY_SCENARIOS,
    KPIS,
    LLM_TIMEOUT_S,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    ZSCORE_FLAG,
)
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
    root_cause_summary: str
    procedure: dict
    action_plan: dict

    reasoning_log: list[str]


# ── Model and LLM access (patched in tests) ───────────────────────────────

@lru_cache(maxsize=1)
def get_model():
    return load_model_and_scaler()


def call_llm(prompt: str, system: str = "") -> Optional[str]:
    """Return the LLM reply, or None if the LLM is unavailable."""
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_HOST, timeout=LLM_TIMEOUT_S)
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        resp = client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.1, "num_predict": 500},
        )
        return resp["message"]["content"].strip()
    except Exception:
        return None


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
    z-score of the latest reading against all earlier readings of the cell.
    Uses median and MAD so that past anomalies in the baseline don't
    inflate the spread and hide the current one.
    """
    latest = df.iloc[-1]
    baseline = df.iloc[:-1] if len(df) > 1 else df
    snapshot, zscores = {}, {}
    for kpi in KPIS:
        values = baseline[kpi].astype(float).values
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med))) * 1.4826
        spread = max(mad, 0.1 * KPIS[kpi][3], 1e-6)   # floor at 10% of nominal std
        snapshot[kpi] = round(float(latest[kpi]), 3)
        zscores[kpi] = round((float(latest[kpi]) - med) / spread, 2)
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
    """
    if z["prb_dl"] > 2 and z["prb_ul"] > 2:
        return "capacity_overload"
    if z["rsrp"] < -2.5:
        # both lose signal power; hardware faults also degrade RSRQ
        return "hardware_fault" if z["rsrq"] < -1.5 else "coverage_gap"
    if z["sinr"] < -2 and z["rsrq"] < -1.5:
        return "interference"
    if z["ho_sr"] < -2:
        return "handover_storm"
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


def root_cause_classifier_node(state: AgentState) -> AgentState:
    log = list(state["reasoning_log"])
    snapshot, z, flagged = state["kpi_snapshot"], state["kpi_zscores"], state["flagged_kpis"]

    kpi_context = "\n".join(
        f"  {k}: {snapshot[k]:.3f} (z={z[k]:.2f})" + (" <- FLAGGED" if k in flagged else "")
        for k in KPIS
    )
    modes = "\n".join(f"  {k}: {v}" for k, v in ANOMALY_SCENARIOS.items())
    prompt = (
        "Classify the failure mode of this 5G cell from its KPI readings.\n\n"
        f"Cell: {state['cell_id']} | Anomaly score: {state['anomaly_score']:.4f} | "
        f"Flagged KPIs: {', '.join(flagged) or 'none'}\n\n"
        f"KPI READINGS:\n{kpi_context}\n\nPOSSIBLE FAILURE MODES:\n{modes}\n\n"
        "Respond with JSON only:\n"
        '{"failure_mode": "<one of the keys above>", "confidence": "<HIGH|MEDIUM|LOW>", '
        '"summary": "<2 sentence technical explanation>"}'
    )

    log.append("[Root Cause Classifier] Asking LLM for failure mode ...")
    result = validate_diagnosis(parse_json(call_llm(prompt, system="You are a 5G RAN engineer. Return only JSON.")))

    if result:
        mode, confidence, summary = result
        source = "llm"
    else:
        mode = rule_based_classify(z)
        confidence, summary, source = "LOW", "", "rules"
        log.append("[Root Cause Classifier] LLM unavailable or output invalid, using rule-based fallback")

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

    return {**state, "failure_mode": mode, "failure_confidence": confidence,
            "classification_source": source, "root_cause_summary": summary,
            "procedure": procedure, "reasoning_log": log}


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
            "type": "MONITORING", "cell_id": state["cell_id"],
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
    apply_now = state["failure_confidence"] == "HIGH" and bool(proc.get("auto_applicable"))
    return {
        "plan_id": f"ACT-{state['cell_id']}-{uuid.uuid4().hex[:8]}",
        "cell_id": state["cell_id"],
        "severity": state["severity"],
        "failure_mode": state["failure_mode"],
        "confidence": state["failure_confidence"],
        "classification_source": state["classification_source"],
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
    text = parse_json(call_llm(prompt, system="You are a 5G RAN engineer. Return only JSON."))
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
        "failure_mode": "", "failure_confidence": "", "classification_source": "",
        "root_cause_summary": "", "procedure": {}, "action_plan": {},
        "reasoning_log": [],
    })
