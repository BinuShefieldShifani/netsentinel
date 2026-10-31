"""
agent_graph.py — NetSentinel LangGraph Agent Pipeline
======================================================

4 nodes, 1 conditional edge:

  [kpi_monitor]
       │
  [anomaly_scorer]   ← Isolation Forest
       │
   (router)
       ├── anomalous → [root_cause_classifier]  ← LLM
       │                        │
       │               [remediation_planner]    ← LLM + KB
       │
       └── normal    → [routine_monitor]        ← LLM light summary
                               │
                             [END]

What makes this agentic (not just ML + LLM):
  - Conditional routing: the graph takes different paths based on anomaly score
  - The remediation planner checks auto_applicable flag:
    HIGH confidence + auto_applicable → applies fix to simulated config
    LOW confidence or manual required → escalates with full diagnostic
  - Each node has a specific tool/skill — no node does everything
"""

import json
import re
import numpy as np
from typing import TypedDict, Optional

import ollama
from langgraph.graph import StateGraph, START, END

from config import (
    OLLAMA_HOST, OLLAMA_MODEL, KPIS,
    ANOMALY_SCORE_CRITICAL, ANOMALY_SCORE_WARNING,
    ANOMALY_SCENARIOS,
)
from detecter import load_model_and_scaler, score_cell_window
from knowledge_base import retrieve_procedure, format_for_llm, PROCEDURES


# ── State ─────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    cell_id:    str
    cell_df:    object      # pandas DataFrame for this cell

    # KPI monitor outputs
    kpi_snapshot:   dict    # latest KPI values
    kpi_zscores:    dict    # z-score of each KPI
    flagged_kpis:   list[str]
    time_window:    str     # e.g. "last 12 readings (3 hours)"

    # Anomaly scorer outputs
    anomaly_score:  float
    is_anomalous:   bool
    severity:       str     # CRITICAL / WARNING / NORMAL

    # LLM outputs
    failure_mode:       str
    failure_confidence: str
    root_cause_summary: str
    procedure:          dict
    action_plan:        dict

    # Trace
    reasoning_log: list[str]


# ── Ollama ────────────────────────────────────────────────────────────────

_client = ollama.Client(host=OLLAMA_HOST)


def call_llm(prompt: str, system: str = "") -> str:
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = _client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.1, "num_predict": 500},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"[LLM unavailable: {e}]"


def parse_json(text: str) -> dict:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    return {"raw": text}


# ── Node 1: KPI Monitor (pure statistics) ─────────────────────────────────

def kpi_monitor_node(state: AgentState) -> AgentState:
    """
    Analyses the cell's KPI time series.
    Computes z-scores for each KPI over the last 12 readings (3 hours).
    Flags KPIs that are > 2σ from their rolling baseline.
    """
    log = state.get("reasoning_log", [])
    cell_id = state["cell_id"]
    df = state["cell_df"]

    log.append(f"[KPI Monitor] Analysing cell {cell_id} ...")
    log.append(f"[KPI Monitor] {len(df)} readings available")

    # Use last 50 readings for baseline, last 12 for current window
    window = min(12, len(df))
    baseline_df = df.iloc[:-window] if len(df) > window else df
    current_df  = df.iloc[-window:]

    kpi_snapshot = {}
    kpi_zscores  = {}
    flagged      = []

    for kpi in KPIS:
        if kpi not in df.columns:
            continue
        current_val = float(current_df[kpi].mean())
        baseline_mean = float(baseline_df[kpi].mean()) if len(baseline_df) > 0 else current_val
        baseline_std  = float(baseline_df[kpi].std())  if len(baseline_df) > 0 else 1.0
        if baseline_std < 1e-6:
            baseline_std = 1.0

        z = (current_val - baseline_mean) / baseline_std
        kpi_snapshot[kpi] = round(current_val, 3)
        kpi_zscores[kpi]  = round(z, 2)

        _, _, _, direction = KPIS[kpi][1], KPIS[kpi][2], KPIS[kpi][3], KPIS[kpi][4]
        if (direction == "low" and z < -2.0) or (direction == "high" and z > 2.0):
            flagged.append(kpi)

    time_window = f"last {window} readings ({window*15} min)"
    log.append(f"[KPI Monitor] Time window: {time_window}")

    if flagged:
        log.append(f"[KPI Monitor] Flagged KPIs (|z| > 2σ): {', '.join(flagged)}")
    else:
        log.append("[KPI Monitor] All KPIs within normal range")

    return {
        **state,
        "kpi_snapshot":  kpi_snapshot,
        "kpi_zscores":   kpi_zscores,
        "flagged_kpis":  flagged,
        "time_window":   time_window,
        "reasoning_log": log,
    }


# ── Node 2: Anomaly Scorer (Isolation Forest) ─────────────────────────────

def anomaly_scorer_node(state: AgentState) -> AgentState:
    """
    Runs the trained Isolation Forest on the current cell window.
    Assigns severity: CRITICAL / WARNING / NORMAL.
    """
    log = state["reasoning_log"]
    log.append("[Anomaly Scorer] Loading Isolation Forest model ...")

    model, scaler = load_model_and_scaler()
    score, is_anom = score_cell_window(state["cell_df"], model, scaler)

    if score <= ANOMALY_SCORE_CRITICAL:
        severity = "CRITICAL"
    elif score <= ANOMALY_SCORE_WARNING:
        severity = "WARNING"
    else:
        severity = "NORMAL"

    log.append(f"[Anomaly Scorer] Score: {score:.4f}  |  Anomalous: {bool(is_anom)}  |  Severity: {severity}")

    return {
        **state,
        "anomaly_score": score,
        "is_anomalous":  bool(is_anom or len(state["flagged_kpis"]) >= 2),
        "severity":      severity,
        "reasoning_log": log,
    }


# ── Router ────────────────────────────────────────────────────────────────

def route_by_anomaly(state: AgentState) -> str:
    if state["is_anomalous"] or state["severity"] in ("CRITICAL", "WARNING"):
        return "root_cause_classifier"
    return "routine_monitor"


# ── Node 3a: Root Cause Classifier (LLM) ─────────────────────────────────

def root_cause_classifier_node(state: AgentState) -> AgentState:
    """
    LLM analyses the flagged KPI pattern and classifies the failure mode.
    Outputs one of the 5 known O-RAN failure scenarios.
    """
    log = state["reasoning_log"]
    log.append("[Root Cause Classifier] Calling LLM for failure mode classification ...")

    snapshot  = state["kpi_snapshot"]
    zscores   = state["kpi_zscores"]
    flagged   = state["flagged_kpis"]
    score     = state["anomaly_score"]

    kpi_context = "\n".join(
        f"  {kpi}: {snapshot.get(kpi,'?'):.3f} (z={zscores.get(kpi,0):.2f})"
        + (" ← FLAGGED" if kpi in flagged else "")
        for kpi in KPIS
    )

    failure_modes_list = "\n".join(
        f"  {k}: {v}" for k, v in ANOMALY_SCENARIOS.items()
    )

    prompt = (
        f"You are a 5G RAN expert. Classify the failure mode from these KPI readings.\n\n"
        f"Cell: {state['cell_id']}  |  Anomaly score: {score:.4f}  |  "
        f"Flagged KPIs: {', '.join(flagged) or 'none via z-score'}\n\n"
        f"KPI READINGS:\n{kpi_context}\n\n"
        f"POSSIBLE FAILURE MODES:\n{failure_modes_list}\n\n"
        f"Respond with JSON only:\n"
        f'{{"failure_mode": "<one of the keys above>", '
        f'"confidence": "<HIGH|MEDIUM|LOW>", '
        f'"primary_indicators": ["<kpi> is <observation>"], '
        f'"summary": "<2 sentence technical explanation>"}}'
    )

    response  = call_llm(prompt, system="You are an expert 5G network engineer. Return only JSON.")
    diagnosis = parse_json(response)

    failure_mode = diagnosis.get("failure_mode", "interference")
    confidence   = diagnosis.get("confidence", "MEDIUM")
    summary      = diagnosis.get("summary", "")

    # Fallback if LLM returns unknown mode
    if failure_mode not in ANOMALY_SCENARIOS:
        failure_mode = _heuristic_classify(flagged, snapshot)
        confidence   = "MEDIUM"

    log.append(f"[Root Cause Classifier] Mode: {failure_mode}  |  Confidence: {confidence}")
    if summary:
        log.append(f"[Root Cause Classifier] {summary}")

    proc = retrieve_procedure(failure_mode)
    log.append(f"[Root Cause Classifier] Procedure retrieved: {proc.id if proc else 'none'}")

    return {
        **state,
        "failure_mode":       failure_mode,
        "failure_confidence": confidence,
        "root_cause_summary": summary,
        "procedure":          {
            "id":            proc.id if proc else "N/A",
            "title":         proc.title if proc else "Manual investigation required",
            "auto_applicable": proc.auto_applicable if proc else False,
            "severity":      proc.severity if proc else "HIGH",
            "actions":       proc.actions if proc else [],
            "parameters":    proc.parameters if proc else [],
        } if proc else {},
        "reasoning_log": log,
    }


def _heuristic_classify(flagged: list, snapshot: dict) -> str:
    """Rule-based fallback classification when LLM is unavailable."""
    if "sinr" in flagged and "rsrq" in flagged:
        return "interference"
    if "prb_dl" in flagged or "prb_ul" in flagged:
        return "capacity_overload"
    if "rsrp" in flagged and snapshot.get("rsrp", -75) < -100:
        return "coverage_gap"
    if "ho_sr" in flagged or "rlf_rate" in flagged:
        return "handover_storm"
    return "hardware_fault"


# ── Node 3b: Routine Monitor (LLM light) ─────────────────────────────────

def routine_monitor_node(state: AgentState) -> AgentState:
    """Healthy cell — generate a brief monitoring summary."""
    log = state["reasoning_log"]
    log.append("[Routine Monitor] Cell is healthy — generating monitoring summary ...")

    best_kpi  = min(state["kpi_zscores"], key=lambda k: abs(state["kpi_zscores"][k]))
    watch_kpi = max(state["kpi_zscores"], key=lambda k: abs(state["kpi_zscores"][k]))

    return {
        **state,
        "failure_mode":       "none",
        "failure_confidence": "HIGH",
        "root_cause_summary": "Cell operating within normal parameters.",
        "procedure": {},
        "action_plan": {
            "type":             "MONITORING",
            "cell_id":          state["cell_id"],
            "anomaly_score":    round(state["anomaly_score"], 4),
            "severity":         "NORMAL",
            "recommendation":   "Continue scheduled monitoring. No action required.",
            "watch_kpi":        watch_kpi,
            "next_check_in":    "15 minutes",
        },
        "reasoning_log": log,
    }


# ── Node 4: Remediation Planner (LLM) ────────────────────────────────────

def remediation_planner_node(state: AgentState) -> AgentState:
    """
    Generates the final action plan.
    HIGH confidence + auto_applicable → includes "apply_immediately: true"
    LOW confidence or manual proc     → escalates to human operator
    """
    log = state["reasoning_log"]
    log.append("[Remediation Planner] Generating action plan ...")

    proc       = state.get("procedure", {})
    confidence = state["failure_confidence"]
    auto_ok    = proc.get("auto_applicable", False)
    apply_now  = (confidence == "HIGH" and auto_ok)

    proc_obj = retrieve_procedure(state["failure_mode"])
    proc_context = format_for_llm(proc_obj) if proc_obj else "No specific procedure found."

    import random
    plan_id = f"ACT-{state['cell_id']}-{random.randint(10000,99999)}"

    prompt = (
        f"Fill in the two <FILL> fields in this action plan JSON. Return JSON only.\n\n"
        f'{{\n'
        f'  "plan_id": "{plan_id}",\n'
        f'  "cell_id": "{state["cell_id"]}",\n'
        f'  "severity": "{state["severity"]}",\n'
        f'  "failure_mode": "{state["failure_mode"]}",\n'
        f'  "confidence": "{confidence}",\n'
        f'  "apply_immediately": {str(apply_now).lower()},\n'
        f'  "anomaly_score": {state["anomaly_score"]:.4f},\n'
        f'  "actions": {json.dumps(proc.get("actions", [])[:4])},\n'
        f'  "parameters_to_tune": {json.dumps(proc.get("parameters", []))},\n'
        f'  "escalate_to_human": {str(not auto_ok).lower()},\n'
        f'  "technical_rationale": <FILL: one sentence explaining the root cause>,\n'
        f'  "expected_outcome": <FILL: one sentence on expected improvement>\n'
        f'}}\n\n'
        f"Context:\n"
        f"Flagged KPIs: {', '.join(state['flagged_kpis'])}\n"
        f"Procedure: {proc_context[:400]}"
    )

    response  = call_llm(prompt, system="You are a 5G network engineer. Return only the completed JSON.")
    plan      = parse_json(response)

    if "raw" in plan:
        plan = {
            "plan_id":             plan_id,
            "cell_id":             state["cell_id"],
            "severity":            state["severity"],
            "failure_mode":        state["failure_mode"],
            "confidence":          confidence,
            "apply_immediately":   apply_now,
            "anomaly_score":       round(state["anomaly_score"], 4),
            "actions":             proc.get("actions", [])[:4],
            "parameters_to_tune":  proc.get("parameters", []),
            "escalate_to_human":   not auto_ok,
            "technical_rationale": (
                f"Cell {state['cell_id']} shows {state['failure_mode']} pattern "
                f"with anomaly score {state['anomaly_score']:.4f}. "
                f"Flagged KPIs: {', '.join(state['flagged_kpis'])}."
            ),
            "expected_outcome": proc_obj.expected_improvement if proc_obj else "Improvement expected after parameter adjustment.",
        }

    log.append(f"[Remediation Planner] Plan: {plan.get('plan_id')}")
    log.append(f"[Remediation Planner] Apply immediately: {apply_now}  |  Escalate: {not auto_ok}")

    return {**state, "action_plan": plan, "reasoning_log": log}


# ── Build Graph ───────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("kpi_monitor",           kpi_monitor_node)
    g.add_node("anomaly_scorer",         anomaly_scorer_node)
    g.add_node("root_cause_classifier",  root_cause_classifier_node)
    g.add_node("routine_monitor",        routine_monitor_node)
    g.add_node("remediation_planner",    remediation_planner_node)

    g.add_edge(START, "kpi_monitor")
    g.add_edge("kpi_monitor", "anomaly_scorer")

    g.add_conditional_edges(
        "anomaly_scorer",
        route_by_anomaly,
        {
            "root_cause_classifier": "root_cause_classifier",
            "routine_monitor":       "routine_monitor",
        }
    )
    g.add_edge("root_cause_classifier", "remediation_planner")
    g.add_edge("remediation_planner",   END)
    g.add_edge("routine_monitor",       END)

    return g.compile()


def run_analysis(cell_id: str, cell_df) -> AgentState:
    graph = build_graph()
    return graph.invoke({
        "cell_id":          cell_id,
        "cell_df":          cell_df,
        "kpi_snapshot":     {},
        "kpi_zscores":      {},
        "flagged_kpis":     [],
        "time_window":      "",
        "anomaly_score":    0.0,
        "is_anomalous":     False,
        "severity":         "NORMAL",
        "failure_mode":     "",
        "failure_confidence": "",
        "root_cause_summary": "",
        "procedure":        {},
        "action_plan":      {},
        "reasoning_log":    [],
    })
