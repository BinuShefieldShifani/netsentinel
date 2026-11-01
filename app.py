"""
app.py - NetSentinel Streamlit dashboard
========================================
Run with:  streamlit run app.py
First run: python model_trainer.py   (generates data and trains the model)

The sidebar works like a time machine: pick a cell and a "current time".
The agent only sees readings up to that point, so you can replay a fault
as it happens.
"""

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import (
    ANOMALY_SCORE_CRITICAL,
    ANOMALY_SCORE_WARNING,
    AUTO_APPLY_ENABLED,
    AUTO_APPLY_REQUIRES_CRITICAL,
    DECISION_POLICY,
    KPIS,
    METRICS_PATH,
    MODEL_PATH,
    PROMPT_VARIANT,
)
from data_generator import generate_dataset
from evaluation import alarms_from_flags
from llm import describe_backend

st.set_page_config(page_title="NetSentinel · 5G RAN monitor", page_icon="📡", layout="wide")

FAULT_COLOUR = "rgba(226, 75, 74, 0.13)"
SURGE_COLOUR = "rgba(128, 128, 128, 0.15)"
SEVERITY_COLOURS = {"CRITICAL": "#E24B4A", "WARNING": "#EF9F27", "NORMAL": "#1D9E75"}
MIN_HISTORY = 50

st.markdown("""
<div style="padding-bottom:8px;">
  <h1 style="margin:0;font-size:26px;">NetSentinel</h1>
  <p style="margin:0;color:gray;font-size:14px;">
    5G RAN anomaly detection with a guarded LLM agent ·
    Isolation Forest + LangGraph · synthetic 3GPP TS 28.552 KPIs
  </p>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run:  `python model_trainer.py`")
    st.stop()


# ── Data and scoring (cached) ─────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    return generate_dataset()


@st.cache_data
def cell_history(cell_id: str) -> pd.DataFrame:
    df = load_data()
    return df[df["cell_id"] == cell_id].sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def cell_scores(cell_id: str) -> pd.Series:
    """Model score for every reading of the cell. Features only use the past,
    so scoring the whole history at once gives the same values as replaying it."""
    from model_trainer import load_model_and_scaler, score_readings
    model, scaler = load_model_and_scaler()
    return pd.Series(score_readings(cell_history(cell_id), model, scaler))


@st.cache_data
def cell_events(cell_id: str) -> pd.DataFrame:
    """Labelled events of one cell (ground truth), 1-based reading numbers."""
    c = cell_history(cell_id)
    rows = []
    for ev_id, g in c[c["event_id"] != ""].groupby("event_id", sort=False):
        benign = bool(g["benign_event"].iloc[0])
        rows.append({
            "event_id": ev_id,
            "kind": "harmless surge" if benign else g["scenario"].iloc[0],
            "subtle": (not benign) and g["severity"].iloc[0] < 0.55,
            "first": int(g.index[0]) + 1,
            "last": int(g.index[-1]) + 1,
            "benign": benign,
        })
    return pd.DataFrame(rows)


def when(ts: str) -> str:
    return pd.Timestamp(ts).strftime("%a %d %b, %H:%M")


df = load_data()
all_cells = sorted(df["cell_id"].unique().tolist())


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Cell")
    cell_id = st.selectbox("5G cell", all_cells, index=0)
    full_cell_df = cell_history(cell_id)
    n_total = len(full_cell_df)
    events = cell_events(cell_id)

    if st.session_state.get("cell_for_time") != cell_id:
        st.session_state["current_reading"] = n_total
        st.session_state["cell_for_time"] = cell_id

    def jump_to_event():
        label = st.session_state["event_pick"]
        if label in event_labels:
            ev = events.iloc[event_labels.index(label)]
            st.session_state["current_reading"] = int(min(ev["first"] + 9, ev["last"]))

    event_labels = [
        f"Readings {e.first}–{e.last}: {e.kind}{' (subtle)' if e.subtle else ''}"
        for e in events.itertuples()
    ]

    st.markdown("### Current time")
    st.number_input(
        "Current reading #", min_value=MIN_HISTORY, max_value=n_total, step=1, key="current_reading",
        help="The agent sees every reading up to and including this one, and nothing after it.",
    )
    end_at = int(st.session_state["current_reading"])
    st.caption(f"Reading {end_at} = **{when(full_cell_df.loc[end_at - 1, 'timestamp'])}**")

    if event_labels:
        st.selectbox(
            "Jump into a labelled event", ["Choose an event"] + event_labels, key="event_pick",
            on_change=jump_to_event,
            help="Sets the current time to about 2 hours into the event (ground truth labels).",
        )

    st.markdown("### Charts")
    n_readings = st.slider("Chart history (readings)", 24, 400, 96, step=4,
                           help="Only changes how far back the charts reach. Does not affect the analysis.")
    st.caption(f"{n_readings} readings = {n_readings * 15 / 60:g} hours before the current time")

    st.markdown("### Agent")
    st.code(describe_backend(), language=None)
    st.caption(f"Prompt: {PROMPT_VARIANT} · policy: {DECISION_POLICY} · "
               f"automatic fixes: {'on' if AUTO_APPLY_ENABLED else 'off'}"
               f"{' (CRITICAL only)' if AUTO_APPLY_REQUIRES_CRITICAL else ''}")
    run_btn = st.button("▶ Run agent on current reading", type="primary", width="stretch")

    st.markdown("### Model results (held-out days)")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            ev = json.load(f)["detectors"]["iforest_baseline"]["events"]
        st.markdown(
            f"| | |\n|---|---|\n"
            f"| Fault events caught | {ev['event_recall']:.0%} |\n"
            f"| Severe / subtle | {ev['event_recall_severe']:.0%} / {ev['event_recall_subtle']:.0%} |\n"
            f"| False alarms per cell per day | {ev['false_alarms_per_cell_day']} |\n"
            f"| Harmless surges alarmed | {ev['benign_surges_alarmed']:.0%} |"
        )
        st.caption("Synthetic data, see README.")


# ── Current view ──────────────────────────────────────────────────────────

history_df = full_cell_df.iloc[:end_at]
view_df = history_df.tail(n_readings)
first_shown = int(view_df.index[0]) + 1
scores = cell_scores(cell_id).iloc[:end_at]
latest_score = float(scores.iloc[-1])
latest_verdict = ("CRITICAL" if latest_score <= ANOMALY_SCORE_CRITICAL
                  else "WARNING" if latest_score < ANOMALY_SCORE_WARNING else "NORMAL")
events_in_view = events[(events["last"] >= first_shown) & (events["first"] <= end_at)] if len(events) else events


def shade_events(fig, row=None, col=None):
    """Background bands for labelled events in view (ground truth)."""
    for e in events_in_view.itertuples():
        x0 = full_cell_df.loc[max(e.first, first_shown) - 1, "timestamp"]
        x1 = full_cell_df.loc[min(e.last, end_at) - 1, "timestamp"]
        kwargs = dict(x0=x0, x1=x1, fillcolor=SURGE_COLOUR if e.benign else FAULT_COLOUR,
                      line_width=0, layer="below")
        if row is not None:
            fig.add_vrect(row=row, col=col, **kwargs)
        else:
            fig.add_vrect(**kwargs)


tab1, tab2, tab3, tab4 = st.tabs(["📡 KPI dashboard", "🔍 Model view", "🤖 Agent trace", "⚡ Action plan"])

with tab1:
    st.markdown(f"#### {cell_id} · readings {first_shown}–{end_at} of {n_total} · "
                f"now: {when(full_cell_df.loc[end_at - 1, 'timestamp'])}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cell type", full_cell_df["cell_type"].iloc[0].replace("_", " ").title())
    c2.metric("Model verdict, current reading", latest_verdict,
              delta=f"score {latest_score:+.3f}", delta_color="off")
    c3.metric("RSRP, current reading", f"{view_df['rsrp'].iloc[-1] + 0.0:.1f} dBm")
    c4.metric("SINR, current reading", f"{round(view_df['sinr'].iloc[-1], 1) + 0.0:.1f} dB")

    if len(events_in_view):
        st.caption("Labelled in this window (ground truth, the model never sees it): " + "; ".join(
            f"{e.kind} at readings {e.first}–{e.last}" for e in events_in_view.itertuples()))
    else:
        st.caption("No labelled faults or surges in this window.")

    selected_kpis = st.multiselect("KPIs to display", list(KPIS.keys()),
                                   default=["rsrp", "sinr", "prb_dl", "ho_sr"])
    if selected_kpis:
        n_cols = 2
        n_rows = (len(selected_kpis) + 1) // n_cols
        fig = make_subplots(
            rows=n_rows, cols=n_cols, vertical_spacing=0.22 if n_rows > 1 else 0.1,
            subplot_titles=[f"{k.upper()} · {KPIS[k][0]} ({KPIS[k][1]})".replace(" ()", "")
                            for k in selected_kpis],
        )
        for i, kpi in enumerate(selected_kpis):
            row, col = i // n_cols + 1, i % n_cols + 1
            shade_events(fig, row, col)
            fig.add_trace(go.Scatter(x=view_df["timestamp"], y=view_df[kpi], mode="lines",
                                     line=dict(color="rgba(55,138,221,0.9)", width=1.5),
                                     name=kpi, showlegend=False), row=row, col=col)
        fig.update_layout(height=max(320, n_rows * 290), margin=dict(t=40, b=20, l=40, r=20),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)", tickformat="%a %H:%M")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Red band: labelled fault. Grey band: labelled harmless surge.")


with tab2:
    st.markdown("#### Model score per reading")
    st.caption("Below 0 the model flags the reading. An alarm needs 2 of the last 3 readings flagged. "
               "Bands show the labelled events.")
    shown_scores = scores.iloc[-len(view_df):].to_numpy()
    fig_score = go.Figure()
    shade_events(fig_score)
    fig_score.add_trace(go.Scatter(
        x=view_df["timestamp"], y=shown_scores, mode="lines+markers",
        marker=dict(size=5, color=[SEVERITY_COLOURS["CRITICAL"] if s <= ANOMALY_SCORE_CRITICAL
                                   else SEVERITY_COLOURS["WARNING"] if s < ANOMALY_SCORE_WARNING
                                   else SEVERITY_COLOURS["NORMAL"] for s in shown_scores]),
        line=dict(color="rgba(100,100,100,0.3)", width=1), name="score", showlegend=False,
    ))
    fig_score.add_hline(y=ANOMALY_SCORE_WARNING, line_dash="dash", line_color="#EF9F27",
                        annotation_text="Warning")
    fig_score.add_hline(y=ANOMALY_SCORE_CRITICAL, line_dash="dash", line_color="#E24B4A",
                        annotation_text="Critical")
    fig_score.update_layout(height=320, margin=dict(t=20, b=20), yaxis_title="Score",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig_score.update_xaxes(tickformat="%a %H:%M")
    st.plotly_chart(fig_score, width="stretch")

    st.markdown("#### Did the model catch the labelled events in view?")
    if len(events_in_view):
        alarms = alarms_from_flags((scores < 0).to_numpy(), full_cell_df["cell_id"].iloc[:end_at].to_numpy())
        rows = []
        for e in events_in_view.itertuples():
            seen_until = min(e.last, end_at)
            hit = [i + 1 for i in range(e.first - 1, seen_until) if alarms[i]]
            if e.benign:
                outcome = "alarmed (unwanted)" if hit else "no alarm (correct)"
            else:
                outcome = f"alarm at reading {hit[0]} ({(hit[0] - e.first) * 15} min in)" if hit else "missed so far"
            rows.append({"readings": f"{e.first}–{e.last}", "label": e.kind + (" (subtle)" if e.subtle else ""),
                         "model": outcome})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("No labelled events in this window.")


# ── Run agent ─────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner(f"Running the agent on {cell_id}, reading {end_at} ..."):
        from pipeline import run_analysis
        result = run_analysis(cell_id, history_df)
        st.session_state["agent_result"] = result
        st.session_state["agent_for"] = (cell_id, end_at)

result = st.session_state.get("agent_result")
result_for = st.session_state.get("agent_for")
stale_note = None
if result and result_for != (cell_id, end_at):
    stale_note = (f"This result is for {result_for[0]} at reading {result_for[1]}. "
                  "Run the agent again for the current selection.")

with tab3:
    st.markdown("#### Agent reasoning trace")
    if not result:
        st.info("Click **Run agent on current reading** in the sidebar.")
        st.code("[kpi_monitor] -> [anomaly_scorer] -> (router)\n"
                "    anomalous -> [root_cause_classifier] -> [remediation_planner]\n"
                "    normal    -> [routine_monitor]", language=None)
    else:
        if stale_note:
            st.warning(stale_note)
        st.caption(f"Analysed {result_for[0]}, reading {result_for[1]}")
        icons = {"KPI Monitor": "📡", "Anomaly Scorer": "🔍", "Root Cause Classifier": "🧠",
                 "Remediation Planner": "⚡", "Routine Monitor": "✅"}
        for entry in result.get("reasoning_log", []):
            icon = next((ico for key, ico in icons.items() if key.lower() in entry.lower()), "→")
            st.markdown(f"`{icon}` {entry}")

with tab4:
    st.markdown("#### Action plan")
    if not result:
        st.info("Run the agent first.")
    else:
        if stale_note:
            st.warning(stale_note)
        plan = result.get("action_plan", {})
        severity = plan.get("severity", "NORMAL")
        st.markdown(
            f'<div style="display:inline-block;padding:4px 14px;background:{SEVERITY_COLOURS.get(severity, "#888")};'
            f'color:white;border-radius:99px;font-weight:500;font-size:13px;margin-bottom:12px;">{severity}</div>',
            unsafe_allow_html=True)

        monitoring = plan.get("type") == "MONITORING"
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Plan ID:** `{plan.get('plan_id', 'n/a')}`")
            st.markdown(f"**Cell:** {plan.get('cell_id', 'n/a')}")
            st.markdown(f"**Failure mode:** {plan.get('failure_mode', 'n/a')}")
            st.markdown(f"**Confidence:** {plan.get('confidence', 'n/a')} "
                        f"(from {plan.get('classification_source', 'n/a')})")
            if plan.get("classification_source") == "llm":
                st.markdown("**Rules agree:** " + ("yes" if plan.get("rules_agree") else "no, so no automatic action"))
        with c2:
            st.markdown(f"**Anomaly score:** `{plan.get('anomaly_score', 'n/a')}`")
            if monitoring:
                st.markdown("**Auto-apply:** not needed, no fault")
                st.markdown("**Escalate:** no")
            else:
                if plan.get("apply_immediately"):
                    auto_text = "✅ yes"
                elif plan.get("auto_apply_eligible"):
                    auto_text = "eligible, but automatic changes are switched off: needs approval"
                else:
                    auto_text = "❌ no, needs human approval"
                st.markdown("**Auto-apply:** " + auto_text)
                st.markdown("**Escalate:** " + ("🔴 yes" if plan.get("escalate_to_human") else "no"))

        st.markdown("---")
        if monitoring:
            st.markdown(f"{plan.get('recommendation', '')} Watch KPI: `{plan.get('watch_kpi', 'n/a')}`.")
        for i, a in enumerate(plan.get("actions", []), 1):
            st.markdown(f"{i}. {a}")
        if plan.get("parameters_to_tune"):
            st.markdown("**Parameters to tune:** `" + "`, `".join(plan["parameters_to_tune"]) + "`")
        if plan.get("technical_rationale"):
            st.markdown(f"**Technical rationale:** {plan['technical_rationale']}")
        if plan.get("expected_outcome"):
            st.markdown(f"**Expected outcome:** {plan['expected_outcome']}")

        st.download_button("⬇ Export action plan (JSON)", data=json.dumps(plan, indent=2),
                           file_name=f"action_plan_{plan.get('cell_id', cell_id)}.json",
                           mime="application/json")
