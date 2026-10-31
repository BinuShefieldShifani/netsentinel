"""
app.py - NetSentinel Streamlit dashboard
========================================
Run with:  streamlit run app.py

Before first run:
  1. python model_trainer.py    ← generates data + trains model
  2. streamlit run app.py
"""

import json
import os

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import (
    ANOMALY_SCORE_CRITICAL,
    ANOMALY_SCORE_WARNING,
    KPIS,
    METRICS_PATH,
    MODEL_PATH,
    OLLAMA_MODEL,
)
from data_generator import generate_dataset

st.set_page_config(
    page_title="NetSentinel — 5G RAN Monitor",
    page_icon="📡",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding-bottom:8px;">
  <h1 style="margin:0;font-size:26px;">NetSentinel</h1>
  <p style="margin:0;color:gray;font-size:14px;">
    5G RAN Anomaly Detection & Self-Healing Agent ·
    Isolation Forest + LangGraph · synthetic 3GPP TS 28.552 KPIs
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Preflight ─────────────────────────────────────────────────────────────

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run:  `python model_trainer.py`")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    return generate_dataset()

df = load_data()
all_cells = sorted(df["cell_id"].unique().tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Cell Selection")
    cell_id = st.selectbox("Select 5G Cell", all_cells, index=0)

    st.markdown("---")
    st.markdown("### Display Window")
    n_readings = st.slider("Readings to show", 20, 400, 96,
                           help="Each reading = 15 min interval")
    full_cell_df = df[df["cell_id"] == cell_id].sort_values("timestamp").reset_index(drop=True)
    end_at = st.slider("Analyse up to reading", 50, len(full_cell_df), len(full_cell_df),
                       help="Move back in time to replay a fault. The agent sees all readings up to this point.")
    fault_starts = full_cell_df.index[
        (full_cell_df["is_anomaly"] == 1) & (full_cell_df["is_anomaly"].shift(1, fill_value=0) == 0)
    ].tolist()
    if fault_starts:
        st.caption("Fault events in this cell start at readings: "
                   + ", ".join(str(i + 1) for i in fault_starts[:12]))

    st.markdown("---")
    st.markdown("### Agent Config")
    st.code(f"Model: {OLLAMA_MODEL}", language=None)

    run_btn = st.button("▶ Run Agent Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### Model results (held-out days)")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            m = json.load(f)
        pm = m["detectors"]["iforest_baseline"]["point"]
        st.markdown(
            f"| Metric | Value |\n|---|---|\n"
            f"| Precision | {pm['precision']:.3f} |\n| Recall | {pm['recall']:.3f} |\n"
            f"| F1 | {pm['f1']:.3f} |"
        )
        ev = m["detectors"]["iforest_baseline"]["events"]
        st.caption(f"Fault events caught: {ev['event_recall']:.0%} · "
                   f"false alarms per cell per day: {ev['false_alarms_per_cell_day']} · "
                   "synthetic data, see README")
    else:
        st.caption("Run `python model_trainer.py` to compute metrics.")

# ── Cell Data ─────────────────────────────────────────────────────────────

history_df = full_cell_df.iloc[:end_at]
cell_df = history_df.tail(n_readings).copy()
has_anomaly = bool(cell_df["is_anomaly"].any())
anomaly_pct = cell_df["is_anomaly"].mean() * 100

# ── Tabs ──────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📡 KPI Dashboard",
    "🔍 Anomaly View",
    "🤖 Agent Trace",
    "⚡ Action Plan",
])

# ── TAB 1: KPI Dashboard ──────────────────────────────────────────────────

with tab1:
    st.markdown(f"#### Cell {cell_id} — Last {n_readings} Readings ({n_readings*15} min)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cell type",    cell_df["cell_type"].iloc[-1].replace("_", " ").title())
    c2.metric("Anomaly rate", f"{anomaly_pct:.1f}%",
              delta="⚠ anomalies detected" if has_anomaly else "✓ normal",
              delta_color="inverse")
    c3.metric("RSRP (latest)", f"{cell_df['rsrp'].iloc[-1]:.1f} dBm")
    c4.metric("SINR (latest)", f"{cell_df['sinr'].iloc[-1]:.1f} dB")

    st.markdown("---")

    selected_kpis = st.multiselect(
        "Select KPIs to display",
        list(KPIS.keys()),
        default=["rsrp", "sinr", "prb_dl", "ho_sr"],
    )

    if selected_kpis:
        n_cols = 2
        n_rows = (len(selected_kpis) + 1) // n_cols
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"{k.upper()} — {KPIS[k][0]} ({KPIS[k][1]})"
                            for k in selected_kpis],
            vertical_spacing=0.14,
        )
        timestamps = cell_df["timestamp"].tolist()

        for i, kpi in enumerate(selected_kpis):
            row = i // n_cols + 1
            col = i % n_cols + 1
            values = cell_df[kpi].values
            anom_mask = cell_df["is_anomaly"].values.astype(bool)

            fig.add_trace(go.Scatter(
                x=timestamps, y=values,
                mode="lines",
                line=dict(color="rgba(55,138,221,0.8)", width=1.5),
                name=kpi, showlegend=False,
            ), row=row, col=col)

            # Highlight anomalous points in red
            if anom_mask.any():
                fig.add_trace(go.Scatter(
                    x=[timestamps[j] for j in range(len(timestamps)) if anom_mask[j]],
                    y=[values[j] for j in range(len(values)) if anom_mask[j]],
                    mode="markers",
                    marker=dict(color="#E24B4A", size=6, symbol="x"),
                    name="anomaly", showlegend=(i == 0),
                ), row=row, col=col)

        fig.update_layout(
            height=max(300, n_rows * 220),
            margin=dict(t=40, b=20, l=40, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
        st.plotly_chart(fig, use_container_width=True)


# ── TAB 2: Anomaly View ───────────────────────────────────────────────────

with tab2:
    st.markdown("#### Anomaly Score Over Time")
    st.markdown("Below 0 = predicted anomaly. Dashed lines show Warning and Critical thresholds.")

    @st.cache_data
    def compute_scores(cell_id, end_at, n):
        from model_trainer import load_model_and_scaler, score_readings
        model, scaler = load_model_and_scaler()
        hist = df[df["cell_id"] == cell_id].sort_values("timestamp").iloc[:end_at]
        scores = score_readings(hist, model, scaler)[-n:]   # score with full history, show the tail
        shown = hist.tail(n)
        return shown["timestamp"].tolist(), scores, shown["is_anomaly"].values

    timestamps, scores, true_labels = compute_scores(cell_id, end_at, n_readings)

    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(
        x=timestamps, y=scores,
        mode="lines+markers",
        marker=dict(
            color=["#E24B4A" if s <= ANOMALY_SCORE_CRITICAL else "#EF9F27" if s < ANOMALY_SCORE_WARNING else "#1D9E75"
                   for s in scores],
            size=5,
        ),
        line=dict(color="rgba(100,100,100,0.3)", width=1),
        name="Anomaly Score",
    ))
    fig_score.add_hline(y=ANOMALY_SCORE_WARNING, line_dash="dash", line_color="#EF9F27",
                        annotation_text="Warning threshold")
    fig_score.add_hline(y=ANOMALY_SCORE_CRITICAL, line_dash="dash", line_color="#E24B4A",
                        annotation_text="Critical threshold")

    fig_score.update_layout(
        height=300, xaxis_title="Timestamp", yaxis_title="Anomaly Score",
        margin=dict(t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # Scenario breakdown
    st.markdown("---")
    st.markdown("#### Anomaly Scenario Breakdown (ground truth labels)")
    scenario_counts = cell_df[cell_df["is_anomaly"] == 1]["scenario"].value_counts()
    if len(scenario_counts) > 0:
        fig_sc = go.Figure(go.Bar(
            x=scenario_counts.index.tolist(),
            y=scenario_counts.values.tolist(),
            marker_color="#534AB7",
        ))
        fig_sc.update_layout(
            height=220,
            xaxis_title="Scenario", yaxis_title="Count",
            margin=dict(t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.success("No anomalies in this window — cell is healthy.")


# ── TAB 3: Agent Trace ────────────────────────────────────────────────────

with tab3:
    st.markdown("#### Multi-Agent Reasoning Trace")

    if "agent_result" not in st.session_state:
        st.info("Click **▶ Run Agent Analysis** in the sidebar to execute the pipeline.")
        st.markdown("""
**Agent pipeline:**
```
[kpi_monitor] → [anomaly_scorer] → (conditional)
    anomalous: → [root_cause_classifier] → [remediation_planner]
    normal:    → [routine_monitor]
```
""")
    else:
        result = st.session_state["agent_result"]
        icons = {
            "KPI Monitor":           "📡",
            "Anomaly Scorer":        "🔍",
            "Root Cause Classifier": "🧠",
            "Remediation Planner":   "⚡",
            "Routine Monitor":       "✅",
        }
        for entry in result.get("reasoning_log", []):
            icon = "→"
            for key, ico in icons.items():
                if key.lower() in entry.lower():
                    icon = ico
                    break
            st.markdown(f"`{icon}` {entry}")


# ── TAB 4: Action Plan ────────────────────────────────────────────────────

with tab4:
    st.markdown("#### Remediation Action Plan")

    if "agent_result" not in st.session_state:
        st.info("Run Agent Analysis first.")
    else:
        result = st.session_state["agent_result"]
        plan   = result.get("action_plan", {})

        if not plan:
            st.warning("No action plan generated.")
        else:
            severity = plan.get("severity", "NORMAL")
            color_map = {"CRITICAL": "#E24B4A", "WARNING": "#EF9F27", "NORMAL": "#1D9E75"}
            badge_color = color_map.get(severity, "#888")

            st.markdown(
                f'<div style="display:inline-block;padding:4px 14px;'
                f'background:{badge_color};color:white;border-radius:99px;'
                f'font-weight:500;font-size:13px;margin-bottom:12px;">'
                f'{severity}</div>',
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Plan ID:** `{plan.get('plan_id','—')}`")
                st.markdown(f"**Cell:** {plan.get('cell_id','—')}")
                st.markdown(f"**Failure Mode:** {plan.get('failure_mode','—')}")
                st.markdown(f"**Confidence:** {plan.get('confidence','—')} "
                            f"({plan.get('classification_source','—')})")
            with c2:
                st.markdown(f"**Anomaly Score:** `{plan.get('anomaly_score','—')}`")
                apply = plan.get("apply_immediately", False)
                escalate = plan.get("escalate_to_human", True)
                st.markdown(f"**Auto-apply:** {'✅ Yes' if apply else '❌ No — human approval needed'}")
                st.markdown(f"**Escalate:** {'🔴 Yes' if escalate else 'No'}")

            st.markdown("---")

            actions = plan.get("actions", [])
            if actions:
                st.markdown("**Actions:**")
                for i, a in enumerate(actions, 1):
                    st.markdown(f"{i}. {a}")

            params = plan.get("parameters_to_tune", [])
            if params:
                st.markdown(f"**Parameters to tune:** `{'`, `'.join(params)}`")

            rationale = plan.get("technical_rationale", "")
            if rationale:
                st.markdown(f"**Technical Rationale:** {rationale}")

            outcome = plan.get("expected_outcome", "")
            if outcome:
                st.markdown(f"**Expected Outcome:** {outcome}")

            st.markdown("---")
            st.download_button(
                "⬇ Export Action Plan JSON",
                data=json.dumps(plan, indent=2),
                file_name=f"action_plan_{cell_id}.json",
                mime="application/json",
            )


# ── Run Agent ─────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner(f"Running 4-node agent pipeline for {cell_id} ..."):
        try:
            from pipeline import run_analysis
            result = run_analysis(cell_id, history_df)
            st.session_state["agent_result"] = result
            sev = result.get("severity", "?")
            fm  = result.get("failure_mode", "?")
            st.success(f"Analysis complete — {cell_id} | Severity: **{sev}** | Mode: **{fm}**")
            st.rerun()
        except Exception as e:
            st.error(f"Agent error: {e}")
            import traceback
            st.code(traceback.format_exc())