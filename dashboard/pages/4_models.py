"""
dashboard/pages/4_models.py — Model Comparison page.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_results, fig_exists, fig_path, MODEL_COLORS

st.set_page_config(page_title="Model Comparison | Fraud Detection", page_icon="🤖", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0d1117; color: #ecf0f1; }
h1, h2, h3 { color: #ecf0f1 !important; }
p, li { color: #bdc3c7; }
.section-title { font-size: 1.3rem; font-weight: 700; color: #3498db; margin: 1.5rem 0 0.5rem 0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f0c29,#302b63,#24243e); }
[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
.winner-badge {
    background: linear-gradient(90deg, #e74c3c, #c0392b);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 700;
    color: white;
    margin-left: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Model Comparison")
st.caption("Baseline XGBoost vs. Sequential-only vs. Spatial-only vs. Proposed Hybrid")
st.divider()

results = get_results()

if results is None or results.empty:
    st.warning("⚠️ Evaluation results not found. Run Phase 5 first.")
    st.code("python pipeline.py", language="bash")
    st.stop()

# ─── Results table ────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Results Summary Table</div>', unsafe_allow_html=True)

def _highlight_hybrid(row):
    if row.get("model") == "Proposed Hybrid":
        return ["background-color: rgba(231,76,60,0.15); font-weight: bold"] * len(row)
    return [""] * len(row)

def _color_metric(val):
    try:
        v = float(val)
        if v >= 0.90: return "color: #2ecc71; font-weight: bold"
        if v >= 0.75: return "color: #f39c12"
        return "color: #e74c3c"
    except:
        return ""

display_cols = [c for c in results.columns if c != "model"]
styled = results.style.apply(_highlight_hybrid, axis=1)
for col in display_cols:
    try:
        styled = styled.applymap(_color_metric, subset=[col])
    except:
        pass
styled = styled.format({c: "{:.4f}" for c in display_cols if results[c].dtype != object})
st.dataframe(styled, use_container_width=True, hide_index=True)

# ─── Grouped bar chart ───────────────────────────────────────
st.markdown('<div class="section-title">Performance Comparison</div>', unsafe_allow_html=True)
metric_cols = [c for c in results.columns if c not in ("model",) and results[c].dtype in (float, np.float64)]
if not metric_cols:
    metric_cols = [c for c in results.columns if c != "model"]

fig = go.Figure()
for _, row in results.iterrows():
    model_name = row["model"]
    values = [row.get(c, 0) for c in metric_cols[:4]]
    fig.add_trace(go.Bar(
        name=model_name,
        x=metric_cols[:4],
        y=values,
        marker_color=MODEL_COLORS.get(model_name, "#7f8c8d"),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))

fig.update_layout(
    barmode="group",
    title="Model Performance — AUC-ROC, PR-AUC, F1",
    yaxis=dict(range=[0, 1.05]),
    plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
    font_color="#ecf0f1", height=480,
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
)
st.plotly_chart(fig, use_container_width=True)

# ─── Radar chart ──────────────────────────────────────────────
if len(metric_cols) >= 3:
    st.markdown('<div class="section-title">Radar Chart — Multi-metric</div>', unsafe_allow_html=True)
    radar_metrics = metric_cols[:5]
    fig2 = go.Figure()

    for _, row in results.iterrows():
        model_name = row["model"]
        vals = [row.get(c, 0) for c in radar_metrics]
        fig2.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_metrics + [radar_metrics[0]],
            fill="toself",
            name=model_name,
            line_color=MODEL_COLORS.get(model_name, "#7f8c8d"),
            opacity=0.6,
        ))
    fig2.update_layout(
        polar=dict(
            bgcolor="#1a1a2e",
            radialaxis=dict(range=[0, 1], gridcolor="#2c3e50", tickfont_color="#7f8c8d"),
            angularaxis=dict(gridcolor="#2c3e50"),
        ),
        paper_bgcolor="#1a1a2e", font_color="#ecf0f1",
        title="Multi-metric Radar — All Models",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=480,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─── Saved figures ────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-title">ROC & PR Curves | Confusion Matrices</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["ROC Curves", "PR Curves", "Confusion Matrices"])
with tab1:
    if fig_exists("roc_comparison.png"):
        st.image(str(fig_path("roc_comparison.png")), use_column_width=True)
    else:
        st.info("ROC plot not available — run evaluation phase.")

with tab2:
    if fig_exists("pr_comparison.png"):
        st.image(str(fig_path("pr_comparison.png")), use_column_width=True)
    else:
        st.info("PR plot not available — run evaluation phase.")

with tab3:
    if fig_exists("confusion_matrices.png"):
        st.image(str(fig_path("confusion_matrices.png")), use_column_width=True)
    else:
        st.info("Confusion matrix plot not available — run evaluation phase.")

# ─── Improvement analysis ────────────────────────────────────
st.divider()
st.markdown('<div class="section-title">Improvement Over Baseline</div>', unsafe_allow_html=True)

baseline_row = results[results["model"] == "Baseline XGBoost"]
hybrid_row   = results[results["model"] == "Proposed Hybrid"]

if len(baseline_row) > 0 and len(hybrid_row) > 0:
    b = baseline_row.iloc[0]
    h = hybrid_row.iloc[0]

    improvements = {}
    for col in metric_cols[:4]:
        try:
            improvements[col] = float(h[col]) - float(b[col])
        except:
            pass

    if improvements:
        imp_df = pd.DataFrame([
            {"Metric": k, "Absolute Improvement": v, "Direction": "↑" if v > 0 else "↓"}
            for k, v in improvements.items()
        ])
        fig3 = px.bar(
            imp_df, x="Metric", y="Absolute Improvement",
            color="Absolute Improvement",
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
            title="Hybrid vs. Baseline — Absolute Metric Improvement",
            text="Absolute Improvement",
        )
        fig3.add_hline(y=0, line_color="#95a5a6", line_dash="dash")
        fig3.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig3.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
            font_color="#ecf0f1",
        )
        st.plotly_chart(fig3, use_container_width=True)
