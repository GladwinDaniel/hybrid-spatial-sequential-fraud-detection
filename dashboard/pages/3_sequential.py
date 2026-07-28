"""
dashboard/pages/3_sequential.py — Sequential Pattern Mining Analysis page.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_anomaly_scores, get_adaptive_support, get_pattern_libraries, fig_exists, fig_path

st.set_page_config(page_title="Sequential Patterns | Fraud Detection", page_icon="🔁", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0d1117; color: #ecf0f1; }
h1, h2, h3 { color: #ecf0f1 !important; }
p, li { color: #bdc3c7; }
.section-title { font-size: 1.3rem; font-weight: 700; color: #3498db; margin: 1.5rem 0 0.5rem 0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f0c29,#302b63,#24243e); }
[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
.pattern-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(52,152,219,0.2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.3rem 0;
    font-family: monospace;
    font-size: 0.85rem;
    color: #3498db;
}
.pattern-card.suspicious {
    border-color: rgba(231,76,60,0.3);
    color: #e74c3c;
}
</style>
""", unsafe_allow_html=True)

st.title("🔁 Sequential Pattern Mining")
st.caption("Adaptive PrefixSpan per spatial cluster — core novelty of the framework")
st.divider()

scores_df  = get_anomaly_scores()
adaptive   = get_adaptive_support()
libraries  = get_pattern_libraries()

if scores_df is None and adaptive is None and libraries is None:
    st.warning("⚠️ Sequential mining outputs not found. Run Phase 3 first.")
    st.code("python pipeline.py --phase 1 2 3", language="bash")
    st.stop()

# ─── Summary metrics ─────────────────────────────────────────
if scores_df is not None and adaptive is not None:
    n_clusters = adaptive["cluster_id"].nunique() if "cluster_id" in adaptive.columns else 0
    total_pats = int(adaptive["n_patterns"].sum()) if "n_patterns" in adaptive.columns else 0
    fraud_mean = scores_df.loc[scores_df["label"]==1, "anomaly_score"].mean() if "label" in scores_df.columns else 0
    legit_mean = scores_df.loc[scores_df["label"]==0, "anomaly_score"].mean() if "label" in scores_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters Mined",     f"{n_clusters}")
    c2.metric("Total Patterns",     f"{total_pats:,}")
    c3.metric("Fraud Anomaly Score", f"{fraud_mean:.4f}")
    c4.metric("Legit Anomaly Score", f"{legit_mean:.4f}")
    st.divider()

# ─── Adaptive support chart ──────────────────────────────────
if adaptive is not None and "min_sup_frac" in adaptive.columns:
    st.markdown('<div class="section-title">Adaptive min_support per Cluster</div>', unsafe_allow_html=True)
    st.markdown("""
The **adaptive support formula** is the core novelty:  
`min_support(Ci) = base_support × (1 / risk_score(Ci))`

High-risk clusters get **lower** thresholds → more patterns detected → more sensitive.
""")

    color_map = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
    adap = adaptive.copy()
    adap["risk_tier"] = adap.get("risk_tier", "MEDIUM")

    fig = go.Figure()
    for tier, color in color_map.items():
        sub = adap[adap.get("risk_tier", pd.Series(["MEDIUM"]*len(adap))) == tier] if "risk_tier" in adap.columns else adap
        if len(sub) == 0: continue
        fig.add_trace(go.Bar(
            x=sub["cluster_id"].astype(str),
            y=sub["min_sup_frac"] * 100,
            name=f"{tier} risk",
            marker_color=color,
        ))

    from config import PREFIXSPAN_BASE_SUPPORT
    fig.add_hline(y=PREFIXSPAN_BASE_SUPPORT * 100, line_dash="dash",
                  line_color="gray", annotation_text=f"Base support ({PREFIXSPAN_BASE_SUPPORT*100:.0f}%)")
    fig.update_layout(
        title="Adaptive PrefixSpan min_support per Spatial Cluster",
        xaxis_title="Cluster ID", yaxis_title="min_support (%)",
        barmode="stack",
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        font_color="#ecf0f1", height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Adaptive support table
    with st.expander("📋 Adaptive Support Details Table"):
        st.dataframe(adap.style.format({
            "risk_score": "{:.3f}",
            "min_sup_frac": "{:.4f}",
        }), use_container_width=True)

# ─── Anomaly score distribution ──────────────────────────────
if scores_df is not None and "anomaly_score" in scores_df.columns:
    st.markdown('<div class="section-title">Anomaly Score Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # Box plot
        plot_df = scores_df.copy()
        if "label" in plot_df.columns:
            plot_df["Class"] = plot_df["label"].map({0: "Legitimate", 1: "Fraud"})
        else:
            plot_df["Class"] = "All"

        fig2 = px.box(
            plot_df, x="Class", y="anomaly_score",
            color="Class",
            color_discrete_map={"Legitimate": "#3498db", "Fraud": "#e74c3c", "All": "#9b59b6"},
            title="Anomaly Score — Box Plot",
            points="outliers",
        )
        fig2.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
            font_color="#ecf0f1", showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Histogram overlay
        fig3 = go.Figure()
        for label_val, color, name in [(0, "#3498db", "Legitimate"), (1, "#e74c3c", "Fraud")]:
            if "label" in scores_df.columns:
                data = scores_df.loc[scores_df["label"] == label_val, "anomaly_score"]
            else:
                data = scores_df["anomaly_score"]
            fig3.add_trace(go.Histogram(
                x=data, name=name, opacity=0.65,
                marker_color=color, nbinsx=40,
            ))
        fig3.update_layout(
            barmode="overlay",
            title="Anomaly Score — Distribution Overlay",
            xaxis_title="Anomaly Score", yaxis_title="Count",
            plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
            font_color="#ecf0f1",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Anomaly score by cluster
    if "cluster_id" in scores_df.columns:
        st.markdown('<div class="section-title">Anomaly Score by Cluster</div>', unsafe_allow_html=True)
        cluster_scores = scores_df.groupby("cluster_id")["anomaly_score"].mean().reset_index()
        cluster_scores = cluster_scores.sort_values("anomaly_score", ascending=False)
        fig4 = px.bar(
            cluster_scores.head(30), x="cluster_id", y="anomaly_score",
            title="Mean Anomaly Score per Spatial Cluster",
            color="anomaly_score",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        )
        fig4.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
            font_color="#ecf0f1",
        )
        st.plotly_chart(fig4, use_container_width=True)

# ─── Pattern libraries browser ───────────────────────────────
if libraries is not None:
    st.markdown('<div class="section-title">Pattern Library Browser</div>', unsafe_allow_html=True)
    cluster_ids = sorted(libraries.keys(), key=lambda x: int(x))
    selected = st.selectbox("Select Cluster", cluster_ids,
                             format_func=lambda x: f"Cluster {x}")

    if selected in libraries:
        lib = libraries[selected]
        n_normal = len(lib.get("normal", []))
        n_suspicious = len(lib.get("suspicious", []))
        col1, col2, col3 = st.columns(3)
        col1.metric("Normal Patterns",     n_normal)
        col2.metric("Suspicious Patterns", n_suspicious)
        col3.metric("Max Support",         lib.get("max_support", "N/A"))

        tab1, tab2 = st.tabs(["✅ Top Normal Patterns", "⚠️ Suspicious Patterns"])

        with tab1:
            normal_pats = sorted(lib.get("normal", []), key=lambda x: x[0], reverse=True)[:10]
            if normal_pats:
                for cnt, pat in normal_pats:
                    pat_str = " → ".join(str(p) for p in pat)
                    st.markdown(f'<div class="pattern-card">support: {cnt} | {pat_str}</div>',
                                unsafe_allow_html=True)
            else:
                st.info("No normal patterns found for this cluster.")

        with tab2:
            susp_pats = sorted(lib.get("suspicious", []), key=lambda x: x[0])[:10]
            if susp_pats:
                for cnt, pat in susp_pats:
                    pat_str = " → ".join(str(p) for p in pat)
                    st.markdown(f'<div class="pattern-card suspicious">support: {cnt} | {pat_str}</div>',
                                unsafe_allow_html=True)
            else:
                st.info("No suspicious patterns found for this cluster.")

# ─── Saved figures ────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-title">Saved Figures</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
if fig_exists("anomaly_scores.png"):
    with col1:
        st.image(str(fig_path("anomaly_scores.png")),
                 caption="Anomaly Score Distribution", use_column_width=True)
if fig_exists("adaptive_support.png"):
    with col2:
        st.image(str(fig_path("adaptive_support.png")),
                 caption="Adaptive Support per Cluster", use_column_width=True)
