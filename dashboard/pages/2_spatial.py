"""
dashboard/pages/2_spatial.py — Spatial Clustering Analysis page.
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
from utils import get_cluster_labels, get_cluster_risk, fig_exists, fig_path, TIER_COLORS

st.set_page_config(page_title="Spatial Analysis | Fraud Detection", page_icon="🗺️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0d1117; color: #ecf0f1; }
h1, h2, h3 { color: #ecf0f1 !important; }
p, li { color: #bdc3c7; }
.section-title { font-size: 1.3rem; font-weight: 700; color: #3498db; margin: 1.5rem 0 0.5rem 0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f0c29,#302b63,#24243e); }
[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
.tier-high { color: #e74c3c; font-weight: 700; }
.tier-medium { color: #f39c12; font-weight: 700; }
.tier-low { color: #2ecc71; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("🗺️ Spatial Clustering Analysis")
st.caption("DBSCAN clusters on addr1, addr2, dist1, dist2 — coloured by fraud risk tier")
st.divider()

cluster_df  = get_cluster_labels()
risk_scores = get_cluster_risk()

if cluster_df is None or risk_scores is None:
    st.warning("⚠️ Spatial clustering outputs not found. Run Phase 2 first.")
    st.code("python pipeline.py --phase 1 2", language="bash")
    st.stop()

# ─── Summary metrics ─────────────────────────────────────────
n_clusters  = cluster_df["cluster_label"].nunique()
n_high      = len(risk_scores[risk_scores["risk_tier"] == "HIGH"])  if "risk_tier" in risk_scores.columns else 0
n_medium    = len(risk_scores[risk_scores["risk_tier"] == "MEDIUM"]) if "risk_tier" in risk_scores.columns else 0
n_low       = len(risk_scores[risk_scores["risk_tier"] == "LOW"])   if "risk_tier" in risk_scores.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Clusters",      f"{n_clusters}")
c2.metric("🔴 High-Risk Clusters", f"{n_high}")
c3.metric("🟡 Medium-Risk",       f"{n_medium}")
c4.metric("🟢 Low-Risk Clusters",  f"{n_low}")
st.divider()

# ─── Risk score bar chart (interactive) ─────────────────────
st.markdown('<div class="section-title">Cluster Risk Scores</div>', unsafe_allow_html=True)
if "risk_score" in risk_scores.columns:
    rs = risk_scores.sort_values("risk_score", ascending=False).head(40)
    color_map = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
    colors = [color_map.get(t, "#7f8c8d") for t in rs.get("risk_tier", ["MEDIUM"] * len(rs))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rs["cluster_label"].astype(str),
        y=rs["risk_score"],
        marker_color=colors,
        text=[f"{v:.2f}" for v in rs["risk_score"]],
        textposition="outside",
        name="Risk Score",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#95a5a6",
                  annotation_text="Global average (1.0)", annotation_position="top right")
    fig.add_hline(y=2.0, line_dash="dot", line_color="#e74c3c",
                  annotation_text="High-risk threshold", annotation_position="top right")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#2ecc71",
                  annotation_text="Low-risk threshold", annotation_position="bottom right")
    fig.update_layout(
        title="Cluster Fraud Risk Scores (risk = cluster_fraud_rate / global_fraud_rate)",
        xaxis_title="Cluster Label",
        yaxis_title="Risk Score",
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        font_color="#ecf0f1", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── Risk score table ────────────────────────────────────────
st.markdown('<div class="section-title">Cluster Risk Score Table</div>', unsafe_allow_html=True)
if "risk_tier" in risk_scores.columns:
    def _style_tier(val):
        colors = {"HIGH": "color: #e74c3c; font-weight:bold",
                  "MEDIUM": "color: #f39c12; font-weight:bold",
                  "LOW": "color: #2ecc71; font-weight:bold"}
        return colors.get(val, "")

    display_cols = [c for c in ["cluster_label", "size", "fraud_count",
                                "fraud_rate", "risk_score", "risk_tier"]
                    if c in risk_scores.columns]
    styled = risk_scores[display_cols].style.applymap(
        _style_tier, subset=["risk_tier"]
    ).format({
        "fraud_rate": "{:.4f}",
        "risk_score": "{:.3f}",
        "size": "{:,}",
        "fraud_count": "{:,}",
    })
    st.dataframe(styled, use_container_width=True, height=350)
else:
    st.dataframe(risk_scores, use_container_width=True)

# ─── Cluster size vs fraud rate scatter ─────────────────────
if "fraud_rate" in risk_scores.columns and "size" in risk_scores.columns:
    st.markdown('<div class="section-title">Cluster Size vs. Fraud Rate</div>', unsafe_allow_html=True)
    scatter_df = risk_scores.copy()
    scatter_df["risk_tier"] = scatter_df.get("risk_tier", "MEDIUM")
    fig2 = px.scatter(
        scatter_df,
        x="size", y="fraud_rate",
        size="size", color="risk_tier",
        color_discrete_map={"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"},
        hover_data=["cluster_label", "risk_score"],
        title="Cluster Size vs. Fraud Rate (bubble = cluster size)",
        labels={"size": "Cluster Size", "fraud_rate": "Fraud Rate"},
        log_x=True,
    )
    fig2.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        font_color="#ecf0f1", height=450,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─── Saved cluster figures ───────────────────────────────────
st.markdown('<div class="section-title">Cluster Visualizations</div>', unsafe_allow_html=True)
figs_to_show = [
    ("cluster_pca.png",   "PCA 2D Projection of Spatial Clusters"),
    ("cluster_umap.png",  "UMAP 2D Projection of Spatial Clusters"),
    ("risk_heatmap.png",  "Cluster Risk Score Heatmap"),
    ("epsilon_plot.png",  "k-Distance Graph (Epsilon Tuning)"),
]
col1, col2 = st.columns(2)
for i, (fname, caption) in enumerate(figs_to_show):
    if fig_exists(fname):
        with [col1, col2][i % 2]:
            st.image(str(fig_path(fname)), caption=caption, use_column_width=True)
