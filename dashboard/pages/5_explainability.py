"""
dashboard/pages/5_explainability.py — SHAP Explainability page.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import fig_exists, fig_path, get_cluster_risk

st.set_page_config(page_title="Explainability | Fraud Detection", page_icon="🧠", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0d1117; color: #ecf0f1; }
h1, h2, h3 { color: #ecf0f1 !important; }
p, li { color: #bdc3c7; }
.section-title { font-size: 1.3rem; font-weight: 700; color: #3498db; margin: 1.5rem 0 0.5rem 0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f0c29,#302b63,#24243e); }
[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
.insight-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(52,152,219,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
}
.insight-title {
    color: #3498db;
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.3rem;
}
.insight-body {
    color: #bdc3c7;
    font-size: 0.85rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Model Explainability — SHAP Analysis")
st.caption("SHAP (SHapley Additive exPlanations) for the XGBoost Hybrid model")
st.divider()

# ─── SHAP overview ───────────────────────────────────────────
st.markdown("""
**SHAP** decomposes each prediction into individual feature contributions.  
A positive SHAP value → pushes prediction toward **fraud**.  
A negative SHAP value → pushes prediction toward **legitimate**.

This analysis validates whether the novel features (`anomaly_score`, `risk_score`) 
actually contribute to the model's fraud decisions.
""")
st.divider()

# ─── Global SHAP ─────────────────────────────────────────────
st.markdown('<div class="section-title">Global Feature Importance</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if fig_exists("shap_summary.png"):
        st.image(str(fig_path("shap_summary.png")),
                 caption="SHAP Beeswarm — Global Feature Importance (Fraud Class)",
                 use_column_width=True)
    else:
        st.info("SHAP summary plot not available. Run Phase 4 to generate it.")

with col2:
    if fig_exists("shap_bar.png"):
        st.image(str(fig_path("shap_bar.png")),
                 caption="SHAP Bar Chart — Mean |SHAP value| per Feature",
                 use_column_width=True)
    else:
        st.info("SHAP bar plot not available. Run Phase 4 to generate it.")

# ─── Per-cluster SHAP ────────────────────────────────────────
st.markdown('<div class="section-title">Per-Cluster SHAP Analysis</div>', unsafe_allow_html=True)
st.markdown("""
Each spatial cluster may have **different feature drivers** for fraud.  
High-risk clusters tend to show stronger `anomaly_score` and `risk_score` contributions.
""")

risk_scores = get_cluster_risk()
available_clusters = []
if risk_scores is not None and "cluster_label" in risk_scores.columns:
    for cid in risk_scores["cluster_label"].tolist():
        if fig_exists(f"shap_cluster_{cid}.png"):
            available_clusters.append(cid)

if available_clusters:
    selected_cluster = st.selectbox(
        "Select Cluster for SHAP Breakdown",
        available_clusters,
        format_func=lambda x: f"Cluster {x}"
    )
    if selected_cluster is not None and fig_exists(f"shap_cluster_{selected_cluster}.png"):
        # Show risk info for selected cluster
        if risk_scores is not None:
            row = risk_scores[risk_scores["cluster_label"] == selected_cluster]
            if len(row) > 0:
                r = row.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Risk Score", f"{r.get('risk_score', 'N/A'):.3f}")
                c2.metric("Risk Tier",  r.get("risk_tier", "N/A"))
                c3.metric("Fraud Rate", f"{r.get('fraud_rate', 0):.4f}")

        st.image(
            str(fig_path(f"shap_cluster_{selected_cluster}.png")),
            caption=f"SHAP Summary — Cluster {selected_cluster}",
            use_column_width=True,
        )
else:
    st.info("Per-cluster SHAP plots not available yet. Run Phase 4 to generate them.")

# ─── Interpretation guide ─────────────────────────────────────
st.divider()
st.markdown('<div class="section-title">How to Read SHAP Plots</div>', unsafe_allow_html=True)

insights = [
    (
        "🔴 anomaly_score — Expected Key Driver",
        "High anomaly_score (sequence deviates from cluster's normal patterns) should have "
        "high positive SHAP values → validates the sequential mining novelty. "
        "If this feature ranks in the top-10, the framework's core contribution is confirmed."
    ),
    (
        "🟠 risk_score — Spatial Context Contribution",
        "The cluster-level risk_score encodes geographic fraud density. High risk_score "
        "with positive SHAP → transactions in high-fraud zones are correctly flagged more aggressively."
    ),
    (
        "🔵 V-columns — Vesta Engineered Features",
        "The 339 V-columns are proprietary Vesta features. High SHAP importance is expected "
        "for a subset. Compare their importance with vs. without the spatial/sequential features."
    ),
    (
        "🟢 TransactionAmt — Amount Signal",
        "Large transaction amounts in certain product categories are classic fraud signals. "
        "Should appear in the top-10 alongside the novel hybrid features."
    ),
    (
        "🟡 D1–D15 — Days Since Features",
        "D-columns capture 'days since last transaction type X'. Unusual gaps (very short or very long) "
        "are captured in the sequence itemsets and reflected here as a SHAP driver."
    ),
]

for title, body in insights:
    st.markdown(f"""
<div class="insight-card">
    <div class="insight-title">{title}</div>
    <div class="insight-body">{body}</div>
</div>""", unsafe_allow_html=True)

st.divider()
st.markdown("""
> **Research Validation**: If `anomaly_score` and `risk_score` both appear in the top-10 SHAP features,
> this empirically validates the novel contribution — spatial cluster context genuinely improves 
> the model's decision-making beyond raw features alone.
""")
