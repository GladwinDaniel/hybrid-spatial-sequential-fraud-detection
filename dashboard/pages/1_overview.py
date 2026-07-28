"""
dashboard/pages/1_overview.py — EDA & Dataset Overview page.
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
from utils import fig_exists, fig_path, load_pkl

st.set_page_config(page_title="Overview | Fraud Detection", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #0d1117; color: #ecf0f1; }
h1, h2, h3 { color: #ecf0f1 !important; }
p, li { color: #bdc3c7; }
.section-title { font-size: 1.3rem; font-weight: 700; color: #3498db; margin: 1.5rem 0 0.5rem 0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f0c29,#302b63,#24243e); }
[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Dataset Overview & EDA")
st.caption("IEEE-CIS Fraud Detection Dataset — Exploratory Data Analysis")
st.divider()

# ─── Load data ───────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_features():
    pkl_path = ROOT / "data" / "processed" / "features.pkl"
    if pkl_path.exists():
        import pickle
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return None

df = load_features()

if df is None:
    st.warning("⚠️ No processed data found. Run `python pipeline.py` first.")
    st.code("python pipeline.py", language="bash")
    st.stop()

# ─── Key stats ───────────────────────────────────────────────
st.markdown('<div class="section-title">Key Statistics</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{len(df):,}")
c2.metric("Fraud Transactions", f"{int(df['isFraud'].sum()):,}")
c3.metric("Fraud Rate", f"{df['isFraud'].mean()*100:.2f}%")
c4.metric("Features", f"{df.shape[1]}")
st.divider()

# ─── Class distribution ───────────────────────────────────────
st.markdown('<div class="section-title">Class Distribution</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    counts = df["isFraud"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
    fig = px.bar(
        x=counts.index, y=counts.values,
        color=counts.index,
        color_discrete_map={"Legitimate": "#3498db", "Fraud": "#e74c3c"},
        labels={"x": "Class", "y": "Count"},
        title="Transaction Class Distribution",
    )
    fig.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        font_color="#ecf0f1", showlegend=False,
        title_font_size=14,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = px.pie(
        values=counts.values, names=counts.index,
        color=counts.index,
        color_discrete_map={"Legitimate": "#3498db", "Fraud": "#e74c3c"},
        title="Fraud vs. Legitimate Ratio",
        hole=0.45,
    )
    fig2.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        font_color="#ecf0f1", title_font_size=14,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─── Transaction Amount ───────────────────────────────────────
st.markdown('<div class="section-title">Transaction Amount Distribution</div>', unsafe_allow_html=True)
fig3 = go.Figure()
for label, color, name in [(0, "#3498db", "Legitimate"), (1, "#e74c3c", "Fraud")]:
    data = np.log1p(df.loc[df["isFraud"] == label, "TransactionAmt"].dropna())
    fig3.add_trace(go.Histogram(
        x=data, name=name, opacity=0.7,
        marker_color=color, nbinsx=60,
    ))
fig3.update_layout(
    barmode="overlay", title="log(1 + TransactionAmt) by Class",
    xaxis_title="log(1 + Amount)", yaxis_title="Count",
    plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
    font_color="#ecf0f1", legend=dict(bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig3, use_container_width=True)

# ─── Fraud rate over time ─────────────────────────────────────
st.markdown('<div class="section-title">Fraud Rate Over Time</div>', unsafe_allow_html=True)
df["dt_bin"] = pd.cut(df["TransactionDT"], bins=60, labels=False)
time_fraud = df.groupby("dt_bin")["isFraud"].mean().reset_index()
fig4 = px.area(
    time_fraud, x="dt_bin", y="isFraud",
    title="Fraud Rate Over Time (TransactionDT bins)",
    labels={"dt_bin": "Time Bin →", "isFraud": "Fraud Rate"},
    color_discrete_sequence=["#e74c3c"],
)
fig4.update_layout(
    plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
    font_color="#ecf0f1",
)
st.plotly_chart(fig4, use_container_width=True)

# ─── Fraud by ProductCD ───────────────────────────────────────
if "ProductCD" in df.columns:
    st.markdown('<div class="section-title">Fraud Rate by ProductCD</div>', unsafe_allow_html=True)
    prod = df.groupby("ProductCD")["isFraud"].mean().reset_index().sort_values("isFraud", ascending=False)
    fig5 = px.bar(
        prod, x="ProductCD", y="isFraud",
        title="Fraud Rate by Product Category",
        labels={"isFraud": "Fraud Rate", "ProductCD": "Product Code"},
        color="isFraud",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
    )
    fig5.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
        font_color="#ecf0f1",
    )
    st.plotly_chart(fig5, use_container_width=True)

# ─── Missing values ───────────────────────────────────────────
st.markdown('<div class="section-title">Top-20 Columns by Missing Rate</div>', unsafe_allow_html=True)
missing = (df.isnull().mean() * 100).sort_values(ascending=False).head(20).reset_index()
missing.columns = ["Column", "Missing %"]
fig6 = px.bar(
    missing, x="Column", y="Missing %",
    title="Missing Value Rate (%) — Top 20 Columns",
    color="Missing %",
    color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
)
fig6.update_layout(
    plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e",
    font_color="#ecf0f1",
    xaxis_tickangle=-45,
)
st.plotly_chart(fig6, use_container_width=True)

# ─── Saved EDA images ────────────────────────────────────────
st.markdown('<div class="section-title">Saved EDA Figures</div>', unsafe_allow_html=True)
eda_figs = [
    ("eda_class_distribution.png", "Class Distribution"),
    ("eda_transaction_amt.png",    "Transaction Amount"),
    ("eda_fraud_over_time.png",    "Fraud Over Time"),
    ("eda_fraud_by_product.png",   "Fraud by Product"),
]
cols = st.columns(2)
for i, (fname, caption) in enumerate(eda_figs):
    if fig_exists(fname):
        with cols[i % 2]:
            st.image(str(fig_path(fname)), caption=caption, use_column_width=True)
