"""
dashboard/app.py — Main Streamlit entry point.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import pipeline_ran, get_results

# ─────────────────────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
}
[data-testid="stSidebar"] * {
    color: #ecf0f1 !important;
}

/* Main background */
.main .block-container {
    background: #0d1117;
    padding-top: 2rem;
}

/* Hero card */
.hero-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(52, 152, 219, 0.3);
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #3498db, #9b59b6, #e74c3c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    color: #bdc3c7;
    font-size: 1rem;
    line-height: 1.6;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(52,152,219,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(52,152,219,0.6);
}
.metric-label {
    color: #7f8c8d;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #3498db;
}
.metric-delta {
    font-size: 0.75rem;
    color: #2ecc71;
    margin-top: 0.2rem;
}

/* Phase badge */
.phase-badge {
    display: inline-block;
    background: rgba(52,152,219,0.15);
    border: 1px solid rgba(52,152,219,0.4);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    color: #3498db;
    font-weight: 600;
    margin: 0.2rem;
}

/* Warning banner */
.run-warning {
    background: linear-gradient(90deg, rgba(231,76,60,0.15), rgba(192,57,43,0.1));
    border-left: 4px solid #e74c3c;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    color: #ecf0f1;
    margin-bottom: 1.5rem;
}

/* Info banner */
.info-banner {
    background: rgba(52,152,219,0.1);
    border-left: 4px solid #3498db;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #bdc3c7;
    font-size: 0.9rem;
}

h1, h2, h3, h4 {
    color: #ecf0f1 !important;
}
p, li, label {
    color: #bdc3c7;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Fraud Detection")
    st.markdown("**Spatial-Context-Aware**  \nSequential Pattern Mining")
    st.divider()
    st.markdown("### 📋 Navigation")
    st.markdown("""
Use the **Pages** menu above to explore:

- 📊 **Overview** — EDA & dataset stats
- 🗺️ **Spatial Analysis** — Cluster maps & risk scores
- 🔁 **Sequential Patterns** — Pattern libraries & anomaly scores
- 🤖 **Model Comparison** — ROC, PR, confusion matrices
- 🧠 **Explainability** — SHAP feature importance
    """)
    st.divider()
    st.markdown("### ⚙️ Pipeline")
    st.code("python pipeline.py\npython pipeline.py --full", language="bash")
    st.markdown("### 📂 Dataset")
    st.code("python scripts/download_data.py", language="bash")
    st.divider()
    st.caption("IEEE-CIS Fraud Detection Dataset  \nKaggle / Vesta Corporation")


# ─────────────────────────────────────────────────────────────
#  Main Page
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🔍 Spatial-Context-Aware Fraud Detection</div>
    <div class="hero-subtitle">
        A unified framework integrating <strong>DBSCAN spatial clustering</strong> with
        <strong>adaptive PrefixSpan sequential mining</strong> for interpretable,
        location-sensitive fraud detection on the IEEE-CIS dataset.
    </div>
</div>
""", unsafe_allow_html=True)

# Pipeline status
ran = pipeline_ran()
if not ran:
    st.markdown("""
<div class="run-warning">
    ⚠️ <strong>Pipeline not yet run.</strong><br>
    Please download the dataset and run the pipeline first:<br><br>
    <code>python scripts/download_data.py</code><br>
    <code>python pipeline.py</code>
</div>
""", unsafe_allow_html=True)
else:
    # Summary metrics from results if available
    results = get_results()
    if results is not None and len(results) > 0:
        hybrid_row = results[results["model"] == "Proposed Hybrid"]
        baseline_row = results[results["model"] == "Baseline XGBoost"]

        if len(hybrid_row) > 0:
            h = hybrid_row.iloc[0]
            b = baseline_row.iloc[0] if len(baseline_row) > 0 else None

            cols = st.columns(4)
            metrics = [
                ("AUC-ROC", f"{h['auc_roc']:.4f}",
                 f"+{h['auc_roc'] - b['auc_roc']:.4f} vs baseline" if b is not None else ""),
                ("PR-AUC",  f"{h['pr_auc']:.4f}",
                 f"+{h['pr_auc'] - b['pr_auc']:.4f} vs baseline" if b is not None else ""),
                ("F1-Score", f"{h['f1']:.4f}",
                 f"+{h['f1'] - b['f1']:.4f} vs baseline" if b is not None else ""),
                ("Model",   "Hybrid", "Spatial + Sequential"),
            ]
            for col, (label, val, delta) in zip(cols, metrics):
                with col:
                    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{val}</div>
    <div class="metric-delta">{delta}</div>
</div>""", unsafe_allow_html=True)
        st.write("")

    st.markdown("""
<div class="info-banner">
    ✅ <strong>Pipeline complete.</strong> Use the sidebar pages to explore results.
</div>
""", unsafe_allow_html=True)

st.write("")

# ─── Framework overview ───
st.markdown("## 🏗️ Framework Architecture")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
### Pipeline Phases
<span class="phase-badge">Phase 1</span> Data Preprocessing & Feature Engineering<br>
<span class="phase-badge">Phase 2</span> DBSCAN Spatial Clustering<br>
<span class="phase-badge">Phase 3</span> Adaptive PrefixSpan Sequential Mining<br>
<span class="phase-badge">Phase 4</span> XGBoost Classification + SHAP<br>
<span class="phase-badge">Phase 5</span> Evaluation & Baseline Comparison
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
### Key Novelty
The **core innovation** is the adaptive min_support formula:

```
min_support(Ci) = base_support × (1 / risk_score(Ci))
```

High-risk spatial clusters → lower threshold → more sensitive detection.  
Low-risk clusters → higher threshold → fewer false positives.
""")

st.divider()

# ─── Dataset overview ───
st.markdown("## 📦 Dataset: IEEE-CIS Fraud Detection")
dcols = st.columns(3)
stats = [
    ("Transactions", "~590K", "training records"),
    ("Features", "435", "transaction + identity"),
    ("Fraud Rate", "~3.5%", "highly imbalanced"),
]
for col, (label, val, sub) in zip(dcols, stats):
    with col:
        st.metric(label=label, value=val, delta=sub)

st.markdown("""
| Feature Group | Columns | Role |
|---|---|---|
| **Spatial** | `addr1`, `addr2`, `dist1`, `dist2` | DBSCAN clustering |
| **Sequential** | `TransactionDT`, `D1–D15`, `TransactionAmt` | PrefixSpan sequences |
| **Classification** | `card1–card6`, `C1–C14`, `V1–V339` | XGBoost features |
| **Identity** | `DeviceType`, `id_12–id_38` | Behavioral context |
""")
