# Spatial-Context-Aware Sequential Pattern Mining for Financial Fraud Detection

> **Course Project** | Sequential Data Mining & Spatial Data Mining  
> **Dataset**: IEEE-CIS Fraud Detection (Kaggle / Vesta Corporation)

---

## 🔬 Research Question

> *Can integrating spatial cluster context into sequential transaction pattern mining significantly improve fraud detection performance over approaches that apply each technique independently?*

---

## 🏗️ Framework Architecture

```
IEEE-CIS Dataset
      │
      ▼
┌─────────────────────────┐
│  Phase 1: Preprocessing │  Load, merge, impute, sequence construction
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  Phase 2: DBSCAN        │  ◄── SPATIAL DATA MINING
│  Spatial Clustering     │  addr1, addr2, dist1, dist2 → clusters + risk scores
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  Phase 3: PrefixSpan    │  ◄── SEQUENTIAL DATA MINING (Core Novelty)
│  Adaptive Mining        │  min_support(Ci) = base × (1 / risk_score(Ci))
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  Phase 4: XGBoost       │  Spatial + Sequential + Raw features
│  Classification + SHAP  │  Interpretability via SHAP values
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│  Phase 5: Evaluation    │  4-model comparison: Baseline / Seq-only / Spa-only / Hybrid
└─────────────────────────┘
```

---

## 📁 Project Structure

```
├── data/
│   ├── raw/            # train_transaction.csv, train_identity.csv
│   └── processed/      # Checkpointed pipeline outputs
│
├── scripts/
│   └── download_data.py        # Kaggle API dataset download
│
├── src/
│   ├── preprocessing.py        # Phase 1
│   ├── spatial.py              # Phase 2
│   ├── sequential.py           # Phase 3
│   ├── classification.py       # Phase 4
│   └── evaluation.py           # Phase 5
│
├── dashboard/
│   ├── app.py                  # Streamlit main page
│   └── pages/
│       ├── 1_overview.py       # EDA & dataset stats
│       ├── 2_spatial.py        # Cluster maps & risk scores
│       ├── 3_sequential.py     # Pattern libraries & anomaly scores
│       ├── 4_models.py         # ROC, PR, confusion matrices
│       └── 5_explainability.py # SHAP analysis
│
├── outputs/
│   ├── figures/        # All generated plots (.png)
│   ├── models/         # Trained models (.pkl)
│   └── reports/        # Results CSVs, LaTeX table
│
├── pipeline.py         # End-to-end runner
├── config.py           # Centralized configuration
└── requirements.txt    # All dependencies
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
python scripts/download_data.py
```

**First-time setup for Kaggle API:**
1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → **Create New Token**
2. Move the downloaded `kaggle.json` to `C:\Users\<username>\.kaggle\kaggle.json`
3. Accept competition rules at: https://www.kaggle.com/competitions/ieee-fraud-detection/rules

### 3. Run the pipeline

```bash
# Sample mode (100K rows, ~10-20 min depending on hardware)
python pipeline.py

# Full dataset (~590K rows, may take 1-2 hours)
python pipeline.py --full

# Re-run from scratch (ignore checkpoints)
python pipeline.py --force

# Run specific phases only
python pipeline.py --phase 3 4 5
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📊 Dashboard Pages

| Page | Content |
|---|---|
| **Overview** | Dataset stats, class distribution, transaction amount, fraud over time |
| **Spatial Analysis** | DBSCAN cluster risk scores, PCA/UMAP projections, cluster size vs. fraud rate |
| **Sequential Patterns** | Adaptive support per cluster, anomaly score distributions, pattern library browser |
| **Model Comparison** | AUC-ROC, PR-AUC, F1 table, grouped bar chart, radar chart, ROC/PR/confusion matrix tabs |
| **Explainability** | Global SHAP beeswarm, SHAP bar chart, per-cluster SHAP viewer, interpretation guide |

---

## 🔑 Core Novelty

The **adaptive minimum support formula** is the key contribution:

```python
min_support(Ci) = base_support × (1 / risk_score(Ci))

# where:
# risk_score(Ci) = fraud_rate(Ci) / global_fraud_rate
```

- **High-risk clusters** (risk_score > 2.0) → lower min_support → more patterns extracted → **more sensitive detection**
- **Low-risk clusters** (risk_score < 0.5) → higher min_support → fewer false positives

---

## ⚙️ Configuration (`config.py`)

All hyperparameters are centralized:

| Parameter | Default | Description |
|---|---|---|
| `USE_SAMPLE` | `True` | Sample 100K rows for fast execution |
| `SAMPLE_SIZE` | `100_000` | Rows to sample |
| `DBSCAN_EPS` | `0.5` | DBSCAN epsilon (auto-tuned via k-distance) |
| `DBSCAN_MIN_SAMPLES` | `5` | DBSCAN min cluster size |
| `PREFIXSPAN_BASE_SUPPORT` | `0.05` | Base minimum support (5%) |
| `SUPPORT_FLOOR` | `0.01` | Minimum adaptive support floor |
| `SUPPORT_CEILING` | `0.20` | Maximum adaptive support ceiling |
| `XGB_PARAMS` | see config | XGBoost hyperparameters |

---

## 📈 Evaluation Targets

| Model | AUC-ROC | PR-AUC |
|---|---|---|
| Baseline XGBoost | ~0.87 | ~0.60 |
| Sequential-only | ~0.89 | ~0.65 |
| Spatial-only | ~0.88 | ~0.63 |
| **Proposed Hybrid** | **> 0.92** | **> 0.75** |

---

## 🛠️ Technology Stack

| Category | Tool |
|---|---|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| ML / Classification | scikit-learn, xgboost |
| Sequential Mining | prefixspan |
| Spatial Clustering | scikit-learn (DBSCAN) |
| Class Imbalance | imbalanced-learn |
| Explainability | shap |
| Dimensionality Reduction | umap-learn |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | streamlit |
| Dataset Download | kaggle |

---

## 📚 References

Key papers informing this framework:
1. Liu et al. (2023) — USPM-FrauD: Sequential pattern mining for banking fraud
2. Kumar et al. (2022) — Fraud detection via DBSCAN + Random Forest
3. Zhang et al. (2021) — PrefixSpan for anomalous transaction sequences
4. Rajapaksha & Bergmeir (2023) — Geospatial risk modeling for AML
5. Tian et al. (2024) — CaT-GNN: Causal temporal GNN (SOTA baseline)

*Full literature review in `research_proposal.md`.*

---

*Dataset: IEEE-CIS Fraud Detection (Kaggle). Framework: Spatial-Context-Aware Sequential Pattern Mining.*
