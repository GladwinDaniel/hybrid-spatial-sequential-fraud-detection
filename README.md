# Spatial-Context-Aware Sequential Pattern Mining for Financial Fraud Detection

This project explores whether adding spatial context to sequential transaction mining improves fraud detection on the IEEE-CIS dataset. The pipeline combines DBSCAN-based spatial clustering, adaptive PrefixSpan pattern mining, and a hybrid classification model that uses both raw transaction features and contextual fraud signals.

The project is designed to run from the command line so the workflow stays focused on the research pipeline, model training, and outputs.

---

## 1. Project goal

The core hypothesis is that fraud signals become more informative when they are evaluated in a spatial and sequential context instead of using transaction features alone.

In practice, this means:
- spatial clusters capture location-based behavior patterns,
- sequential anomaly patterns capture unusual transaction trajectories,
- the hybrid model combines these signals with standard fraud features.

---

## 2. Key pipeline components

1. Data preprocessing and feature engineering
2. DBSCAN spatial clustering using address and distance features
3. Adaptive PrefixSpan mining with cluster-aware minimum support
4. Hybrid classification using CatBoost with contextual features
5. Evaluation against baseline and ablation models

The key innovation is the adaptive support logic:

```python
min_support(Ci) = base_support * (1 / risk_score(Ci))
```

This lowers the support threshold for high-risk clusters and raises it for lower-risk clusters, allowing the model to surface more suspicious patterns in risk-dense areas while reducing noise in low-risk regions.

---

## 3. Key findings from the generated outputs

The final evaluation results saved in `outputs/reports/final_results.csv` show that the proposed hybrid model outperforms the baseline and ablation variants across the key metrics.

| Model | AUC-ROC | PR-AUC | F1 | Precision | Recall | precision@top10% |
|---|---:|---:|---:|---:|---:|---:|
| Baseline XGBoost | 0.8836 | 0.5149 | 0.2320 | 0.1375 | 0.7408 | 0.1580 |
| Sequential-only | 0.8824 | 0.5229 | 0.2278 | 0.1346 | 0.7408 | 0.1580 |
| Spatial-only | 0.8829 | 0.5287 | 0.2416 | 0.1450 | 0.7225 | 0.1540 |
| Proposed Hybrid | 0.8919 | 0.5576 | 0.2579 | 0.1558 | 0.7477 | 0.1610 |

### Main takeaways
- The hybrid model achieved the best AUC-ROC of 0.8919, improving over the baseline by about +0.0083.
- The hybrid model achieved the best PR-AUC of 0.5576, improving over the baseline by about +0.0427.
- The hybrid model achieved the best F1 score of 0.2579, improving over baseline by about +0.0259.
- The hybrid model also produced the strongest precision at the top 10% risk threshold: 0.1610 vs 0.1580 baseline.

These results suggest the spatial-sequential fusion provides measurable gains in fraud-ranking quality and classification performance compared with the raw-feature-only baseline.

---

## 4. Ablation study

The ablation setup compares four variants trained under the same evaluation pipeline:

1. Baseline XGBoost — raw transaction features only
2. Sequential-only — raw features plus anomaly score
3. Spatial-only — raw features plus spatial cluster/risk features
4. Proposed Hybrid — complete feature set with spatial and sequential context

### Ablation performance summary

| Variant | AUC-ROC | PR-AUC | F1 |
|---|---:|---:|---:|
| Baseline XGBoost | 0.8836 | 0.5149 | 0.2320 |
| Sequential-only | 0.8824 | 0.5229 | 0.2278 |
| Spatial-only | 0.8829 | 0.5287 | 0.2416 |
| Proposed Hybrid | 0.8919 | 0.5576 | 0.2579 |

### Interpretation
- Spatial context provides a stronger lift than sequential context in this setup: the spatial-only model has better PR-AUC and F1 than the sequential-only model.
- The hybrid model is the best performer, which indicates that spatial context and sequential anomaly information are complementary rather than redundant.
- The improvement is especially visible in precision-recall metrics, which are more informative under class imbalance than a raw accuracy score.

---

## 5. How to run the project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download the IEEE-CIS dataset

```bash
python scripts/download_data.py
```

### Run the pipeline

```bash
# Sample run (default)
python pipeline.py

# Full dataset run
python pipeline.py --full

# Re-run from scratch
python pipeline.py --force

# Run only selected phases
python pipeline.py --phase 1 2 3 4 5
```

### Output locations

- Reports: `outputs/reports/`
- Figures: `outputs/figures/`
- Trained models: `outputs/models/`
- Pipeline log: `outputs/pipeline.log`

---

## 6. Notes

- This research uses the IEEE-CIS Fraud Detection dataset from Kaggle / Vesta Corporation.
- The project assumes the Kaggle API credentials are configured correctly before the dataset download step.
- The final results are taken from the generated output files in `outputs/reports/final_results.csv` and represent the current experimental run for this repository state.

---

## 7. Files in the repository

```text
├── config.py
├── pipeline.py
├── requirements.txt
├── scripts/
│   └── download_data.py
├── src/
│   ├── preprocessing.py
│   ├── spatial.py
│   ├── sequential.py
│   ├── classification.py
│   └── evaluation.py
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
└── README.md
```
