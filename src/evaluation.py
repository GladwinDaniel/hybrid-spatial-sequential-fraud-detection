"""
src/evaluation.py — Phase 5: Model Evaluation, Comparison, and Reporting.

Computes AUC-ROC, PR-AUC, F1, Precision@K for all 4 model variants:
  1. Baseline XGBoost (raw features only)
  2. Sequential-only  (raw + anomaly_score, no spatial)
  3. Spatial-only     (raw + cluster/risk features, no anomaly_score)
  4. Proposed Hybrid  (all features)

Key figures (saved to outputs/figures/):
  - roc_comparison.png
  - pr_comparison.png
  - confusion_matrices.png

Key reports (saved to outputs/reports/):
  - final_results.csv
"""

import sys
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    FIGURES_DIR, REPORTS_DIR, RESULTS_CSV,
    XGB_PARAMS, RF_PARAMS, LR_PARAMS,
    TOP_K_PERCENT, RANDOM_SEED,
)

MODEL_COLORS = {
    "Baseline XGBoost":   "#95a5a6",
    "Sequential-only":    "#3498db",
    "Spatial-only":       "#f39c12",
    "Proposed Hybrid":    "#e74c3c",
}


# ─────────────────────────────────────────────────────────────
#  5.1  Compute Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    top_k: float = TOP_K_PERCENT,
    model_name: str = "",
) -> dict:
    """
    Compute all evaluation metrics.

    Returns dict with keys:
        auc_roc, pr_auc, f1, precision, recall, precision_at_k
    """
    y_pred = (y_proba >= threshold).astype(int)

    auc_roc  = roc_auc_score(y_true, y_proba)
    pr_auc   = average_precision_score(y_true, y_proba)
    f1       = f1_score(y_true, y_pred, zero_division=0)
    prec     = precision_score(y_true, y_pred, zero_division=0)
    rec      = recall_score(y_true, y_pred, zero_division=0)

    # Precision @ top-K%
    k = max(1, int(top_k * len(y_true)))
    top_k_idx = np.argsort(y_proba)[::-1][:k]
    prec_at_k = y_true[top_k_idx].mean()

    return {
        "model":          model_name,
        "auc_roc":        round(auc_roc, 4),
        "pr_auc":         round(pr_auc, 4),
        "f1":             round(f1, 4),
        "precision":      round(prec, 4),
        "recall":         round(rec, 4),
        f"prec@top{int(top_k*100)}%": round(prec_at_k, 4),
    }


# ─────────────────────────────────────────────────────────────
#  5.2  Build Baseline Feature Subsets
# ─────────────────────────────────────────────────────────────

def _drop_cols(X: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    return X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors="ignore")


def _retrain_xgb(X_train, y_train, X_test, y_test):
    """Retrain XGBoost on a given feature subset and return predicted probabilities."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return np.zeros(len(y_test))

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos = neg / max(pos, 1)
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos}
    params.pop("use_label_encoder", None)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    return model.predict_proba(X_test)[:, 1]


def compare_baselines(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    hybrid_model,
) -> pd.DataFrame:
    """
    Train and evaluate 4 model variants and return comparison DataFrame.
    """
    print("[5.2] Comparing baseline models ...")

    SPATIAL_COLS  = ["cluster_label", "risk_score", "addr1_enc", "addr2_enc",
                     "dist1_norm", "dist2_norm"]
    ANOMALY_COLS  = ["anomaly_score"]
    COMBINED_DROP = SPATIAL_COLS + ANOMALY_COLS

    results = []

    # 1. Baseline XGBoost — raw features only
    print("      Training Baseline XGBoost (no spatial/sequential) ...")
    X_tr_base = _drop_cols(X_train, COMBINED_DROP)
    X_te_base = _drop_cols(X_test,  COMBINED_DROP)
    proba_base = _retrain_xgb(X_tr_base, y_train, X_te_base, y_test)
    results.append(compute_metrics(y_test, proba_base, model_name="Baseline XGBoost"))

    # 2. Sequential-only — raw + anomaly_score, no spatial
    print("      Training Sequential-only ...")
    X_tr_seq = _drop_cols(X_train, SPATIAL_COLS)
    X_te_seq = _drop_cols(X_test,  SPATIAL_COLS)
    proba_seq = _retrain_xgb(X_tr_seq, y_train, X_te_seq, y_test)
    results.append(compute_metrics(y_test, proba_seq, model_name="Sequential-only"))

    # 3. Spatial-only — raw + spatial, no anomaly_score
    print("      Training Spatial-only ...")
    X_tr_spa = _drop_cols(X_train, ANOMALY_COLS)
    X_te_spa = _drop_cols(X_test,  ANOMALY_COLS)
    proba_spa = _retrain_xgb(X_tr_spa, y_train, X_te_spa, y_test)
    results.append(compute_metrics(y_test, proba_spa, model_name="Spatial-only"))

    # 4. Proposed Hybrid — all features
    print("      Evaluating Proposed Hybrid ...")
    proba_hyb = hybrid_model.predict_proba(X_test)[:, 1]
    results.append(compute_metrics(y_test, proba_hyb, model_name="Proposed Hybrid"))

    results_df = pd.DataFrame(results)

    # Store probas for curve plots
    results_df._probas = {
        "Baseline XGBoost": proba_base,
        "Sequential-only":  proba_seq,
        "Spatial-only":     proba_spa,
        "Proposed Hybrid":  proba_hyb,
    }

    # Save results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n      Results saved to final_results.csv")
    print(results_df.to_string(index=False))
    return results_df


# ─────────────────────────────────────────────────────────────
#  5.3  ROC Curves
# ─────────────────────────────────────────────────────────────

def plot_roc_curves(
    probas: Dict[str, np.ndarray],
    y_test: np.ndarray,
) -> None:
    """Overlay ROC curves for all model variants."""
    print("[5.3] Plotting ROC curves ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.50)")

    for model_name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        lw  = 2.5 if model_name == "Proposed Hybrid" else 1.5
        ax.plot(fpr, tpr,
                color=MODEL_COLORS.get(model_name, "#7f8c8d"),
                linewidth=lw,
                label=f"{model_name}  (AUC={auc:.4f})")

    ax.set_title("ROC Curve Comparison — All Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_comparison.png", dpi=150)
    plt.close(fig)
    print("      Saved: roc_comparison.png")


# ─────────────────────────────────────────────────────────────
#  5.4  PR Curves
# ─────────────────────────────────────────────────────────────

def plot_pr_curves(
    probas: Dict[str, np.ndarray],
    y_test: np.ndarray,
) -> None:
    """Overlay Precision-Recall curves for all models."""
    print("[5.4] Plotting PR curves ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    baseline_prec = y_test.mean()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(baseline_prec, color="k", linestyle="--", linewidth=1, alpha=0.5,
               label=f"Random ({baseline_prec:.3f})")

    for model_name, proba in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        lw = 2.5 if model_name == "Proposed Hybrid" else 1.5
        ax.plot(rec, prec,
                color=MODEL_COLORS.get(model_name, "#7f8c8d"),
                linewidth=lw,
                label=f"{model_name}  (PR-AUC={pr_auc:.4f})")

    ax.set_title("Precision-Recall Curve Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_comparison.png", dpi=150)
    plt.close(fig)
    print("      Saved: pr_comparison.png")


# ─────────────────────────────────────────────────────────────
#  5.5  Confusion Matrices
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    probas: Dict[str, np.ndarray],
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> None:
    """2×2 grid of confusion matrices, one per model."""
    print("[5.5] Plotting confusion matrices ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_names = list(probas.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for ax, model_name in zip(axes, model_names):
        proba = probas[model_name]
        y_pred = (proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax,
            cmap="Blues", linewidths=0.5, linecolor="white",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
            cbar=False,
        )
        f1  = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, proba)
        ax.set_title(f"{model_name}\nAUC={auc:.4f} | F1={f1:.4f}",
                     fontsize=11, fontweight="bold",
                     color=MODEL_COLORS.get(model_name, "#2c3e50"))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("      Saved: confusion_matrices.png")


# ─────────────────────────────────────────────────────────────
#  5.6  LaTeX Results Table
# ─────────────────────────────────────────────────────────────

def generate_results_table(results_df: pd.DataFrame) -> str:
    """Return LaTeX table string for the results."""
    latex = results_df.to_latex(index=False, float_format="%.4f",
                                caption="Model Comparison — AUC-ROC, PR-AUC, F1, Precision\\@10\\%",
                                label="tab:results")
    out_path = REPORTS_DIR / "results_table.tex"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex)
    print(f"      LaTeX table saved: {out_path.name}")
    return latex


# ─────────────────────────────────────────────────────────────
#  Main Runner
# ─────────────────────────────────────────────────────────────

def run_evaluation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    hybrid_model,
    force: bool = False,
) -> dict:
    """
    Run full evaluation phase. Returns results DataFrame and probability dict.
    """
    if not force and RESULTS_CSV.exists():
        print("[Phase 5] ✅ Checkpoint found — loading results from disk ...")
        results_df = pd.read_csv(RESULTS_CSV)
        print(results_df.to_string(index=False))
        return {"results_df": results_df}

    print("\n" + "="*60)
    print("PHASE 5 — Evaluation & Comparison")
    print("="*60)

    results_df = compare_baselines(X_train, y_train, X_test, y_test, hybrid_model)

    # Extract stored probas
    probas = getattr(results_df, "_probas", {})

    if probas:
        plot_roc_curves(probas, y_test)
        plot_pr_curves(probas, y_test)
        plot_confusion_matrices(probas, y_test)
    else:
        print("      ⚠️  No probabilities found for curve plots")

    generate_results_table(results_df)

    print("\n[Phase 5] ✅ Evaluation complete")
    return {"results_df": results_df, "probas": probas}


if __name__ == "__main__":
    import pickle
    from config import TRAIN_TEST_PKL, CLUSTER_CSV, ANOMALY_CSV, MODELS_DIR
    from src.classification import build_feature_vector, load_model

    with open(TRAIN_TEST_PKL, "rb") as f:
        train_df, test_df = pickle.load(f)
    cluster_df = pd.read_csv(CLUSTER_CSV)
    scores_df  = pd.read_csv(ANOMALY_CSV)

    X_train, y_train = build_feature_vector(train_df, cluster_df, scores_df)
    X_test,  y_test  = build_feature_vector(test_df,  cluster_df, scores_df)
    hybrid_model = load_model("xgb")

    result = run_evaluation(X_train, y_train, X_test, y_test, hybrid_model)
    print("\nPhase 5 complete ✅")
