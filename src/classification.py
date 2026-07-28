"""
src/classification.py — Phase 4: Feature Vector Construction, Model Training,
and SHAP Explainability.

Trains three models:
  - XGBoost (primary)
  - Random Forest (comparison)
  - Logistic Regression (baseline)

Key outputs (saved to outputs/models/):
  - xgb_model.pkl, rf_model.pkl, lr_model.pkl

Key figures (saved to outputs/figures/):
  - shap_summary.png
  - shap_cluster_<id>.png  (per top-3 clusters)
"""

import sys
import pickle
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
    XGB_PARAMS, RF_PARAMS, LR_PARAMS,
    SHAP_SAMPLE_SIZE, TOP_K_PERCENT, RANDOM_SEED,
    D_COLS, C_COLS,
)


# ─────────────────────────────────────────────────────────────
#  4.1  Feature Vector Construction
# ─────────────────────────────────────────────────────────────

def build_feature_vector(
    df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Assemble the final feature matrix by combining:
      - Spatial context:  cluster_label, risk_score
      - Sequential:       anomaly_score, sequence length proxy (C1 count)
      - Raw transaction:  TransactionAmt, ProductCD, card features, D/C cols
      - Identity:         DeviceType, encoded id cols

    Returns
    -------
    X : pd.DataFrame — feature matrix
    y : np.ndarray   — binary labels
    """
    print("[4.1] Building feature matrix ...")

    df = df.copy()

    # Merge spatial cluster features
    if "TransactionID" in cluster_df.columns:
        # Coerce both to str to avoid int64/str merge type conflict
        df["TransactionID"] = df["TransactionID"].astype(str)
        cluster_df = cluster_df.copy()
        cluster_df["TransactionID"] = cluster_df["TransactionID"].astype(str)
        df = df.merge(
            cluster_df[["TransactionID", "cluster_label", "risk_score"]].drop_duplicates("TransactionID"),
            on="TransactionID", how="left",
        )
    df["cluster_label"] = df.get("cluster_label", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["risk_score"]    = df.get("risk_score", pd.Series(1.0, index=df.index)).fillna(1.0)

    # Merge anomaly scores (keyed by user_key)
    if "user_key" not in df.columns:
        df["user_key"] = (
            df["card1"].astype(str) + "_"
            + df["addr1"].astype(str) + "_"
            + df["P_emaildomain"].astype(str)
        )
    if "user_key" in scores_df.columns and "anomaly_score" in scores_df.columns:
        user_score = scores_df[["user_key", "anomaly_score"]].drop_duplicates("user_key")
        df = df.merge(user_score, on="user_key", how="left")
    df["anomaly_score"] = df.get("anomaly_score", pd.Series(0.5, index=df.index)).fillna(0.5)

    # ── Feature selection ──────────────────────────────────
    feature_cols = []

    # Spatial
    for col in ["cluster_label", "risk_score", "addr1_enc", "addr2_enc",
                "dist1_norm", "dist2_norm"]:
        if col in df.columns:
            feature_cols.append(col)

    # Sequential / anomaly
    feature_cols.append("anomaly_score")

    # Raw transaction
    for col in ["TransactionAmt", "card1", "card2", "card3", "card5"]:
        if col in df.columns:
            feature_cols.append(col)

    # Categorical encode
    for col in ["ProductCD", "card4", "card6", "DeviceType"]:
        if col in df.columns:
            enc_col = col + "_enc"
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[col].astype(str))
            feature_cols.append(enc_col)

    # D and C columns
    for col in D_COLS + C_COLS:
        if col in df.columns:
            feature_cols.append(col)

    # V columns — select first 50 available (feature-rich)
    v_cols = [c for c in df.columns if c.startswith("V")][:50]
    feature_cols.extend(v_cols)

    # De-duplicate
    feature_cols = list(dict.fromkeys(feature_cols))

    X = df[feature_cols].copy()
    # Ensure numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    y = df["isFraud"].values if "isFraud" in df.columns else np.zeros(len(df))

    print(f"      Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"      Fraud rate:     {y.mean()*100:.2f}%")
    return X, y


# ─────────────────────────────────────────────────────────────
#  4.2  XGBoost Training
# ─────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> object:
    """Train XGBoost classifier with class-weight balancing."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("❌ xgboost not installed. Run: pip install xgboost")
        return None

    print("[4.2] Training XGBoost ...")
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos = neg / max(pos, 1)
    print(f"      Class ratio — neg:{neg:,} / pos:{pos:,} → scale_pos_weight={scale_pos:.1f}")

    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos}
    # Remove any param that XGBoost doesn't accept in newer versions
    params.pop("use_label_encoder", None)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    print(f"      XGBoost trained ✅  ({params['n_estimators']} trees)")
    return model


# ─────────────────────────────────────────────────────────────
#  4.3  Random Forest Training
# ─────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    print("[4.3] Training Random Forest ...")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    print(f"      Random Forest trained ✅  ({RF_PARAMS['n_estimators']} trees)")
    return model


# ─────────────────────────────────────────────────────────────
#  4.4  Logistic Regression Baseline
# ─────────────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Pipeline:
    """Train Logistic Regression baseline with imputation pipeline."""
    print("[4.4] Training Logistic Regression (baseline) ...")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("lr",      LogisticRegression(**LR_PARAMS)),
    ])
    pipeline.fit(X_train, y_train)
    print("      Logistic Regression trained ✅")
    return pipeline


# ─────────────────────────────────────────────────────────────
#  4.5  SHAP Explainability
# ─────────────────────────────────────────────────────────────

def run_shap_analysis(
    model,
    X_test: pd.DataFrame,
    cluster_labels: Optional[np.ndarray] = None,
    n_samples: int = SHAP_SAMPLE_SIZE,
) -> None:
    """
    Compute global SHAP values for XGBoost and save:
      - shap_summary.png        : Global beeswarm feature importance
      - shap_cluster_<id>.png   : Per-cluster waterfall for top-3 clusters
    """
    try:
        import shap
    except ImportError:
        print("❌ shap not installed. Run: pip install shap")
        return

    print(f"[4.5] Running SHAP analysis on {min(n_samples, len(X_test)):,} samples ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Sample for speed
    sample_idx = X_test.sample(min(n_samples, len(X_test)),
                                random_state=RANDOM_SEED).index
    X_sample = X_test.loc[sample_idx]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # If binary output has shape (n, features, 2) take the fraud class
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        # ── Global summary plot ──
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, plot_type="dot", show=False,
                          max_display=20, color_bar=True)
        plt.title("SHAP Feature Importance — Global (Fraud Class)", fontsize=13,
                  fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("      Saved: shap_summary.png")

        # ── Bar plot ──
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        shap.summary_plot(sv, X_sample, plot_type="bar", show=False, max_display=20)
        plt.title("SHAP Mean |value| — Feature Ranking", fontsize=13,
                  fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("      Saved: shap_bar.png")

        # ── Per-cluster waterfall (top 3 clusters by size) ──
        if cluster_labels is not None and "cluster_label" in X_sample.columns:
            top_clusters = (
                X_sample["cluster_label"].value_counts().head(3).index.tolist()
            )
            for cid in top_clusters:
                mask = X_sample["cluster_label"] == cid
                if mask.sum() < 5:
                    continue
                sv_cluster = sv[mask.values]
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                shap.summary_plot(sv_cluster, X_sample[mask], plot_type="dot",
                                  show=False, max_display=15)
                plt.title(f"SHAP Summary — Cluster {cid}", fontsize=12,
                          fontweight="bold", pad=12)
                plt.tight_layout()
                out_path = FIGURES_DIR / f"shap_cluster_{cid}.png"
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"      Saved: shap_cluster_{cid}.png")

    except Exception as e:
        print(f"      ⚠️  SHAP analysis failed: {e}")


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def save_model(model, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      Saved: {path.name}")
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name}_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
#  Main Runner
# ─────────────────────────────────────────────────────────────

def run_classification(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    force: bool = False,
) -> dict:
    """
    Run full classification phase. Returns trained models and feature matrices.
    """
    xgb_path = MODELS_DIR / "xgb_model.pkl"
    rf_path  = MODELS_DIR / "rf_model.pkl"
    lr_path  = MODELS_DIR / "lr_model.pkl"

    if not force and xgb_path.exists() and rf_path.exists() and lr_path.exists():
        print("[Phase 4] ✅ Checkpoint found — loading models from disk ...")
        models = {
            "xgb": load_model("xgb"),
            "rf":  load_model("rf"),
            "lr":  load_model("lr"),
        }
        X_train, y_train = build_feature_vector(train_df, cluster_df, scores_df)
        X_test,  y_test  = build_feature_vector(test_df,  cluster_df, scores_df)
        return {"models": models, "X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test}

    print("\n" + "="*60)
    print("PHASE 4 — Classification & Explainability")
    print("="*60)

    X_train, y_train = build_feature_vector(train_df, cluster_df, scores_df)
    X_test,  y_test  = build_feature_vector(test_df,  cluster_df, scores_df)

    xgb_model = train_xgboost(X_train, y_train)
    rf_model  = train_random_forest(X_train, y_train)
    lr_model  = train_logistic_regression(X_train, y_train)

    models = {"xgb": xgb_model, "rf": rf_model, "lr": lr_model}

    # Save models
    for name, model in models.items():
        if model is not None:
            save_model(model, name)

    # SHAP on XGBoost
    if xgb_model is not None:
        cluster_labels = X_test["cluster_label"].values if "cluster_label" in X_test.columns else None
        run_shap_analysis(xgb_model, X_test, cluster_labels=cluster_labels)

    return {"models": models, "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test}


if __name__ == "__main__":
    import pickle
    from config import FEATURES_PKL, TRAIN_TEST_PKL, CLUSTER_CSV, ANOMALY_CSV
    with open(TRAIN_TEST_PKL, "rb") as f:
        train_df, test_df = pickle.load(f)
    cluster_df = pd.read_csv(CLUSTER_CSV)
    scores_df  = pd.read_csv(ANOMALY_CSV)
    result = run_classification(train_df, test_df, cluster_df, scores_df)
    print("\nPhase 4 complete ✅")
