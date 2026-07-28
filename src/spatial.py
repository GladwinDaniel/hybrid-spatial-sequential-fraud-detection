"""
src/spatial.py — Phase 2: Spatial Clustering via DBSCAN.

Assigns every transaction a spatial cluster label and computes
a cluster-level fraud risk score used to calibrate PrefixSpan in Phase 3.

Key outputs (saved to data/processed/):
  - cluster_labels.csv : TransactionID → cluster_label, risk_score, risk_tier

Key figures (saved to outputs/figures/):
  - cluster_pca.png
  - cluster_umap.png (if umap-learn installed)
  - risk_heatmap.png
  - epsilon_plot.png
"""

import sys
import pickle
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    CLUSTER_CSV, FIGURES_DIR, DATA_PROCESSED, REPORTS_DIR,
    DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
    HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD,
    RANDOM_SEED,
)

SPATIAL_FEATURE_COLS = ["addr1_enc", "addr2_enc", "dist1_norm", "dist2_norm"]


# ─────────────────────────────────────────────────────────────
#  2.1  Epsilon Tuning
# ─────────────────────────────────────────────────────────────

def tune_epsilon(
    features: np.ndarray,
    k: int = 5,
    save_fig: bool = True,
) -> float:
    """
    Use k-distance graph to recommend an epsilon for DBSCAN.
    Uses the 95th percentile of k-NN distances as a robust elbow estimate,
    avoiding the instability of second-derivative methods on large datasets.
    """
    print(f"[2.1] Tuning DBSCAN epsilon via k-distance graph (k={k}) ...")

    # Sub-sample for speed if features is large
    MAX_TUNE = 10_000
    if len(features) > MAX_TUNE:
        idx = np.random.choice(len(features), MAX_TUNE, replace=False)
        feat_sample = features[idx]
    else:
        feat_sample = features

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1).fit(feat_sample)
    distances, _ = nbrs.kneighbors(feat_sample)
    k_distances = np.sort(distances[:, k - 1])

    # Use the "knee" as the point where distances start rising sharply
    # Robust method: use a percentile that captures the natural density threshold
    # For fraud data, the 90th percentile of k-NN distances gives a good eps
    recommended_eps = float(np.percentile(k_distances, 90))
    # Clamp to a sensible range for standardized features
    recommended_eps = float(np.clip(recommended_eps, 0.1, 2.0))

    # Find approximate elbow index for plotting
    elbow_idx = np.searchsorted(k_distances, recommended_eps)

    print(f"      90th-percentile k-NN distance → recommended ε = {recommended_eps:.4f}")

    if save_fig:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(k_distances, color="#3498db", linewidth=1.5, label=f"{k}-NN distance")
        ax.axvline(elbow_idx, color="#e74c3c", linestyle="--",
                   label=f"Elbow (ε ≈ {recommended_eps:.3f})")
        ax.axhline(recommended_eps, color="#e74c3c", linestyle=":", alpha=0.5)
        ax.set_title(f"k-Distance Graph (k={k}) for DBSCAN Epsilon Tuning",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Points sorted by distance")
        ax.set_ylabel(f"{k}-NN Distance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "epsilon_plot.png", dpi=150)
        plt.close(fig)
        print(f"      Saved: epsilon_plot.png")

    return recommended_eps



# ─────────────────────────────────────────────────────────────
#  2.2  DBSCAN Clustering
# ─────────────────────────────────────────────────────────────

def run_dbscan(
    df: pd.DataFrame,
    eps: Optional[float] = None,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit DBSCAN on spatial feature columns.
    Sub-samples to MAX_DBSCAN points for fitting to avoid O(n²) memory,
    then assigns remaining points via 1-NN.

    Returns
    -------
    labels     : np.ndarray — cluster label per row (-1 = noise)
    features   : np.ndarray — scaled spatial feature matrix
    """
    MAX_DBSCAN = 20_000   # Max points for DBSCAN fitting
    print("[2.2] Running DBSCAN spatial clustering ...")

    # Extract and scale
    available = [c for c in SPATIAL_FEATURE_COLS if c in df.columns]
    features_raw = df[available].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features_raw)

    # Tune eps if not provided
    if eps is None:
        try:
            eps = tune_epsilon(features)
        except Exception as e:
            print(f"      Epsilon tuning failed ({e}). Using default eps=0.5")
            eps = 0.5

    print(f"      DBSCAN params: eps={eps:.4f}, min_samples={min_samples}")
    print(f"      Dataset size: {len(features):,} points")

    # Sub-sample for DBSCAN if dataset is large
    if len(features) > MAX_DBSCAN:
        print(f"      Sub-sampling to {MAX_DBSCAN:,} points for DBSCAN fitting ...")
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(features), MAX_DBSCAN, replace=False)
        feat_sample = features[sample_idx]

        db = DBSCAN(eps=eps, min_samples=min_samples,
                    algorithm='ball_tree', n_jobs=-1)
        sample_labels = db.fit_predict(feat_sample)

        n_clusters = len(set(sample_labels)) - (1 if -1 in sample_labels else 0)
        n_noise = int((sample_labels == -1).sum())
        print(f"      Clusters found on sample: {n_clusters} | Noise: {n_noise:,}")

        # Assign full dataset via 1-NN to sample
        print(f"      Assigning all {len(features):,} points via 1-NN to sample labels ...")
        core_mask = sample_labels != -1
        if core_mask.sum() == 0:
            # All noise — fall back to single cluster 0
            print("      WARNING: All sample points are noise. Lowering eps and retrying ...")
            eps_retry = eps * 2.0
            db2 = DBSCAN(eps=eps_retry, min_samples=2, algorithm='ball_tree', n_jobs=-1)
            sample_labels = db2.fit_predict(feat_sample)
            core_mask = sample_labels != -1

        if core_mask.sum() > 0:
            knn = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', n_jobs=-1)
            knn.fit(feat_sample[core_mask], sample_labels[core_mask])
            labels = knn.predict(features)
        else:
            labels = np.zeros(len(features), dtype=int)
    else:
        db = DBSCAN(eps=eps, min_samples=min_samples,
                    algorithm='ball_tree', n_jobs=-1)
        labels = db.fit_predict(features)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"      Final: {n_clusters} clusters | {n_noise:,} noise points")

    return labels, features



# ─────────────────────────────────────────────────────────────
#  2.3  Assign Noise Points to Nearest Cluster
# ─────────────────────────────────────────────────────────────

def assign_noise_to_nearest(
    labels: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    """
    Reclassify noise points (-1) to the nearest non-noise cluster centroid
    using 1-NN classification.
    """
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels

    print(f"[2.3] Assigning {noise_mask.sum():,} noise points to nearest cluster ...")
    labels = labels.copy()
    core_mask = ~noise_mask

    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn.fit(features[core_mask], labels[core_mask])
    labels[noise_mask] = knn.predict(features[noise_mask])

    print(f"      All noise points reassigned. Remaining noise: {(labels == -1).sum()}")
    return labels


# ─────────────────────────────────────────────────────────────
#  2.4  Cluster Risk Scoring
# ─────────────────────────────────────────────────────────────

def compute_cluster_risk_scores(
    df: pd.DataFrame,
    labels: np.ndarray,
    label_col: str = "isFraud",
) -> pd.DataFrame:
    """
    Compute per-cluster fraud risk score:
        risk_score(Ci) = fraud_rate(Ci) / global_fraud_rate

    Returns DataFrame with columns: cluster_label, size, fraud_count,
    fraud_rate, risk_score, risk_tier
    """
    print("[2.4] Computing cluster risk scores ...")

    df = df.copy()
    df["cluster_label"] = labels

    global_fraud_rate = df[label_col].mean()
    if global_fraud_rate == 0:
        global_fraud_rate = 1e-6  # Avoid division by zero

    stats = (
        df.groupby("cluster_label")
        .agg(
            size=(label_col, "count"),
            fraud_count=(label_col, "sum"),
        )
        .reset_index()
    )
    stats["fraud_rate"] = stats["fraud_count"] / stats["size"]
    stats["risk_score"] = stats["fraud_rate"] / global_fraud_rate

    def _tier(rs):
        if rs >= HIGH_RISK_THRESHOLD:
            return "HIGH"
        elif rs <= LOW_RISK_THRESHOLD:
            return "LOW"
        return "MEDIUM"

    stats["risk_tier"] = stats["risk_score"].apply(_tier)
    stats = stats.sort_values("risk_score", ascending=False).reset_index(drop=True)

    print(f"      Risk score summary:")
    tier_counts = stats["risk_tier"].value_counts()
    for tier, cnt in tier_counts.items():
        print(f"        {tier}: {cnt} clusters")

    return stats


# ─────────────────────────────────────────────────────────────
#  2.5  Visualization
# ─────────────────────────────────────────────────────────────

def visualize_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    risk_scores: pd.DataFrame,
) -> None:
    """Save PCA 2D cluster scatter plot."""
    print("[2.5] Visualizing clusters ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(features)
    explained = pca.explained_variance_ratio_ * 100

    unique_labels = sorted(set(labels))
    cmap = plt.colormaps["tab20"].resampled(max(len(unique_labels), 1))

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = cmap(i)
        cluster_name = f"C{label}" if label >= 0 else "Noise"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=4, alpha=0.5, color=color, label=cluster_name, rasterized=True,
        )

    ax.set_title(
        f"Spatial Clusters — PCA Projection\n"
        f"(PC1: {explained[0]:.1f}% | PC2: {explained[1]:.1f}% variance explained)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    if len(unique_labels) <= 20:
        ax.legend(markerscale=4, loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_pca.png", dpi=150)
    plt.close(fig)
    print("      Saved: cluster_pca.png")

    # UMAP (optional)
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=RANDOM_SEED, n_jobs=1, verbose=False)
        umap_coords = reducer.fit_transform(features[:min(20000, len(features))])
        sub_labels = labels[:min(20000, len(labels))]

        fig2, ax2 = plt.subplots(figsize=(10, 7))
        for i, label in enumerate(sorted(set(sub_labels))):
            mask = sub_labels == label
            ax2.scatter(
                umap_coords[mask, 0], umap_coords[mask, 1],
                s=4, alpha=0.5, color=cmap(i), rasterized=True,
                label=f"C{label}" if label >= 0 else "Noise",
            )
        ax2.set_title("Spatial Clusters — UMAP Projection", fontsize=13, fontweight="bold")
        ax2.set_xlabel("UMAP-1"); ax2.set_ylabel("UMAP-2")
        if len(unique_labels) <= 20:
            ax2.legend(markerscale=4, loc="best", fontsize=8, ncol=2)
        fig2.tight_layout()
        fig2.savefig(FIGURES_DIR / "cluster_umap.png", dpi=150)
        plt.close(fig2)
        print("      Saved: cluster_umap.png")
    except ImportError:
        print("      umap-learn not installed — skipping UMAP plot")


def plot_risk_heatmap(risk_scores: pd.DataFrame) -> None:
    """Bar chart of cluster risk scores coloured by tier."""
    print("[2.5] Plotting risk score heatmap ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tier_colors = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}
    colors = [tier_colors.get(t, "#95a5a6") for t in risk_scores["risk_tier"]]

    top_n = min(30, len(risk_scores))
    subset = risk_scores.head(top_n)

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.4), 5))
    bars = ax.bar(
        subset["cluster_label"].astype(str),
        subset["risk_score"],
        color=[tier_colors.get(t, "#95a5a6") for t in subset["risk_tier"]],
        edgecolor="white", linewidth=0.8,
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Global average (1.0)")
    ax.axhline(HIGH_RISK_THRESHOLD, color="#e74c3c", linestyle=":", linewidth=1,
               label=f"High-risk threshold ({HIGH_RISK_THRESHOLD}x)")
    ax.axhline(LOW_RISK_THRESHOLD, color="#2ecc71", linestyle=":", linewidth=1,
               label=f"Low-risk threshold ({LOW_RISK_THRESHOLD}x)")
    ax.set_title("Cluster Fraud Risk Scores", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Risk Score (relative to global fraud rate)")
    ax.legend(fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "risk_heatmap.png", dpi=150)
    plt.close(fig)
    print("      Saved: risk_heatmap.png")


# ─────────────────────────────────────────────────────────────
#  Main Runner
# ─────────────────────────────────────────────────────────────

def run_spatial_clustering(df: pd.DataFrame, force: bool = False) -> dict:
    """
    Run full spatial clustering pipeline. Returns dict with cluster outputs.
    """
    if not force and CLUSTER_CSV.exists():
        print("[Phase 2] ✅ Checkpoint found — loading cluster labels from disk ...")
        cluster_df = pd.read_csv(CLUSTER_CSV)
        print(f"           {len(cluster_df):,} rows loaded from cluster_labels.csv")
        # Reconstruct risk_scores summary
        risk_scores = (
            cluster_df.groupby("cluster_label")
            .agg(size=("cluster_label", "count"),
                 fraud_count=("isFraud", "sum") if "isFraud" in cluster_df.columns
                 else ("cluster_label", "count"))
            .reset_index()
        )
        if "risk_score" in cluster_df.columns:
            risk_scores = cluster_df[["cluster_label", "risk_score", "risk_tier"]].drop_duplicates("cluster_label")
        return {"cluster_df": cluster_df, "risk_scores": risk_scores}

    print("\n" + "="*60)
    print("PHASE 2 — Spatial Clustering")
    print("="*60)

    labels, features = run_dbscan(df)
    labels = assign_noise_to_nearest(labels, features)
    risk_scores = compute_cluster_risk_scores(df, labels)
    visualize_clusters(features, labels, risk_scores)
    plot_risk_heatmap(risk_scores)

    # Build output DataFrame
    cluster_df = df[["TransactionID", "isFraud"]].copy() if "isFraud" in df.columns else df[["TransactionID"]].copy()
    cluster_df["cluster_label"] = labels

    # Merge risk scores
    cluster_df = cluster_df.merge(
        risk_scores[["cluster_label", "risk_score", "risk_tier"]],
        on="cluster_label", how="left",
    )

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(CLUSTER_CSV, index=False)
    print(f"[2.6] Saved: cluster_labels.csv ({len(cluster_df):,} rows)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    risk_scores.to_csv(REPORTS_DIR / "cluster_risk_scores.csv", index=False)
    print(f"      Saved: cluster_risk_scores.csv")

    return {"cluster_df": cluster_df, "risk_scores": risk_scores}


if __name__ == "__main__":
    import pickle
    from config import FEATURES_PKL
    with open(FEATURES_PKL, "rb") as f:
        df = pickle.load(f)
    result = run_spatial_clustering(df)
    print("\nPhase 2 complete ✅")
