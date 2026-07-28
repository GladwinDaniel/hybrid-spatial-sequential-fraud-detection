"""
src/sequential.py — Phase 3: Spatial-Context-Aware Sequential Pattern Mining.

Core novelty: PrefixSpan is run per spatial cluster with an adaptive
minimum support threshold inversely proportional to that cluster's risk score.

Key outputs (saved to data/processed/):
  - pattern_libraries.json : Per-cluster frequent + suspicious pattern libraries
  - anomaly_scores.csv     : Anomaly score per transaction

Key figures (saved to outputs/figures/):
  - anomaly_scores.png
  - adaptive_support.png
"""

import sys
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    PATTERN_JSON, ANOMALY_CSV, FIGURES_DIR, DATA_PROCESSED, REPORTS_DIR,
    PREFIXSPAN_BASE_SUPPORT, SUPPORT_FLOOR, SUPPORT_CEILING,
    MIN_SEQUENCE_LENGTH, MIN_CLUSTER_SEQUENCES,
    RANDOM_SEED,
)


# ─────────────────────────────────────────────────────────────
#  3.1  Group Sequences by Cluster
# ─────────────────────────────────────────────────────────────

def group_sequences_by_cluster(
    sequences: List[dict],
    cluster_df: pd.DataFrame,
) -> Dict[int, List[dict]]:
    """
    Partition user sequences by the dominant spatial cluster of their transactions.

    Parameters
    ----------
    sequences  : output of preprocessing.build_user_sequences()
    cluster_df : DataFrame with user_key → cluster_label mapping

    Returns
    -------
    dict mapping cluster_label → list of sequence dicts
    """
    print("[3.1] Grouping sequences by spatial cluster ...")

    # Build lookup: user_key → most common cluster_label
    # Prefer TransactionID join if available, else fall back to addr1
    if "TransactionID" in cluster_df.columns and "cluster_label" in cluster_df.columns:
        # Build user_key in cluster_df the same way preprocessing does
        if "user_key" not in cluster_df.columns:
            cluster_df = cluster_df.copy()
            for col in ["card1", "addr1", "P_emaildomain"]:
                if col not in cluster_df.columns:
                    cluster_df[col] = "na"
            cluster_df["user_key"] = (
                cluster_df["card1"].astype(str) + "_"
                + cluster_df["addr1"].astype(str) + "_"
                + cluster_df["P_emaildomain"].astype(str)
            )
        user_to_cluster = (
            cluster_df.groupby("user_key")["cluster_label"]
            .agg(lambda x: int(x.mode().iloc[0]) if len(x) > 0 else 0)
            .to_dict()
        )
    elif "addr1_hint" in cluster_df.columns or "addr1" in cluster_df.columns:
        addr_col = "addr1_hint" if "addr1_hint" in cluster_df.columns else "addr1"
        user_to_cluster = (
            cluster_df.groupby(addr_col)["cluster_label"]
            .agg(lambda x: int(x.mode().iloc[0]) if len(x) > 0 else 0)
            .to_dict()
        )
        user_to_cluster = {str(k): v for k, v in user_to_cluster.items()}
    else:
        user_to_cluster = {}

    print(f"      user→cluster map size: {len(user_to_cluster):,}")

    cluster_sequences: Dict[int, List[dict]] = defaultdict(list)

    for seq_dict in sequences:
        # Try user_key first (most reliable), then addr1_hint
        uk = str(seq_dict.get("user_key", ""))
        cluster_id = user_to_cluster.get(uk, None)
        if cluster_id is None:
            addr_hint = str(seq_dict.get("addr1_hint", ""))
            cluster_id = user_to_cluster.get(addr_hint, 0)
        seq_dict = {**seq_dict, "cluster_id": int(cluster_id)}
        cluster_sequences[int(cluster_id)].append(seq_dict)

    # Merge clusters that are too small
    small_clusters = [
        cid for cid, seqs in cluster_sequences.items()
        if len(seqs) < MIN_CLUSTER_SEQUENCES
    ]
    if small_clusters:
        print(f"      Merging {len(small_clusters)} small clusters (< {MIN_CLUSTER_SEQUENCES} seqs) into cluster -1")
        for cid in small_clusters:
            cluster_sequences[-1].extend(cluster_sequences.pop(cid))

    total_seqs = sum(len(v) for v in cluster_sequences.values())
    print(f"      Cluster groups: {len(cluster_sequences)} | Total sequences: {total_seqs:,}")
    for cid, seqs in sorted(cluster_sequences.items()):
        fraud_n = sum(1 for s in seqs if s.get("label", 0) == 1)
        print(f"        Cluster {cid:3d}: {len(seqs):5,} sequences | {fraud_n} fraud")

    return dict(cluster_sequences)



# ─────────────────────────────────────────────────────────────
#  3.2  Adaptive Minimum Support
# ─────────────────────────────────────────────────────────────

def compute_adaptive_support(
    base_support: float,
    risk_score: float,
) -> float:
    """
    Adaptive min_support inversely proportional to cluster risk score.

        min_support(Ci) = base_support × (1 / risk_score(Ci))

    Clamped to [SUPPORT_FLOOR, SUPPORT_CEILING].
    """
    if risk_score <= 0:
        risk_score = 1e-3
    adaptive = base_support * (1.0 / risk_score)
    return float(np.clip(adaptive, SUPPORT_FLOOR, SUPPORT_CEILING))


# ─────────────────────────────────────────────────────────────
#  3.3  PrefixSpan Mining Per Cluster
# ─────────────────────────────────────────────────────────────

def run_prefixspan_per_cluster(
    cluster_sequences: Dict[int, List[dict]],
    risk_scores: pd.DataFrame,
    base_support: float = PREFIXSPAN_BASE_SUPPORT,
) -> Dict[int, List[Tuple[int, List]]]:
    """
    Run PrefixSpan on each cluster's sequence subset with adaptive support.

    Returns
    -------
    dict: cluster_id → list of (support_count, pattern) tuples
    """
    try:
        from prefixspan import PrefixSpan
    except ImportError:
        print("❌ prefixspan not installed. Run: pip install prefixspan")
        return {}

    print("[3.3] Running PrefixSpan mining per spatial cluster ...")

    # Build risk score lookup
    if "cluster_label" in risk_scores.columns and "risk_score" in risk_scores.columns:
        rs_lookup = dict(zip(
            risk_scores["cluster_label"].astype(int),
            risk_scores["risk_score"].astype(float),
        ))
    else:
        rs_lookup = {}

    all_patterns: Dict[int, List[Tuple[int, List]]] = {}
    adaptive_rows = []

    for cluster_id, seq_dicts in sorted(cluster_sequences.items()):
        risk_score = rs_lookup.get(cluster_id, 1.0)
        min_sup_frac = compute_adaptive_support(base_support, risk_score)

        # Convert sequences to PrefixSpan format: list of lists of hashable items
        seqs_for_mining = [
            [str(item) for item in seq_dict["sequence"]]
            for seq_dict in seq_dicts
        ]

        n = len(seqs_for_mining)
        min_sup_count = max(1, int(min_sup_frac * n))

        print(f"      Cluster {cluster_id:3d}: {n:5,} seqs | "
              f"risk={risk_score:.2f} | min_sup={min_sup_frac:.3f} ({min_sup_count} occurrences)")

        try:
            ps = PrefixSpan(seqs_for_mining)
            patterns = ps.frequent(min_sup_count)
            all_patterns[cluster_id] = patterns
            print(f"               → {len(patterns):,} frequent patterns found")
        except Exception as e:
            print(f"               ⚠️  PrefixSpan error: {e}")
            all_patterns[cluster_id] = []

        adaptive_rows.append({
            "cluster_id":    cluster_id,
            "n_sequences":   n,
            "risk_score":    risk_score,
            "min_sup_frac":  min_sup_frac,
            "min_sup_count": min_sup_count,
            "n_patterns":    len(all_patterns.get(cluster_id, [])),
        })

    # Save adaptive support table
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    adaptive_df = pd.DataFrame(adaptive_rows)
    adaptive_df.to_csv(REPORTS_DIR / "adaptive_support.csv", index=False)
    print(f"\n      Adaptive support table saved.")

    # Plot adaptive support
    _plot_adaptive_support(adaptive_df)

    return all_patterns


def _plot_adaptive_support(adaptive_df: pd.DataFrame) -> None:
    """Bar chart: min_support per cluster coloured by risk tier."""
    if adaptive_df.empty:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(adaptive_df) * 0.5), 5))
    colors = [
        "#e74c3c" if rs >= 2.0 else "#f39c12" if rs >= 0.5 else "#2ecc71"
        for rs in adaptive_df["risk_score"]
    ]
    bars = ax.bar(
        adaptive_df["cluster_id"].astype(str),
        adaptive_df["min_sup_frac"] * 100,
        color=colors, edgecolor="white", linewidth=0.8,
    )
    ax.axhline(PREFIXSPAN_BASE_SUPPORT * 100, color="gray", linestyle="--",
               linewidth=1, label=f"Base support ({PREFIXSPAN_BASE_SUPPORT*100:.0f}%)")
    ax.set_title("Adaptive PrefixSpan min_support per Spatial Cluster",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Minimum Support (%)")
    ax.legend()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "adaptive_support.png", dpi=150)
    plt.close(fig)
    print("      Saved: adaptive_support.png")


# ─────────────────────────────────────────────────────────────
#  3.4  Build Pattern Libraries
# ─────────────────────────────────────────────────────────────

def build_pattern_libraries(
    all_patterns: Dict[int, List[Tuple[int, List]]],
    cluster_sequences: Dict[int, List[dict]],
    suspicious_quantile: float = 0.25,
) -> Dict[int, dict]:
    """
    Split patterns into 'normal' (high-support) and 'suspicious' (low-support).

    The bottom `suspicious_quantile` of support values → suspicious patterns.

    Returns
    -------
    dict: cluster_id → {"normal": [...], "suspicious": [...], "max_support": int}
    """
    print("[3.4] Building pattern libraries ...")

    libraries: Dict[int, dict] = {}

    for cluster_id, patterns in all_patterns.items():
        if not patterns:
            libraries[cluster_id] = {"normal": [], "suspicious": [], "max_support": 1}
            continue

        supports = [p[0] for p in patterns]
        threshold = np.quantile(supports, suspicious_quantile)
        max_support = max(supports)

        normal = [(cnt, pat) for cnt, pat in patterns if cnt > threshold]
        suspicious = [(cnt, pat) for cnt, pat in patterns if cnt <= threshold]

        libraries[cluster_id] = {
            "normal":      [(cnt, pat) for cnt, pat in normal],
            "suspicious":  [(cnt, pat) for cnt, pat in suspicious],
            "max_support": int(max_support),
            "n_sequences": len(cluster_sequences.get(cluster_id, [])),
        }

        print(f"      Cluster {cluster_id:3d}: "
              f"{len(normal)} normal patterns | {len(suspicious)} suspicious patterns")

    return libraries


# ─────────────────────────────────────────────────────────────
#  3.5  Anomaly Scoring
# ─────────────────────────────────────────────────────────────

def _match_score(sequence: List[str], library: List[Tuple[int, List]], max_support: int) -> float:
    """
    Compute max match support for a sequence against a pattern library.
    A pattern matches if it is a subsequence of the input sequence.
    Returns normalised support [0, 1].
    """
    if not library or max_support == 0:
        return 0.0

    seq_str = " ".join(sequence)
    best = 0

    for support, pattern in library:
        pat_str = " ".join(str(p) for p in pattern)
        # Subsequence check (linear scan)
        si, pi = 0, 0
        seq_list = sequence
        pat_list = [str(p) for p in pattern]
        pi = 0
        for item in seq_list:
            if pi < len(pat_list) and item == pat_list[pi]:
                pi += 1
        if pi == len(pat_list):  # full pattern matched
            best = max(best, support)

    return best / max_support


def compute_anomaly_score(
    user_sequence: List[str],
    cluster_id: int,
    libraries: Dict[int, dict],
) -> float:
    """
    Anomaly score: AS = 1 - max_match_support(sequence, cluster_library).
    AS close to 1 → highly anomalous. AS close to 0 → matches normal patterns.
    """
    lib = libraries.get(cluster_id, libraries.get(-1, {}))
    if not lib:
        return 0.5  # Unknown cluster → neutral score

    normal_lib   = lib.get("normal", [])
    max_support  = lib.get("max_support", 1)
    match        = _match_score(user_sequence, normal_lib, max_support)
    return float(1.0 - match)


def batch_compute_anomaly_scores(
    sequences: List[dict],
    cluster_sequences: Dict[int, List[dict]],
    libraries: Dict[int, dict],
) -> pd.DataFrame:
    """
    Compute anomaly scores for all sequences.

    Returns
    -------
    pd.DataFrame with columns: user_key, cluster_id, anomaly_score, label
    """
    print("[3.5] Computing anomaly scores for all sequences ...")

    rows = []
    for seq_dict in sequences:
        cluster_id = int(seq_dict.get("cluster_id", 0))
        sequence   = [str(item) for item in seq_dict.get("sequence", [])]
        as_score   = compute_anomaly_score(sequence, cluster_id, libraries)
        rows.append({
            "user_key":     seq_dict.get("user_key", ""),
            "cluster_id":   cluster_id,
            "anomaly_score": as_score,
            "label":        seq_dict.get("label", 0),
            "n_txn":        seq_dict.get("n_txn", 0),
        })

    scores_df = pd.DataFrame(rows)

    fraud_mean = scores_df.loc[scores_df["label"] == 1, "anomaly_score"].mean()
    legit_mean = scores_df.loc[scores_df["label"] == 0, "anomaly_score"].mean()
    print(f"      Mean anomaly score — Fraud: {fraud_mean:.4f} | Legitimate: {legit_mean:.4f}")

    _plot_anomaly_scores(scores_df)
    return scores_df


def _plot_anomaly_scores(scores_df: pd.DataFrame) -> None:
    """Violin + KDE (or histogram fallback) of anomaly score distribution."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Violin plot
    plot_df = scores_df.copy()
    plot_df["label_str"] = plot_df["label"].map({0: "Legitimate", 1: "Fraud"})
    try:
        sns.violinplot(
            data=plot_df, x="label_str", y="anomaly_score",
            hue="label_str",
            palette={"Legitimate": "#3498db", "Fraud": "#e74c3c"},
            ax=axes[0], inner="box", legend=False,
        )
    except Exception:
        # Fallback to boxplot if violin fails (e.g. constant data)
        sns.boxplot(
            data=plot_df, x="label_str", y="anomaly_score",
            hue="label_str",
            palette={"Legitimate": "#3498db", "Fraud": "#e74c3c"},
            ax=axes[0], legend=False,
        )
    axes[0].set_title("Anomaly Score Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Transaction Label")
    axes[0].set_ylabel("Anomaly Score")

    # KDE overlay — fall back to histogram if data is constant
    for label_val, color, label_name in [(0, "#3498db", "Legitimate"), (1, "#e74c3c", "Fraud")]:
        data = scores_df.loc[scores_df["label"] == label_val, "anomaly_score"]
        if len(data) < 2:
            continue
        try:
            if data.nunique() < 2:          # Constant → KDE undefined
                raise ValueError("constant")
            data.plot(kind="kde", ax=axes[1], color=color, label=label_name, linewidth=2)
        except Exception:
            # Histogram fallback
            axes[1].hist(data, bins=20, color=color, alpha=0.6, label=label_name, density=True)
    axes[1].set_title("Anomaly Score Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Anomaly Score")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    fig.suptitle("Sequence Anomaly Scores — Fraud vs. Legitimate", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "anomaly_scores.png", dpi=150)
    plt.close(fig)
    print("      Saved: anomaly_scores.png")



# ─────────────────────────────────────────────────────────────
#  Serialization helpers
# ─────────────────────────────────────────────────────────────

def _libraries_to_json_safe(libraries: Dict[int, dict]) -> dict:
    """Convert libraries to JSON-serializable format."""
    safe = {}
    for cid, lib in libraries.items():
        safe[str(cid)] = {
            "normal":      [[int(cnt), [str(i) for i in pat]] for cnt, pat in lib.get("normal", [])],
            "suspicious":  [[int(cnt), [str(i) for i in pat]] for cnt, pat in lib.get("suspicious", [])],
            "max_support": int(lib.get("max_support", 1)),
            "n_sequences": int(lib.get("n_sequences", 0)),
        }
    return safe


def _libraries_from_json(data: dict) -> Dict[int, dict]:
    """Restore libraries from JSON-safe format."""
    libs = {}
    for cid_str, lib in data.items():
        cid = int(cid_str)
        libs[cid] = {
            "normal":      [(cnt, pat) for cnt, pat in lib.get("normal", [])],
            "suspicious":  [(cnt, pat) for cnt, pat in lib.get("suspicious", [])],
            "max_support": lib.get("max_support", 1),
            "n_sequences": lib.get("n_sequences", 0),
        }
    return libs


# ─────────────────────────────────────────────────────────────
#  Main Runner
# ─────────────────────────────────────────────────────────────

def run_sequential_mining(
    sequences: List[dict],
    cluster_df: pd.DataFrame,
    risk_scores: pd.DataFrame,
    force: bool = False,
) -> dict:
    """
    Run full sequential mining pipeline. Returns dict with outputs.
    """
    if not force and PATTERN_JSON.exists() and ANOMALY_CSV.exists():
        print("[Phase 3] ✅ Checkpoint found — loading from disk ...")
        with open(PATTERN_JSON, "r") as f:
            libraries = _libraries_from_json(json.load(f))
        scores_df = pd.read_csv(ANOMALY_CSV)
        print(f"           Libraries: {len(libraries)} clusters | "
              f"Anomaly scores: {len(scores_df):,} rows")
        return {"libraries": libraries, "scores_df": scores_df}

    print("\n" + "="*60)
    print("PHASE 3 — Sequential Pattern Mining")
    print("="*60)

    cluster_sequences = group_sequences_by_cluster(sequences, cluster_df)

    # Assign cluster_id back to sequences list
    cluster_id_map: Dict[str, int] = {}
    for cid, seq_list in cluster_sequences.items():
        for s in seq_list:
            cluster_id_map[s.get("user_key", "")] = cid
    for s in sequences:
        s["cluster_id"] = cluster_id_map.get(s.get("user_key", ""), 0)

    all_patterns = run_prefixspan_per_cluster(cluster_sequences, risk_scores)
    libraries    = build_pattern_libraries(all_patterns, cluster_sequences)
    scores_df    = batch_compute_anomaly_scores(sequences, cluster_sequences, libraries)

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(PATTERN_JSON, "w") as f:
        json.dump(_libraries_to_json_safe(libraries), f, indent=2)
    scores_df.to_csv(ANOMALY_CSV, index=False)
    print(f"[3.6] Saved: pattern_libraries.json + anomaly_scores.csv")

    return {"libraries": libraries, "scores_df": scores_df}


if __name__ == "__main__":
    from config import SEQUENCES_PKL, CLUSTER_CSV
    with open(SEQUENCES_PKL, "rb") as f:
        sequences = pickle.load(f)
    cluster_df = pd.read_csv(CLUSTER_CSV)
    risk_scores = pd.read_csv(ROOT / "outputs" / "reports" / "cluster_risk_scores.csv")
    result = run_sequential_mining(sequences, cluster_df, risk_scores)
    print("\nPhase 3 complete ✅")
