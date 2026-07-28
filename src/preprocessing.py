"""
src/preprocessing.py — Phase 1: Data Loading, Cleaning, Feature Engineering,
and Transaction Sequence Construction.

Key outputs (saved to data/processed/):
  - features.pkl       : Clean merged feature DataFrame
  - sequences.pkl      : List of (user_key, sequence, label) tuples
  - train_test_split.pkl : (X_train, X_test, y_train, y_test) DataFrames
"""

import sys
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    TRANSACTION_FILE, IDENTITY_FILE, FEATURES_PKL, SEQUENCES_PKL,
    TRAIN_TEST_PKL, FIGURES_DIR, DATA_PROCESSED,
    USER_KEY_COLS, SPATIAL_COLS, D_COLS, C_COLS,
    AMOUNT_BINS, AMOUNT_LABELS, TIME_DELTA_BINS, TIME_DELTA_LABELS,
    TRAIN_RATIO, RANDOM_SEED, USE_SAMPLE, SAMPLE_SIZE,
    MIN_SEQUENCE_LENGTH,
)

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _ensure_dirs():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  1.1  Load & Merge
# ─────────────────────────────────────────────────────────────

def load_and_merge(
    sample: bool = USE_SAMPLE,
    n: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Load train_transaction.csv + train_identity.csv, merge on TransactionID.

    Parameters
    ----------
    sample : bool
        If True, sample `n` rows from the transaction file (time-ordered).
    n : int
        Number of rows to sample when sample=True.

    Returns
    -------
    pd.DataFrame — merged DataFrame sorted by TransactionDT.
    """
    print(f"[1.1] Loading transaction data from {TRANSACTION_FILE.name} ...")
    df_trans = pd.read_csv(TRANSACTION_FILE, dtype={"TransactionID": str}, encoding="utf-8")
    print(f"      Transactions: {len(df_trans):,} rows × {df_trans.shape[1]} cols")

    print(f"[1.1] Loading identity data from {IDENTITY_FILE.name} ...")
    df_id = pd.read_csv(IDENTITY_FILE, dtype={"TransactionID": str})
    print(f"      Identity:     {len(df_id):,} rows × {df_id.shape[1]} cols")

    # Time-ordered sample (take first n rows, then sample maintains temporal order)
    if sample:
        df_trans = df_trans.sort_values("TransactionDT").head(n).reset_index(drop=True)
        print(f"      Sampled:      {len(df_trans):,} rows (sample mode)")

    df = df_trans.merge(df_id, on="TransactionID", how="left")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    fraud_rate = df["isFraud"].mean() * 100
    print(f"[1.1] Merged: {len(df):,} rows | Fraud rate: {fraud_rate:.2f}%")
    return df


# ─────────────────────────────────────────────────────────────
#  1.2  Missing Value Imputation
# ─────────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - addr1, addr2 → mode (most frequent category)
    - dist1, dist2 → median
    - D cols       → median (or 0 if all NaN)
    - C cols       → 0
    - Identity cols → -1 (missing indicator)
    - P_emaildomain → 'unknown'
    """
    print("[1.2] Handling missing values ...")

    # Spatial address codes
    for col in ["addr1", "addr2"]:
        if col in df.columns:
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
            df[col] = df[col].fillna(fill_val)

    # Spatial distance metrics
    for col in ["dist1", "dist2"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # D columns — days-since features
    for col in D_COLS:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med if not pd.isna(med) else 0)

    # C columns — count features
    for col in C_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Email domain
    if "P_emaildomain" in df.columns:
        df["P_emaildomain"] = df["P_emaildomain"].fillna("unknown")
    if "R_emaildomain" in df.columns:
        df["R_emaildomain"] = df["R_emaildomain"].fillna("unknown")

    # Card columns
    for col in ["card4", "card6"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # ProductCD
    if "ProductCD" in df.columns:
        df["ProductCD"] = df["ProductCD"].fillna("unknown")

    # Identity numeric columns — fill with -1 (missing indicator)
    id_cols = [c for c in df.columns if c.startswith("id_")]
    for col in id_cols:
        if df[col].dtype == object:
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(-1)

    # DeviceType, DeviceInfo
    for col in ["DeviceType", "DeviceInfo"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # V columns — Vesta-engineered features, fill with 0
    v_cols = [c for c in df.columns if c.startswith("V")]
    for col in v_cols:
        df[col] = df[col].fillna(0)

    remaining_na = df.isnull().sum().sum()
    print(f"      Remaining NaN cells: {remaining_na:,}")
    return df


# ─────────────────────────────────────────────────────────────
#  1.3  Spatial Feature Engineering
# ─────────────────────────────────────────────────────────────

def engineer_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare spatial features for DBSCAN:
    - Label-encode addr1 and addr2 (already numeric mostly, but may have NaN after encoding)
    - MinMax-normalize dist1, dist2 to [0, 1]
    - Store encoded cols as addr1_enc, addr2_enc, dist1_norm, dist2_norm
    """
    print("[1.3] Engineering spatial features ...")

    le = LabelEncoder()

    # addr1, addr2 may already be numeric; cast to string to handle mixed types
    for col, enc_col in [("addr1", "addr1_enc"), ("addr2", "addr2_enc")]:
        if col in df.columns:
            df[enc_col] = le.fit_transform(df[col].astype(str))

    scaler = MinMaxScaler()
    for col, norm_col in [("dist1", "dist1_norm"), ("dist2", "dist2_norm")]:
        if col in df.columns:
            df[norm_col] = scaler.fit_transform(df[[col]])
        else:
            df[norm_col] = 0.0

    # P_emaildomain — encode as integer for downstream use
    if "P_emaildomain" in df.columns:
        df["email_enc"] = le.fit_transform(df["P_emaildomain"].astype(str))

    return df


# ─────────────────────────────────────────────────────────────
#  1.4  Sequence Construction
# ─────────────────────────────────────────────────────────────

def build_user_sequences(df: pd.DataFrame) -> List[dict]:
    """
    Construct transaction sequences per composite user identity.

    User key: card1 + addr1 + P_emaildomain

    Each sequence item (itemset) is a tuple:
        (amount_bin, ProductCD, time_delta_bin, hour_of_day)

    Returns
    -------
    List of dicts with keys:
        user_key, cluster_hint, sequence (list of tuples), label (0/1), n_fraud
    """
    print("[1.4] Building user transaction sequences ...")

    # Bin TransactionAmt
    df["amount_bin"] = pd.cut(
        df["TransactionAmt"],
        bins=AMOUNT_BINS,
        labels=AMOUNT_LABELS,
        right=False,
    ).astype(str)

    # Derive approximate datetime offset → hour of day
    # TransactionDT is seconds since a reference point (not real time, but relative)
    df["hour_of_day"] = (df["TransactionDT"] // 3600 % 24).astype(int)

    # Composite user key
    df["user_key"] = (
        df["card1"].astype(str) + "_"
        + df["addr1"].astype(str) + "_"
        + df["P_emaildomain"].astype(str)
    )

    # Time delta between transactions (per user)
    df = df.sort_values(["user_key", "TransactionDT"])
    df["time_delta"] = df.groupby("user_key")["TransactionDT"].diff().fillna(0)

    df["time_delta_bin"] = pd.cut(
        df["time_delta"],
        bins=TIME_DELTA_BINS,
        labels=TIME_DELTA_LABELS,
        right=False,
    ).astype(str)

    # Build sequence per user
    sequences = []
    grouped = df.groupby("user_key")

    for user_key, grp in grouped:
        grp = grp.sort_values("TransactionDT")
        seq = list(
            zip(
                grp["amount_bin"].tolist(),
                grp["ProductCD"].astype(str).tolist(),
                grp["time_delta_bin"].tolist(),
                grp["hour_of_day"].astype(str).tolist(),
            )
        )
        if len(seq) < MIN_SEQUENCE_LENGTH:
            continue

        n_fraud = int(grp["isFraud"].sum())
        label = 1 if n_fraud > 0 else 0

        # Dominant spatial cluster hint: addr1 mode for this user
        addr1_mode = grp["addr1"].mode()
        addr1_val = str(addr1_mode.iloc[0]) if len(addr1_mode) > 0 else "unknown"

        sequences.append({
            "user_key":     user_key,
            "addr1_hint":   addr1_val,
            "sequence":     seq,
            "label":        label,
            "n_txn":        len(seq),
            "n_fraud":      n_fraud,
        })

    fraud_seqs = sum(1 for s in sequences if s["label"] == 1)
    print(f"      Total sequences: {len(sequences):,} | Fraud sequences: {fraud_seqs:,}")
    return sequences


# ─────────────────────────────────────────────────────────────
#  1.5  Train / Test Split (Time-Based)
# ─────────────────────────────────────────────────────────────

def time_based_split(
    df: pd.DataFrame,
    ratio: float = TRAIN_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/test based on TransactionDT order (no shuffle).
    First `ratio` fraction → train; remainder → test.
    """
    print(f"[1.5] Time-based train/test split ({int(ratio*100)}/{int((1-ratio)*100)}) ...")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * ratio)
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    print(f"      Train: {len(train):,} rows | Test: {len(test):,} rows")
    print(f"      Train fraud rate: {train['isFraud'].mean()*100:.2f}% | "
          f"Test fraud rate: {test['isFraud'].mean()*100:.2f}%")
    return train, test


# ─────────────────────────────────────────────────────────────
#  1.6  EDA Plots
# ─────────────────────────────────────────────────────────────

def run_eda_plots(df: pd.DataFrame) -> None:
    """Save EDA figures to outputs/figures/."""
    print("[1.6] Generating EDA plots ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    palette = ["#2ecc71", "#e74c3c"]

    # --- Fig 1: Fraud class distribution ---
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["isFraud"].value_counts()
    bars = ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values,
                  color=["#3498db", "#e74c3c"], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Transaction Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_class_distribution.png", dpi=150)
    plt.close(fig)

    # --- Fig 2: TransactionAmt distribution (log scale) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, fraud_val, label, color in zip(
        axes, [0, 1], ["Legitimate", "Fraud"], ["#3498db", "#e74c3c"]
    ):
        data = df.loc[df["isFraud"] == fraud_val, "TransactionAmt"].dropna()
        ax.hist(np.log1p(data), bins=50, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(f"log(1 + TransactionAmt) — {label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("log(1 + Amount)")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_transaction_amt.png", dpi=150)
    plt.close(fig)

    # --- Fig 3: Missing value heatmap (top-30 cols with most missing) ---
    missing = df.isnull().mean().sort_values(ascending=False).head(30)
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        missing.plot(kind="bar", ax=ax, color="#e67e22", edgecolor="white")
        ax.set_title("Top-30 Columns by Missing Rate", fontsize=13, fontweight="bold")
        ax.set_ylabel("Missing Fraction")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "eda_missing_values.png", dpi=150)
        plt.close(fig)

    # --- Fig 4: Fraud rate by ProductCD ---
    if "ProductCD" in df.columns:
        prod_fraud = df.groupby("ProductCD")["isFraud"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        prod_fraud.plot(kind="bar", ax=ax, color="#9b59b6", edgecolor="white")
        ax.set_title("Fraud Rate by ProductCD", fontsize=13, fontweight="bold")
        ax.set_ylabel("Fraud Rate")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "eda_fraud_by_product.png", dpi=150)
        plt.close(fig)

    # --- Fig 5: Fraud rate over time (binned by TransactionDT) ---
    df["dt_bin"] = pd.cut(df["TransactionDT"], bins=50)
    time_fraud = df.groupby("dt_bin")["isFraud"].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(time_fraud)), time_fraud.values, color="#e74c3c", linewidth=1.5)
    ax.fill_between(range(len(time_fraud)), time_fraud.values, alpha=0.2, color="#e74c3c")
    ax.set_title("Fraud Rate Over Time (TransactionDT bins)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Bin →")
    ax.set_ylabel("Fraud Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_fraud_over_time.png", dpi=150)
    plt.close(fig)

    print(f"      EDA plots saved to {FIGURES_DIR}")


# ─────────────────────────────────────────────────────────────
#  Main Runner
# ─────────────────────────────────────────────────────────────

def run_preprocessing(sample: bool = USE_SAMPLE, force: bool = False) -> dict:
    """
    Run full preprocessing pipeline. Returns dict of outputs.
    Uses checkpointing — skips if outputs already exist (unless force=True).
    """
    _ensure_dirs()

    if not force and FEATURES_PKL.exists() and SEQUENCES_PKL.exists() and TRAIN_TEST_PKL.exists():
        print("[Phase 1] ✅ Checkpoint found — loading from disk ...")
        with open(FEATURES_PKL, "rb") as f:
            df = pickle.load(f)
        with open(SEQUENCES_PKL, "rb") as f:
            sequences = pickle.load(f)
        with open(TRAIN_TEST_PKL, "rb") as f:
            train, test = pickle.load(f)
        print(f"           Features: {df.shape} | Sequences: {len(sequences):,}")
        return {"df": df, "sequences": sequences, "train": train, "test": test}

    print("\n" + "="*60)
    print("PHASE 1 — Data Preprocessing")
    print("="*60)

    df = load_and_merge(sample=sample)
    df = handle_missing_values(df)
    df = engineer_spatial_features(df)
    sequences = build_user_sequences(df)
    train, test = time_based_split(df)
    run_eda_plots(df)

    # Save checkpoints
    print("[1.7] Saving checkpoints ...")
    with open(FEATURES_PKL, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(SEQUENCES_PKL, "wb") as f:
        pickle.dump(sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(TRAIN_TEST_PKL, "wb") as f:
        pickle.dump((train, test), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      features.pkl, sequences.pkl, train_test_split.pkl saved.")

    return {"df": df, "sequences": sequences, "train": train, "test": test}


if __name__ == "__main__":
    result = run_preprocessing()
    print("\nPhase 1 complete ✅")
