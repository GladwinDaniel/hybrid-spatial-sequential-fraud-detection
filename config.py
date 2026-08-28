"""
config.py — Centralized configuration for the Spatial-Context-Aware
Sequential Pattern Mining pipeline.

All paths, hyperparameters, and flags are defined here.
Override USE_SAMPLE from the CLI via --full flag in pipeline.py.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
#  Project Root
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────────
#  Data Paths
# ─────────────────────────────────────────────
DATA_RAW         = ROOT / "data" / "raw"
DATA_PROCESSED   = ROOT / "data" / "processed"

TRANSACTION_FILE = DATA_RAW / "train_transaction.csv"
IDENTITY_FILE    = DATA_RAW / "train_identity.csv"

# Processed checkpoints
FEATURES_PKL     = DATA_PROCESSED / "features.pkl"
SEQUENCES_PKL    = DATA_PROCESSED / "sequences.pkl"
CLUSTER_CSV      = DATA_PROCESSED / "cluster_labels.csv"
ANOMALY_CSV      = DATA_PROCESSED / "anomaly_scores.csv"
PATTERN_JSON     = DATA_PROCESSED / "pattern_libraries.json"
TRAIN_TEST_PKL   = DATA_PROCESSED / "train_test_split.pkl"

# ─────────────────────────────────────────────
#  Output Paths
# ─────────────────────────────────────────────
OUTPUTS          = ROOT / "outputs"
FIGURES_DIR      = OUTPUTS / "figures"
MODELS_DIR       = OUTPUTS / "models"
REPORTS_DIR      = OUTPUTS / "reports"

RESULTS_CSV      = REPORTS_DIR / "final_results.csv"
ADAPTIVE_CSV     = REPORTS_DIR / "adaptive_support.csv"
PIPELINE_LOG     = OUTPUTS / "pipeline.log"

# ─────────────────────────────────────────────
#  Sampling
# ─────────────────────────────────────────────
USE_SAMPLE   = True        # Set False via --full CLI flag for full 590K run
SAMPLE_SIZE  = 100_000
RANDOM_SEED  = 42

# ─────────────────────────────────────────────
#  Feature Engineering
# ─────────────────────────────────────────────
USER_KEY_COLS   = ["card1", "addr1", "P_emaildomain"]
SPATIAL_COLS    = ["addr1", "addr2", "dist1", "dist2"]
D_COLS          = [f"D{i}" for i in range(1, 16)]
C_COLS          = [f"C{i}" for i in range(1, 15)]
AMOUNT_BINS     = [0, 10, 50, 100, 250, 500, 1000, 5000, float("inf")]
AMOUNT_LABELS   = ["<10", "10-50", "50-100", "100-250", "250-500", "500-1k", "1k-5k", "5k+"]
TIME_DELTA_BINS = [0, 3600, 21600, 86400, 259200, 604800, float("inf")]
TIME_DELTA_LABELS = ["<1h", "1-6h", "6-24h", "1-3d", "3-7d", "7d+"]
TRAIN_RATIO     = 0.80   # Time-based split

# ─────────────────────────────────────────────
#  Phase 2 — DBSCAN
# ─────────────────────────────────────────────
DBSCAN_EPS        = 0.5    # Will be auto-tuned; this is the starting estimate
DBSCAN_MIN_SAMPLES = 5
HIGH_RISK_THRESHOLD = 2.0  # risk_score > this → HIGH risk cluster
LOW_RISK_THRESHOLD  = 0.5  # risk_score < this → LOW risk cluster

# ─────────────────────────────────────────────
#  Phase 3 — Sequential Mining
# ─────────────────────────────────────────────
PREFIXSPAN_BASE_SUPPORT = 0.05   # 5% base minimum support
SUPPORT_FLOOR           = 0.01   # Never go below 1%
SUPPORT_CEILING         = 0.20   # Never go above 20%
MIN_SEQUENCE_LENGTH     = 2      # Ignore single-item "sequences"
MIN_CLUSTER_SEQUENCES   = 50     # Merge clusters smaller than this

# ─────────────────────────────────────────────
#  Phase 4 — Classification
# ─────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":    300,
    "max_depth":       6,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "use_label_encoder": False,
    "eval_metric":     "auc",
    "random_state":    RANDOM_SEED,
    "n_jobs":          -1,
}
CATBOOST_PARAMS = {
    "iterations":       500,
    "depth":             7,
    "learning_rate":     0.05,
    "loss_function":     "Logloss",
    "eval_metric":       "AUC",
    "auto_class_weights": "Balanced",
    "l2_leaf_reg":        5.0,
    "random_strength":    1.0,
    "random_seed":        RANDOM_SEED,
    "thread_count":       -1,
    "verbose":            False,
}
RF_PARAMS = {
    "n_estimators":  200,
    "max_depth":     10,
    "class_weight":  "balanced",
    "random_state":  RANDOM_SEED,
    "n_jobs":        -1,
}
LR_PARAMS = {
    "C":             0.1,
    "max_iter":      1000,
    "class_weight":  "balanced",
    "random_state":  RANDOM_SEED,
}
SHAP_SAMPLE_SIZE = 1000   # Rows to compute SHAP on (speed)
TOP_K_PERCENT    = 0.10   # Precision @ top-10%

# ─────────────────────────────────────────────
#  Kaggle
# ─────────────────────────────────────────────
KAGGLE_COMPETITION = "ieee-fraud-detection"
