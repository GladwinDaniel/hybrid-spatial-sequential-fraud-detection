"""
pipeline.py — End-to-end runner for the Spatial-Context-Aware Sequential
Pattern Mining pipeline.
"""
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
"""

Usage:
    python pipeline.py              # Sample mode (100K rows, default)
    python pipeline.py --full       # Full dataset (~590K rows)
    python pipeline.py --force      # Re-run all phases, ignore checkpoints
    python pipeline.py --phase 1 2  # Run only specific phases (1-5)

Phases:
    1 — Data Preprocessing
    2 — Spatial Clustering
    3 — Sequential Pattern Mining
    4 — Classification & Explainability
    5 — Evaluation & Comparison
"""

import sys
import time
import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    USE_SAMPLE, SAMPLE_SIZE, PIPELINE_LOG,
    FEATURES_PKL, SEQUENCES_PKL, TRAIN_TEST_PKL,
    CLUSTER_CSV, ANOMALY_CSV, PATTERN_JSON, RESULTS_CSV,
    REPORTS_DIR,
)

# ─────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────

PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(PIPELINE_LOG, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────
#  Phase runners
# ─────────────────────────────────────────────────────────────

def phase1(sample: bool, force: bool) -> dict:
    from src.preprocessing import run_preprocessing
    return run_preprocessing(sample=sample, force=force)


def phase2(artifacts: dict, force: bool) -> dict:
    from src.spatial import run_spatial_clustering
    df = artifacts.get("df")
    if df is None:
        with open(FEATURES_PKL, "rb") as f:
            df = pickle.load(f)
    return run_spatial_clustering(df, force=force)


def phase3(artifacts: dict, force: bool) -> dict:
    from src.sequential import run_sequential_mining
    import json
    sequences   = artifacts.get("sequences")
    cluster_df  = artifacts.get("cluster_df")
    risk_scores = artifacts.get("risk_scores")

    if sequences is None:
        with open(SEQUENCES_PKL, "rb") as f:
            sequences = pickle.load(f)
    if cluster_df is None:
        cluster_df = pd.read_csv(CLUSTER_CSV)
    if risk_scores is None:
        rs_path = REPORTS_DIR / "cluster_risk_scores.csv"
        if rs_path.exists():
            risk_scores = pd.read_csv(rs_path)
        else:
            risk_scores = cluster_df[["cluster_label"]].drop_duplicates()
            risk_scores["risk_score"] = 1.0
            risk_scores["risk_tier"]  = "MEDIUM"

    return run_sequential_mining(sequences, cluster_df, risk_scores, force=force)


def phase4(artifacts: dict, force: bool) -> dict:
    from src.classification import run_classification
    train_df    = artifacts.get("train")
    test_df     = artifacts.get("test")
    cluster_df  = artifacts.get("cluster_df")
    scores_df   = artifacts.get("scores_df")

    if train_df is None or test_df is None:
        with open(TRAIN_TEST_PKL, "rb") as f:
            train_df, test_df = pickle.load(f)
    if cluster_df is None:
        cluster_df = pd.read_csv(CLUSTER_CSV)
    if scores_df is None:
        scores_df = pd.read_csv(ANOMALY_CSV)

    return run_classification(train_df, test_df, cluster_df, scores_df, force=force)


def phase5(artifacts: dict, force: bool) -> dict:
    from src.evaluation import run_evaluation
    from src.classification import build_feature_vector, load_model

    X_train = artifacts.get("X_train")
    y_train = artifacts.get("y_train")
    X_test  = artifacts.get("X_test")
    y_test  = artifacts.get("y_test")
    hybrid_model = artifacts.get("models", {}).get("xgb")

    if X_train is None or X_test is None:
        # Rebuild feature vectors
        with open(TRAIN_TEST_PKL, "rb") as f:
            train_df, test_df = pickle.load(f)
        cluster_df = pd.read_csv(CLUSTER_CSV)
        scores_df  = pd.read_csv(ANOMALY_CSV)
        X_train, y_train = build_feature_vector(train_df, cluster_df, scores_df)
        X_test,  y_test  = build_feature_vector(test_df,  cluster_df, scores_df)

    if hybrid_model is None:
        hybrid_model = load_model("xgb")

    return run_evaluation(X_train, y_train, X_test, y_test, hybrid_model, force=force)


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

PHASE_RUNNERS = {
    1: phase1,
    2: phase2,
    3: phase3,
    4: phase4,
    5: phase5,
}

PHASE_NAMES = {
    1: "Data Preprocessing",
    2: "Spatial Clustering",
    3: "Sequential Pattern Mining",
    4: "Classification & Explainability",
    5: "Evaluation & Comparison",
}


def main():
    parser = argparse.ArgumentParser(
        description="Spatial-Context-Aware Sequential Pattern Mining Pipeline"
    )
    parser.add_argument("--full", action="store_true",
                        help="Run on full dataset (590K). Default: 100K sample.")
    parser.add_argument("--force", action="store_true",
                        help="Ignore checkpoints and re-run all phases from scratch.")
    parser.add_argument("--phase", nargs="+", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only specific phases (e.g. --phase 3 4 5).")
    args = parser.parse_args()

    sample = not args.full
    force  = args.force
    phases = sorted(args.phase) if args.phase else [1, 2, 3, 4, 5]

    log.info("=" * 65)
    log.info("  SPATIAL-CONTEXT-AWARE SEQUENTIAL PATTERN MINING PIPELINE")
    log.info("=" * 65)
    log.info(f"  Mode:   {'FULL (590K)' if not sample else f'SAMPLE ({SAMPLE_SIZE:,})'}")
    log.info(f"  Force:  {force}")
    log.info(f"  Phases: {phases}")
    log.info("=" * 65)

    artifacts = {}
    total_start = time.time()

    for phase_num in phases:
        runner = PHASE_RUNNERS[phase_num]
        name   = PHASE_NAMES[phase_num]

        log.info(f"\n{'─'*65}")
        log.info(f"  PHASE {phase_num}: {name}")
        log.info(f"{'─'*65}")

        t0 = time.time()
        try:
            if phase_num == 1:
                result = runner(sample, force)
            else:
                result = runner(artifacts, force)
            artifacts.update(result or {})
            elapsed = time.time() - t0
            log.info(f"\n  [OK] Phase {phase_num} complete in {elapsed:.1f}s")
        except Exception as e:
            log.error(f"\n  [FAILED] Phase {phase_num}: {e}")
            import traceback
            traceback.print_exc()
            log.error("  Pipeline halted.")
            sys.exit(1)

    total = time.time() - total_start
    log.info(f"\n{'='*65}")
    log.info(f"  ALL PHASES COMPLETE  |  Total time: {total:.1f}s")
    log.info(f"  Outputs saved to: outputs/")
    log.info(f"  Run dashboard:    streamlit run dashboard/app.py")
    log.info(f"{'='*65}\n")


if __name__ == "__main__":
    main()
