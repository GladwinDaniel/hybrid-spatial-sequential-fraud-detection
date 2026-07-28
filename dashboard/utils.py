"""
dashboard/utils.py — Shared helpers for the Streamlit dashboard.
Loads all pre-computed artifacts from outputs/.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _p(rel: str) -> Path:
    return ROOT / rel


def load_csv(rel_path: str) -> Optional[pd.DataFrame]:
    p = _p(rel_path)
    if p.exists():
        return pd.read_csv(p)
    return None


def load_json(rel_path: str) -> Optional[dict]:
    p = _p(rel_path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def load_pkl(rel_path: str):
    p = _p(rel_path)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def fig_path(name: str) -> Path:
    return ROOT / "outputs" / "figures" / name


def fig_exists(name: str) -> bool:
    return fig_path(name).exists()


def report_path(name: str) -> Path:
    return ROOT / "outputs" / "reports" / name


def pipeline_ran() -> bool:
    """Return True if at least Phase 1 has been run (features.pkl exists)."""
    return _p("data/processed/features.pkl").exists()


def get_results() -> Optional[pd.DataFrame]:
    return load_csv("outputs/reports/final_results.csv")


def get_cluster_risk() -> Optional[pd.DataFrame]:
    return load_csv("outputs/reports/cluster_risk_scores.csv")


def get_adaptive_support() -> Optional[pd.DataFrame]:
    return load_csv("outputs/reports/adaptive_support.csv")


def get_anomaly_scores() -> Optional[pd.DataFrame]:
    return load_csv("data/processed/anomaly_scores.csv")


def get_cluster_labels() -> Optional[pd.DataFrame]:
    return load_csv("data/processed/cluster_labels.csv")


def get_pattern_libraries() -> Optional[dict]:
    return load_json("data/processed/pattern_libraries.json")


TIER_COLORS = {
    "HIGH":   "#e74c3c",
    "MEDIUM": "#f39c12",
    "LOW":    "#2ecc71",
}

MODEL_COLORS = {
    "Baseline XGBoost": "#95a5a6",
    "Sequential-only":  "#3498db",
    "Spatial-only":     "#f39c12",
    "Proposed Hybrid":  "#e74c3c",
}
