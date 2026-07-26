"""
Append-only CSV logging for evaluate.py and feature_importance.py.

Two files, results/results_log.csv and results/importance_log.csv, each
grow by one row (or one row per feature) per run. Deliberately dead simple -
plain pandas concat + overwrite - since these files stay small and are meant
to be read back by a notebook, not queried at scale.
"""

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

log = logging.getLogger(__name__)

RESULTS_DIR = RAW_DIR.parent.parent / "results"
RESULTS_LOG_PATH = RESULTS_DIR / "results_log.csv"
IMPORTANCE_LOG_PATH = RESULTS_DIR / "importance_log.csv"


def _append_rows(path: Path, rows: pd.DataFrame):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, rows], ignore_index=True)
    else:
        combined = rows
    combined.to_csv(path, index=False)
    log.info("Logged %d row(s) -> %s", len(rows), path)


def log_results(
    horizon: int,
    n_train: int,
    n_test: int,
    train_start, train_end, test_start, test_end,
    results: dict,
    run_tag: str = "",
):
    """
    results: {model_name: {"MAE": ..., "RMSE": ..., "R2": ...}}
    """
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    rows = pd.DataFrame([
        {
            "timestamp": timestamp,
            "run_tag": run_tag,
            "horizon_hours": horizon,
            "n_train": n_train,
            "n_test": n_test,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "model": model_name,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "R2": m["R2"],
        }
        for model_name, m in results.items()
    ])
    _append_rows(RESULTS_LOG_PATH, rows)


def log_importances(model_name: str, df: pd.DataFrame, horizon: int, run_tag: str = ""):
    """
    df: output of feature_importance.load_importances - columns
    [feature, importance, is_weather]
    """
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    rows = df.copy()
    rows.insert(0, "timestamp", timestamp)
    rows.insert(1, "run_tag", run_tag)
    rows.insert(2, "horizon_hours", horizon)
    rows.insert(3, "model", model_name)
    _append_rows(IMPORTANCE_LOG_PATH, rows)