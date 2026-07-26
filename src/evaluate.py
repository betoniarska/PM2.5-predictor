"""
Loads the models saved by train.py and evaluates them against
data/processed/test.parquet | first time test data is touched.

Also computes a naive persistence baseline (predict pm25(t+horizon) = pm25(t))
so the trained models' numbers mean something. A model that barely beats
"assume nothing changes" is not actually adding much value.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.config import RAW_DIR, HORIZON_HOURS

log = logging.getLogger(__name__)

PROCESSED_DIR = RAW_DIR.parent / "processed"
MODELS_DIR = RAW_DIR.parent.parent / "models"

MODEL_NAMES = ["ridge_tuned", "random_forest_tuned", "xgboost_tuned"]


def load_test_data():
    df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    return df


def persistence_baseline(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Predicts pm25(t+horizon) as simply pm25(t) - the naive 'nothing changes' forecast."""
    return df["pm25"].to_numpy()


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_all():
    df = load_test_data()
    bundle = joblib.load(MODELS_DIR / f"{MODEL_NAMES[0]}.joblib")
    horizon = bundle["horizon"]
    target_col = f"pm25_target_{horizon}h"
    y_true = df[target_col]

    results = {}

    baseline_pred = persistence_baseline(df, horizon)
    results["persistence_baseline"] = compute_metrics(y_true, baseline_pred)

    for name in MODEL_NAMES:
        bundle = joblib.load(MODELS_DIR / f"{name}.joblib")
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]
        X_test = df[feature_cols]
        y_pred = model.predict(X_test)
        results[name] = compute_metrics(y_true, y_pred)

    return results


def print_results(results: dict):

    print(f"{'model':<22}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    for name, m in results.items():
        print(f"{name:<22}{m['MAE']:>10.3f}{m['RMSE']:>10.3f}{m['R2']:>10.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
 
    from src.logger import log_results
 
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
 
    results = evaluate_all()
    print_results(results)
 
    bundle = joblib.load(MODELS_DIR / f"{MODEL_NAMES[0]}.joblib")
    horizon = bundle["horizon"]
 
    # Log the results to a CSV file for later analysis.
    log_results(
        horizon=horizon,
        n_train=len(train_df),
        n_test=len(test_df),
        train_start=train_df["datetime_utc"].min(),
        train_end=train_df["datetime_utc"].max(),
        test_start=test_df["datetime_utc"].min(),
        test_end=test_df["datetime_utc"].max(),
        results=results,
        run_tag="",  # fill in e.g. "1yr_untuned", "1yr_tuned" when calling manually
    )