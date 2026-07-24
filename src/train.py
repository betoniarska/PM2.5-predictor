"""
Trains three model families on data/processed/train.parquet:
  - Ridge regression (linear baseline, scaled features)
  - Random Forest
  - XGBoost
"""

import logging
from pathlib import Path

import joblib

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import RAW_DIR

log = logging.getLogger(__name__)

PROCESSED_DIR = RAW_DIR.parent / "processed"
MODELS_DIR = RAW_DIR.parent.parent / "models"

HORIZON_HOURS = 24
TARGET_COL = f"pm25_target_{HORIZON_HOURS}h"
NON_FEATURE_COLS = {"datetime_utc", TARGET_COL}


def load_train_data():
    df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]
    return X, y, feature_cols


def build_models() -> dict:
    return {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,          # capped depth - ~2000 rows / ~50 features overfits fast unrestricted
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_and_save(X, y, feature_cols):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = build_models()

    for name, model in models.items():
        log.info("Fitting %s on %d rows, %d features", name, len(X), len(feature_cols))
        model.fit(X, y)
        out_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump({"model": model, "feature_cols": feature_cols, "horizon": HORIZON_HOURS}, out_path)
        log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    X, y, feature_cols = load_train_data()
    train_and_save(X, y, feature_cols)