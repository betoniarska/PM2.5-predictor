"""
Searches hyperparameters for Ridge, Random Forest, and XGBoost using
RandomizedSearchCV with TimeSeriesSplit as the cross-validator.

TimeSeriesSplit to avoid data leakage, and RandomizedSearchCV to avoid combinatorial explosion of the search space.

Only touches train.parquet. Test set stays untouched until evaluate.py.

Saves tuned models as models/{name}_tuned.joblib, alongside (not overwriting)
the original fixed-config models from train.py, so both can be compared.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from src.config import RAW_DIR, HORIZON_HOURS

log = logging.getLogger(__name__)

PROCESSED_DIR = RAW_DIR.parent / "processed"
MODELS_DIR = RAW_DIR.parent.parent / "models"

TARGET_COL = f"pm25_target_{HORIZON_HOURS}h"
NON_FEATURE_COLS = {"datetime_utc", TARGET_COL}

N_SPLITS = 5
N_ITER = 40
SCORING = "neg_mean_absolute_error"
RANDOM_STATE = 42


def load_train_data():
    df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]
    return X, y, feature_cols


def build_search_spaces() -> dict:
    return {
        "ridge": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]),
            {"model__alpha": loguniform(1e-2, 1e2)},
        ),
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": randint(100, 600),
                "max_depth": [3, 4, 5, 6, 8, 10, None],
                "min_samples_leaf": randint(1, 15),
                "max_features": ["sqrt", "log2", 0.5, None],
            },
        ),
        "xgboost": (
            XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": randint(100, 600),
                "max_depth": randint(2, 8),
                "learning_rate": loguniform(1e-2, 3e-1),
                "subsample": uniform(0.6, 0.4),        # 0.6 to 1.0
                "colsample_bytree": uniform(0.6, 0.4),  # 0.6 to 1.0
                "min_child_weight": randint(1, 10),
            },
        ),
    }


def tune_and_save(X, y, feature_cols):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    spaces = build_search_spaces()

    for name, (estimator, param_distributions) in spaces.items():
        log.info("Tuning %s: %d candidates x %d folds", name, N_ITER, N_SPLITS)

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=N_ITER,
            scoring=SCORING,
            cv=tscv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(X, y)

        log.info(
            "%s best CV %s: %.4f | best params: %s",
            name, SCORING, search.best_score_, search.best_params_,
        )

        out_path = MODELS_DIR / f"{name}_tuned.joblib"
        joblib.dump(
            {"model": search.best_estimator_, "feature_cols": feature_cols,
             "horizon": HORIZON_HOURS, "best_params": search.best_params_,
             "cv_score": search.best_score_},
            out_path,
        )
        log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    X, y, feature_cols = load_train_data()
    tune_and_save(X, y, feature_cols)