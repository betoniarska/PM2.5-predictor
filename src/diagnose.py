"""
Isolates the effect of training-window length from the effect of which test
window happens to be evaluated. Reuses tune.py's exact search spaces and CV
strategy so the only thing that differs between the two runs is how much
training history is used.

Fixed: test.parquet (unchanged - the current 2yr run's held-out window)
Varied: training data = full train.parquet (~2yr) vs. last 365 days of it

Logs both under distinct run_tags so they're directly comparable in the
results notebook.
"""

import logging

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.config import RAW_DIR
from src.tune import build_search_spaces, N_SPLITS, N_ITER, RANDOM_STATE, TARGET_COL, NON_FEATURE_COLS
from src.evaluate import compute_metrics
from src.logger import log_results

log = logging.getLogger(__name__)

PROCESSED_DIR = RAW_DIR.parent / "processed"


def load_data():
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    return train, test


def make_recent_slice(train: pd.DataFrame, days: int = 365) -> pd.DataFrame:
    cutoff = train["datetime_utc"].max() - pd.Timedelta(days=days)
    return train[train["datetime_utc"] >= cutoff].reset_index(drop=True)


def run_search(train_slice: pd.DataFrame, test: pd.DataFrame, run_tag: str):
    feature_cols = [c for c in train_slice.columns if c not in NON_FEATURE_COLS]
    X_train, y_train = train_slice[feature_cols], train_slice[TARGET_COL]
    X_test, y_test = test[feature_cols], test[TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    spaces = build_search_spaces()
    results = {}

    for name, (estimator, param_distributions) in spaces.items():
        log.info("[%s] tuning %s on %d rows", run_tag, name, len(train_slice))
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=N_ITER,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        y_pred = search.best_estimator_.predict(X_test)
        results[name] = compute_metrics(y_test, y_pred)
        log.info("[%s] %s test metrics: %s", run_tag, name, results[name])

    horizon = int(TARGET_COL.split("_")[-1].replace("h", ""))
    log_results(
        horizon=horizon,
        n_train=len(train_slice),
        n_test=len(test),
        train_start=train_slice["datetime_utc"].min(),
        train_end=train_slice["datetime_utc"].max(),
        test_start=test["datetime_utc"].min(),
        test_end=test["datetime_utc"].max(),
        results=results,
        run_tag=run_tag,
    )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train, test = load_data()
    train_recent = make_recent_slice(train, days=365)

    log.info("Full train: %d rows (%s to %s)", len(train), train["datetime_utc"].min(), train["datetime_utc"].max())
    log.info("Recent (1yr) train: %d rows (%s to %s)", len(train_recent), train_recent["datetime_utc"].min(), train_recent["datetime_utc"].max())
    log.info("Fixed test set: %d rows (%s to %s)", len(test), test["datetime_utc"].min(), test["datetime_utc"].max())

    results_full = run_search(train, test, run_tag="diag_2yr_train_fixedtest")
    results_recent = run_search(train_recent, test, run_tag="diag_1yr_train_fixedtest")

    print("\n=== Full 2yr training window ===")
    for name, m in results_full.items():
        print(f"{name:<15} MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} R2={m['R2']:.3f}")

    print("\n=== Recent 1yr training window (same test set) ===")
    for name, m in results_recent.items():
        print(f"{name:<15} MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} R2={m['R2']:.3f}")