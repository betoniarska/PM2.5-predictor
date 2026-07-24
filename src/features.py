"""
Builds the model-ready feature matrix from data/processed/helsinki_hourly_filled.parquet.

Steps:
  1. Drop non-feature metadata columns (sensor_count, avg_percent_complete)
  2. Construct the target: pm25 shifted forward by horizon hours
  3. Build lag + rolling features for all pollutants (never the current/future value) : (pollutants heavily correlated)
  4. Add cyclical time-of-day / day-of-week features
  5. Drop rows that can't have complete features (start-of-series lag NaNs,
     end-of-series target NaNs)
  6. Time-based train/test split (no random split - lag features make rows
     temporally dependent, and a real forecast never sees the future)

"""

import logging

import numpy as np
import pandas as pd

from src.config import RAW_DIR

log = logging.getLogger(__name__)

PROCESSED_DIR = RAW_DIR.parent / "processed"

HORIZON_HOURS = 24

POLLUTANT_COLS = ["pm25", "pm10", "no2", "o3"]
LAG_HOURS = [1, 2, 3, 6, 24]
ROLLING_WINDOWS = [3, 6, 24]

TEST_FRACTION = 0.15  # held out from the end of the series, not randomly sampled


def drop_metadata_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if c.endswith(("_sensor_count", "_avg_percent_complete"))]
    return df.drop(columns=drop_cols)


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df[f"pm25_target_{horizon}h"] = df["pm25"].shift(-horizon)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in POLLUTANT_COLS:
        for lag in LAG_HOURS:
            df[f"{col}_lag{lag}h"] = df[col].shift(lag)
        for window in ROLLING_WINDOWS:

            # shift(1) first so the rolling window never includes the current hour -
            # at prediction time "now" is the most recent lag, not a peek at itself

            df[f"{col}_roll{window}h_mean"] = df[col].shift(1).rolling(window).mean()
    return df

# Add cyclical time features i.e. to address the fact that 23:00 and 00:00 are close in time, but far apart numerically.
# https://feature-engine.trainindata.com/en/1.8.x/user_guide/creation/CyclicalFeatures.html

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:

    hour = df["datetime_utc"].dt.hour
    dow = df["datetime_utc"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def build_features(horizon: int = HORIZON_HOURS) -> pd.DataFrame:
    path = PROCESSED_DIR / "helsinki_hourly_filled.parquet"
    df = pd.read_parquet(path)
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    df = drop_metadata_cols(df)
    df = add_target(df, horizon)
    df = add_lag_and_rolling_features(df)
    df = add_time_features(df)

    print(df.size, df.shape, df.columns)

    return df


def drop_incomplete_rows(df: pd.DataFrame, horizon: int) -> pd.DataFrame:

    target_col = f"pm25_target_{horizon}h"
    n_before = len(df)
    df = df.dropna(subset=[target_col])
    n_after_target = len(df)

    feature_cols = [c for c in df.columns if c not in ("datetime_utc", target_col)]
    df = df.dropna(subset=feature_cols)
    n_after_features = len(df)

    log.info(
        "Dropped %d rows with missing target, %d further rows with missing features (%d -> %d)",
        n_before - n_after_target, n_after_target - n_after_features, n_before, n_after_features,
    )
    return df


def time_based_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    n = len(df)
    split_idx = int(n * (1 - test_fraction))
    split_time = df.iloc[split_idx]["datetime_utc"]

    train = df[df["datetime_utc"] < split_time].reset_index(drop=True)
    test = df[df["datetime_utc"] >= split_time].reset_index(drop=True)

    log.info(
        "Train: %d rows (%s to %s) | Test: %d rows (%s to %s)",
        len(train), train["datetime_utc"].min(), train["datetime_utc"].max(),
        len(test), test["datetime_utc"].min(), test["datetime_utc"].max(),
    )
    return train, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = build_features(HORIZON_HOURS)
    df = drop_incomplete_rows(df, HORIZON_HOURS)
    train, test = time_based_split(df)

    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    log.info("Saved -> %s, %s", PROCESSED_DIR / "train.parquet", PROCESSED_DIR / "test.parquet")