"""
Averages raw OpenAQ sensor data into one Helsinki-level hourly series per pollutant.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

log = logging.getLogger(__name__)

# Columns kept from the full OpenAQ raw schema 

OPENAQ_KEEP_COLS = [
    "value",
    "parameter.name",
    "parameter.units",
    "period.datetime_from.utc",
    "coverage.observed_count",
    "coverage.expected_count",
    "coverage.percent_complete",
    "sensor_id",
]

POLLUTANTS = ["pm25", "pm10", "no2", "o3"]


def load_openaq_raw(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"openaq_{name}.parquet"
    df = pd.read_parquet(path)

    missing = [c for c in OPENAQ_KEEP_COLS if c not in df.columns]

    if missing:
        raise ValueError(f"{path} missing expected columns: {missing}")

    df = df[OPENAQ_KEEP_COLS].copy()
    df = df.rename(columns={"period.datetime_from.utc": "datetime_utc"})
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

    unique_params = df["parameter.name"].unique()
    if len(unique_params) != 1:
        log.warning("%s: expected one parameter, found %s", path, unique_params)

    return df


def aggregate_pollutant_hourly(df: pd.DataFrame, name: str) -> pd.DataFrame:

    """Average across sensors per hour. Keeps sensor_count and avg_percent_complete
    so a downstream user can see how many sensors and how reliable each averaged
    hour was, rather than treating every hourly value as equally trustworthy."""
    
    agg = (
        df.groupby("datetime_utc")
        .agg(
            value=("value", "mean"),
            sensor_count=("sensor_id", "nunique"),
            avg_percent_complete=("coverage.percent_complete", "mean"),
        )
        .reset_index()
    )
    agg = agg.rename(columns={
        "value": name,
        "sensor_count": f"{name}_sensor_count",
        "avg_percent_complete": f"{name}_avg_percent_complete",
    })
    return agg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for name in POLLUTANTS:
        path = RAW_DIR / f"openaq_{name}.parquet"
        if not path.exists():
            log.warning("Skipping %s, no raw file at %s", name, path)
            continue

        raw = load_openaq_raw(name)
        hourly = aggregate_pollutant_hourly(raw, name)

        n_sensors = raw["sensor_id"].nunique()
        log.info(
            "%s: %d sensors, raw %d rows -> hourly %d rows, %s to %s",
            name, n_sensors, len(raw), len(hourly),
            hourly["datetime_utc"].min(), hourly["datetime_utc"].max(),
        )
        print(hourly.head())
        print(hourly[f"{name}_sensor_count"].value_counts().sort_index())
        print()