"""
Resamples the raw FMI weather pull (10-min observations) to hourly.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

log = logging.getLogger(__name__)

PROCESSED_DIR = RAW_DIR.parent / "processed"

# How to resample each FMI variable from 10-min to hourly.
# Anything not listed here defaults to "mean".

FMI_AGG = {
    "Precipitation amount": "sum",
    "Precipitation intensity": "mean",
    "Snow depth": "mean",
    "Cloud amount": "mean",
    "Present weather (auto)": "first",  # WMO weather code, not a continuous quantity
}


def load_fmi_raw() -> pd.DataFrame:
    path = RAW_DIR / "fmi_weather.parquet"
    df = pd.read_parquet(path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df


def resample_fmi_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample the raw FMI weather pull (10-min observations) to hourly."""

    df = df.drop(columns=["location"], errors="ignore")

    value_cols = [c for c in df.columns if c != "datetime_utc"]

    agg_map = {c: FMI_AGG.get(c, "mean") for c in value_cols}

    hourly = (
        df.set_index("datetime_utc")
        .resample("h")
        .agg(agg_map)
        .reset_index()
    )
    return hourly


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load the raw FMI weather data from data/raw
    raw = load_fmi_raw()
    log.info("Raw: %d rows, %s to %s", len(raw), raw["datetime_utc"].min(), raw["datetime_utc"].max())

    # Resample to hourly and log the result.
    hourly = resample_fmi_hourly(raw)
    log.info("Hourly: %d rows, %s to %s", len(hourly), hourly["datetime_utc"].min(), hourly["datetime_utc"].max())

    print(hourly.head())
    print("Missing values per column:")
    print(hourly.isna().sum())