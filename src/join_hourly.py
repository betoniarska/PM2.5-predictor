"""
Outer-joins the hourly OpenAQ pollutant series (aggregate_openaq.py) with the
hourly FMI weather series (resample_fmi.py) into one wide dataframe, indexed
by datetime_utc.
"""

import logging

import pandas as pd

from src.config import RAW_DIR
from src.resample_fmi import load_fmi_raw, resample_fmi_hourly
from src.aggregate_openaq import load_openaq_raw, aggregate_pollutant_hourly, POLLUTANTS

log = logging.getLogger(__name__)

PROCESSED_DIR = RAW_DIR.parent / "processed"


def load_openmeteo_forecast() -> pd.DataFrame:
    path = RAW_DIR / "openmeteo_previous_day1.parquet"
    if not path.exists():
        log.warning("No Open-Meteo file at %s, skipping forecast features", path)
        return pd.DataFrame()
 
    df = pd.read_parquet(path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
 
    keep_cols = ["datetime_utc"] + [c for c in df.columns if c.endswith("_previous_day1")]
    return df[keep_cols]


def build_joined_hourly() -> pd.DataFrame:
    weather = resample_fmi_hourly(load_fmi_raw())

    merged = weather
    for name in POLLUTANTS:
        path = RAW_DIR / f"openaq_{name}.parquet"
        if not path.exists():
            log.warning("Skipping %s, no raw file at %s", name, path)
            continue
        pollutant_hourly = aggregate_pollutant_hourly(load_openaq_raw(name), name)
        merged = merged.merge(pollutant_hourly, on="datetime_utc", how="outer")

    forecast = load_openmeteo_forecast()
    if not forecast.empty:
        merged = merged.merge(forecast, on="datetime_utc", how="outer")
 
    merged = merged.sort_values("datetime_utc").reset_index(drop=True)
    # Keep only rows where either Air temperature is present or the datetime is after the first weather observation to avoid NaN's from the early OpenAQ data before the weather pull started.
    merged = merged[merged["Air temperature"].notna() | (merged["datetime_utc"] >= weather["datetime_utc"].min())]
    return merged


def report_gaps(df: pd.DataFrame) -> None:
    n = len(df)
    log.info("Joined dataset: %d rows, %s to %s", n, df["datetime_utc"].min(), df["datetime_utc"].max())
    for col in df.columns:
        if col == "datetime_utc":
            continue
        missing = df[col].isna().sum()
        if missing:
            log.info("  %-30s missing %5d / %d (%.1f%%)", col, missing, n, 100 * missing / n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = build_joined_hourly()
    report_gaps(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "helsinki_hourly_joined.parquet"
    print(df.head(10))
    df.to_parquet(out_path, index=False)
    log.info("Saved -> %s", out_path)