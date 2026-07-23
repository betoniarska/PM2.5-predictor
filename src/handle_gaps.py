"""
Fills small gaps in the joined hourly dataset via linear interpolation,
capped at a max run length so long outages aren't papered over with a
straight line across most of a day's diurnal pattern.
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)

# Columns eligible for interpolation - anything numeric except the
# datetime index and the sensor-count/coverage metadata columns, which
# should not be interpolated (a count is not a continuous quantity).

NON_INTERPOLATE_COLS = {"datetime_utc"}
NON_INTERPOLATE_SUFFIXES = ("_sensor_count", "_avg_percent_complete")

DEFAULT_MAX_GAP_HOURS = 3 


def interpolate_small_gaps(df: pd.DataFrame, max_gap_hours: int = DEFAULT_MAX_GAP_HOURS) -> pd.DataFrame:
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    cols = [
        c for c in df.columns
        if c not in NON_INTERPOLATE_COLS and not c.endswith(NON_INTERPOLATE_SUFFIXES)
    ]

    for col in cols:
        df[col] = df[col].interpolate(method="linear", limit=max_gap_hours, limit_area="inside")

    return df


def report_remaining_gaps(df: pd.DataFrame) -> None:
    n = len(df)
    log.info("After interpolation (%d rows):", n)
    for col in df.columns:
        if col in NON_INTERPOLATE_COLS or col.endswith(NON_INTERPOLATE_SUFFIXES):
            continue
        missing = df[col].isna().sum()
        if missing:
            log.info("  %-30s still missing %5d / %d (%.1f%%)", col, missing, n, 100 * missing / n)


if __name__ == "__main__":
    from src.config import RAW_DIR
    from src.join_hourly import build_joined_hourly

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = build_joined_hourly()
    df = interpolate_small_gaps(df)
    report_remaining_gaps(df)

    processed_dir = RAW_DIR.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "helsinki_hourly_filled.parquet"
    df.to_parquet(out_path, index=False)
    print(df.head(10))
    log.info("Saved -> %s", out_path)