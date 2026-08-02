"""
Open-Meteo Previous Runs API client.

Unlike FMI's weather observations (ground truth - what actually happened),
this pulls what was FORECAST at a fixed lead time before the valid hour -
e.g. temperature_2m_previous_day1 is what the model predicted 24h in advance.
This is the genuinely new signal for the 24h+ horizon: anticipated change,
not just current conditions.

No API key required for non-commercial use. Plain HTTP GET, JSON response,
no SDK. Coverage starts January 2024 for most models.

IMPORTANT: only a specific subset of variables support the _previous_dayN
offset (temperature, humidity, dewpoint, precipitation, pressure, cloud
cover, wind speed/direction, weather code) - "visibility" and "snow depth"
are NOT available this way, so this supplements but doesn't fully replace
the FMI weather features.

This module is built from Open-Meteo's documentation but has not been
tested against a live request from this environment - verify against a
small date range before trusting it for a bulk pull, same as every other
ingestion module in this project.

Run: python -m src.ingestion.openmeteo
"""

from __future__ import annotations

import time
import logging
from pathlib import Path

import requests
import pandas as pd

log = logging.getLogger(__name__)

BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Variables confirmed by Open-Meteo's docs to support the _previous_dayN
# lead-time offset. Adjust if you check the docs and find more.

FORECASTABLE_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
]


class OpenMeteoClient:

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _fetch_chunk(
        self, lat: float, lon: float, date_from: str, date_to: str,
        lead_days: list[int], variables: list[str],
        max_retries: int = 4, backoff_base: float = 5.0,
    ) -> pd.DataFrame:
        hourly_vars = list(variables)  # day-0 (ground truth from this API too, optional)
        for var in variables:
            for lead in lead_days:
                hourly_vars.append(f"{var}_previous_day{lead}")

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(hourly_vars),
            "start_date": date_from,
            "end_date": date_to,
            "timezone": "UTC",
        }

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == max_retries:
                    log.error(
                        "%s->%s failed after %d attempts, skipping chunk: %s",
                        date_from, date_to, max_retries, e,
                    )
                    return pd.DataFrame()
                wait = backoff_base * (2 ** (attempt - 1))
                log.warning(
                    "%s->%s attempt %d/%d failed (%s), retrying in %.0fs",
                    date_from, date_to, attempt, max_retries, e, wait,
                )
                time.sleep(wait)

        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            return pd.DataFrame()

        df = pd.DataFrame(hourly)
        df = df.rename(columns={"time": "datetime_utc"})
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
        return df

    def fetch_range(
        self,
        lat: float, lon: float, date_from: str, date_to: str,
        lead_days: list[int] = [1],
        variables: list[str] = FORECASTABLE_VARIABLES,
        chunk_days: int = 180,
    ) -> pd.DataFrame:
        start = pd.Timestamp(date_from)
        end = pd.Timestamp(date_to)
        step = pd.Timedelta(days=chunk_days)

        frames = []
        cur = start
        while cur < end:
            chunk_end = min(cur + step, end)
            df = self._fetch_chunk(
                lat, lon, cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"),
                lead_days, variables,
            )
            if not df.empty:
                frames.append(df)
            cur = chunk_end
            time.sleep(0.5)  # courtesy, free-tier rate limit

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_and_save(
        self, lat: float, lon: float, date_from: str, date_to: str,
        out_path: Path, lead_days: list[int] = [1],
        variables: list[str] = FORECASTABLE_VARIABLES,
    ) -> pd.DataFrame:
        df = self.fetch_range(lat, lon, date_from, date_to, lead_days, variables)

        if not df.empty:
            df.to_parquet(out_path, index=False)
            log.info("open-meteo previous_day%s: %d rows -> %s", lead_days, len(df), out_path)
        else:
            log.warning("open-meteo: no data returned")
            
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from src.config import CITY_LAT, CITY_LON, RAW_DIR

    with OpenMeteoClient() as client:
        client.fetch_and_save(
            lat=CITY_LAT,
            lon=CITY_LON,
            date_from="2024-07-01",
            date_to="2026-07-15",
            out_path=RAW_DIR / "openmeteo_previous_day1.parquet",
            lead_days=[1],
        )