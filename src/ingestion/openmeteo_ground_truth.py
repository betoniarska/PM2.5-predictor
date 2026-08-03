"""
Open-Meteo Historical Forecast API client - separate from openmeteo.py
because this pulls a genuinely different kind of data: GROUND TRUTH values
from the ECMWF IFS HRES 9km model tier, not forecast-at-lead-time data.

Some variables (e.g. boundary_layer_height) aren't available with the
_previous_dayN offset at all via the Previous Runs API's 0.25° model. They
only exist on this HRES 9km tier, and only as "what actually happened,"
not "what was forecast in advance." That means these features can't be
used the way the _previous_day1 columns are (as genuine forward-looking
signal) --> Same category as the FMI weather columns.

Coverage has known gaps: boundary_layer_height
has a real multi-month outage in Open-Meteo's own backend
(present through mid-2025, missing by Oct 2025,
back by Jan 2026): handle_blh_gap() in features.py imputes this rather
than letting it silently delete otherwise-complete rows via dropna.

"""
from __future__ import annotations

import time
import logging
from pathlib import Path

import requests
import pandas as pd

log = logging.getLogger(__name__)

BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
DEFAULT_MODEL = "ecmwf_ifs"  # HRES 9km


class OpenMeteoGroundTruthClient:

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
        variables: list[str], model: str,
        max_retries: int = 4, backoff_base: float = 5.0,
    ) -> pd.DataFrame:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "start_date": date_from,
            "end_date": date_to,
            "models": model,
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
        self, lat: float, lon: float, date_from: str, date_to: str,
        variables: list[str] = ["boundary_layer_height"],
        model: str = DEFAULT_MODEL,
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
                variables, model,
            )
            if not df.empty:
                frames.append(df)
            cur = chunk_end
            time.sleep(0.5)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_and_save(
        self, lat: float, lon: float, date_from: str, date_to: str,
        out_path: Path, variables: list[str] = ["boundary_layer_height"],
        model: str = DEFAULT_MODEL,
    ) -> pd.DataFrame:
        df = self.fetch_range(lat, lon, date_from, date_to, variables, model)
        if not df.empty:
            df.to_parquet(out_path, index=False)
            log.info("open-meteo ground truth %s: %d rows -> %s", variables, len(df), out_path)
        else:
            log.warning("open-meteo ground truth: no data returned")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from src.config import CITY_LAT, CITY_LON, RAW_DIR

    with OpenMeteoGroundTruthClient() as client:
        client.fetch_and_save(
            lat=CITY_LAT,
            lon=CITY_LON,
            date_from="2024-07-01",
            date_to="2026-07-15",
            out_path=RAW_DIR / "openmeteo_blh.parquet",
            variables=["boundary_layer_height"],
        )