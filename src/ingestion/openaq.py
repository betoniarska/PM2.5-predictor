import time
import logging
from pathlib import Path
import dataclasses

import pandas as pd
from openaq import OpenAQ

from src.config import OPENAQ_API_KEY

import httpx
from openaq._sync.transport import Transport


log = logging.getLogger(__name__)

class LongTimeoutTransport(Transport):
    def __init__(self, timeout: float = 30.0):
        self.client = httpx.Client(timeout=timeout)


class OpenAQClient:

    def __init__(self, api_key: str = OPENAQ_API_KEY):
        if not api_key:
            raise ValueError("OpenAQ API key not set (check src/config.py or env var)")
        self.api_key = api_key

        # httpx's default timeout is too short for deep-pagination queries
        # use a custom transport with a longer timeout for the OpenAQ client
   
        self._client = OpenAQ(api_key=self.api_key, _transport=LongTimeoutTransport(timeout=30.0))

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # get all sensor ids for a given parameter near a coordinate
    def get_sensor_ids(self, lat: float, lon: float, radius_m: int, parameter_id: int) -> list[int]:

        
        locs = self._client.locations.list(
            coordinates=(lat, lon),
            radius=radius_m,
            parameters_id=parameter_id,
        )
        
        sensor_ids = [
            s.id for loc in locs.results for s in loc.sensors
            if s.parameter.id == parameter_id
        ]
        log.info("Found %d sensors for parameter_id=%d", len(sensor_ids), parameter_id)
        return sensor_ids

    # fetch hourly data for a specific sensor

    def fetch_sensor_hourly_chunk(
        self, sensor_id: int, date_from: str, date_to: str,
        max_retries: int = 4, backoff_base: float = 5.0,
    ) -> pd.DataFrame:
        rows = []
        page = 1
        while True:
            res = None
            for attempt in range(1, max_retries + 1):
                try:
                    res = self._client.measurements.list(
                        sensors_id=sensor_id,
                        data="hours",
                        datetime_from=date_from,
                        datetime_to=date_to,
                        limit=1000,
                        page=page,
                    )
                    break
                except Exception as e:
                    if attempt == max_retries:
                        log.error(
                            "sensor %s page %s failed after %d attempts, stopping pagination early "
                            "(keeping %d rows already fetched): %s",
                            sensor_id, page, max_retries, len(rows), e,
                        )
                        res = None
                    else:
                        wait = backoff_base * (2 ** (attempt - 1))
                        log.warning(
                            "sensor %s page %s attempt %d/%d failed (%s), retrying in %.0fs",
                            sensor_id, page, attempt, max_retries, e, wait,
                        )
                        time.sleep(wait)

            if res is None:
                break  # retries exhausted this page - keep what we have, stop paginating this sensor

            if not res.results:
                break
            rows.extend(dataclasses.asdict(r) for r in res.results)
            if len(res.results) < 1000:
                break
            page += 1
            time.sleep(0.2)  # free-tier rate limit courtesy

        if not rows:
            return pd.DataFrame()
        df = pd.json_normalize(rows)
        df["sensor_id"] = sensor_id
        return df
    
    def fetch_sensor_hourly(
        self, sensor_id: int, date_from: str, date_to: str,
        chunk_days: int = 45,
    ) -> pd.DataFrame:
        """
        Fetches one sensor's full date range by splitting it into smaller
        windows first. OpenAQ's API returns an explicit 408 on deep pagination
        over a wide date range ("try a smaller time frame") - this sidesteps
        that by keeping each individual query's date span small, regardless
        of how many total hours are being pulled overall.
        """
        start = pd.Timestamp(date_from)
        end = pd.Timestamp(date_to)
        step = pd.Timedelta(days=chunk_days)
 
        frames = []
        cur = start
        while cur < end:
            chunk_end = min(cur + step, end)
            df = self.fetch_sensor_hourly_chunk(
                sensor_id, cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
            )
            if not df.empty:
                frames.append(df)
            cur = chunk_end
 
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
 

    # fetch hourly data for every sensor matching a parameter near a coordinate
    def fetch_parameter(
        self,
        lat: float,
        lon: float,
        radius_m: int,
        parameter_id: int,
        date_from: str,
        date_to: str,
    ) -> pd.DataFrame:
        """Fetch hourly data for every sensor matching a parameter near a coordinate."""
        sensor_ids = self.get_sensor_ids(lat, lon, radius_m, parameter_id)
        frames = []
        for sid in sensor_ids:
            try:
                frames.append(self.fetch_sensor_hourly(sid, date_from, date_to))
            except Exception as e:
                log.error("Giving up on sensor %s entirely after retries: %s", sid, e)
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # fetch and save data for a parameter near a coordinate
    def fetch_and_save(
        self,
        lat: float,
        lon: float,
        radius_m: int,
        parameter_id: int,
        date_from: str,
        date_to: str,
        out_path: Path,
        name: str,
    ) -> pd.DataFrame:
        df = self.fetch_parameter(lat, lon, radius_m, parameter_id, date_from, date_to)
        if not df.empty:
            df.to_parquet(out_path, index=False)
            log.info("%s: %d rows -> %s", name, len(df), out_path)
        else:
            log.warning("%s: no data returned", name)

        
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from src.config import CITY_LAT, CITY_LON, RADIUS_M, DATE_FROM, DATE_TO, PARAMETERS, RAW_DIR

    with OpenAQClient() as client:
        for name, pid in PARAMETERS.items():
            client.fetch_and_save(
                lat=CITY_LAT,
                lon=CITY_LON,
                radius_m=RADIUS_M,
                parameter_id=pid,
                date_from=DATE_FROM,
                date_to=DATE_TO,
                out_path=RAW_DIR / f"openaq_{name}.parquet",
                name=name,
            )