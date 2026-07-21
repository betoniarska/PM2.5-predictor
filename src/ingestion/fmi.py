import time
import logging
import datetime as dt
from pathlib import Path

import pandas as pd
from fmiopendata.wfs import download_stored_query

log = logging.getLogger(__name__)


class FMIClient:
    """
    FMI's open data WFS has no API key and no sensor-id lookup like OpenAQ.
    You query a fixed `place` (or fmisid) directly and get back a time series
    of every requested variable in one response, keyed by timestamp.
    """

    STORED_QUERY_ID = "fmi::observations::weather::multipointcoverage"

    def __init__(self):
        pass  # no client/session object needed; fmiopendata handles requests internally

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def fetch_chunk(self, fmisid: int, start_time: dt.datetime, end_time: dt.datetime) -> pd.DataFrame:
        """Fetch one chunk of hourly weather observations for a place/time range."""
        args = [
            f"fmisid={fmisid}",
            f"starttime={start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            f"endtime={end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        ]
        obs = download_stored_query(self.STORED_QUERY_ID, args=args)

        rows = []
        for timestamp, locations in obs.data.items():
            for loc_name, params in locations.items():
                row = {"datetime_utc": timestamp, "location": loc_name}
                for var_name, val in params.items():
                    row[var_name] = val.get("value")
                rows.append(row)

        return pd.DataFrame(rows)

    def fetch_range(
        self,
        fmisid: int,
        date_from: str,
        date_to: str,
        chunk_days: int = 7,
    ) -> pd.DataFrame:
        """
        FMI's multipointcoverage query is capped to a few days per request for
        hourly data, so page through the date range in chunks.
        """
        start = dt.datetime.fromisoformat(date_from)
        end = dt.datetime.fromisoformat(date_to)
        step = dt.timedelta(days=chunk_days)

        frames = []
        cur = start
        while cur < end:
            chunk_end = min(cur + step, end)
            try:
                df = self.fetch_chunk(fmisid, cur, chunk_end)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                log.warning("FMI fetch failed for %s - %s: %s", cur, chunk_end, e)
            cur = chunk_end
            time.sleep(0.2)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def fetch_and_save(
        self,
        fmisid: int,
        date_from: str,
        date_to: str,
        out_path: Path,
        chunk_days: int = 7,
    ) -> pd.DataFrame:
        df = self.fetch_range(fmisid, date_from, date_to, chunk_days=chunk_days)
        if not df.empty:
            df.to_parquet(out_path, index=False)
            log.info("FMI weather: %d rows -> %s", len(df), out_path)
        else:
            log.warning("FMI weather: no data returned")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from src.config import FMISID, DATE_FROM, DATE_TO, RAW_DIR

    with FMIClient() as client:
        client.fetch_and_save(
            fmisid=FMISID,
            date_from=DATE_FROM,
            date_to=DATE_TO,
            out_path=RAW_DIR / "fmi_weather.parquet",
        )