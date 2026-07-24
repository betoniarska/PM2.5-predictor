import logging

from src.ingestion.openaq import OpenAQClient
from src.ingestion.fmi import FMIClient
from src.config import CITY_LAT, CITY_LON, RADIUS_M, PARAMETERS, RAW_DIR, FMI_FMISID

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATE_FROM = "2025-07-01"
DATE_TO = "2026-07-15"

if __name__ == "__main__":
    with OpenAQClient() as c:
        for name, pid in PARAMETERS.items():
            try:
                c.fetch_and_save(
                    CITY_LAT, CITY_LON, RADIUS_M, pid, DATE_FROM, DATE_TO,
                    RAW_DIR / f"openaq_{name}.parquet", name,
                )
            except Exception as e:
                logging.getLogger(__name__).error("Failed to fetch %s entirely: %s", name, e)

    with FMIClient() as c:
        c.fetch_and_save(FMI_FMISID, DATE_FROM, DATE_TO, RAW_DIR / "fmi_weather.parquet")