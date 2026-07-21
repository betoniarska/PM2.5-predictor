from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
STORED_QUERY_ID = os.getenv("STORED_QUERY_ID")
RAW_DIR = Path("data/raw")
CITY_LAT = 60.1699
CITY_LON = 24.9384
RADIUS_M = 10000
PARAMETERS = {
    "pm25": 2,
    "pm10": 1,
    "no2": 5,
    "o3": 3,
}

FMI_FMISID = 100971
