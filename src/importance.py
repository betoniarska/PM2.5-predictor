"""
Loads the Random Forest and XGBoost models and reports feature importances
"""

import logging

import joblib
import pandas as pd

from src.config import RAW_DIR

log = logging.getLogger(__name__)

MODELS_DIR = RAW_DIR.parent.parent / "models"

WEATHER_COLS_HINTS = (
    "Air temperature", "Wind", "Gust", "Relative humidity",
    "Dew-point", "Precipitation", "Snow", "Pressure", "Horizontal visibility",
    "Cloud amount", "Present weather",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "pressure_msl", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
)


def load_importances(name: str) -> pd.DataFrame:
    bundle = joblib.load(MODELS_DIR / f"{name}_tuned.joblib")
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["is_weather"] = df["feature"].str.startswith(WEATHER_COLS_HINTS)
    return df


def summarize(name: str, df: pd.DataFrame, top_n: int = 15):
    print(f"\n=== {name}: top {top_n} features ===")
    print(df.head(top_n).to_string(index=False))

    weather_share = df[df["is_weather"]]["importance"].sum()
    pollutant_share = df[~df["is_weather"]]["importance"].sum()

    print(f"\n{name}: weather features account for {weather_share:.1%} of total importance")
    print(f"{name}: pollutant/time features account for {pollutant_share:.1%} of total importance")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from src.logger import log_importances

    for name in ["random_forest", "xgboost"]:
        bundle_path = MODELS_DIR / f"{name}_tuned.joblib"
        bundle = joblib.load(bundle_path)
        horizon = bundle["horizon"]

        df = load_importances(name)
        summarize(name, df)
        log_importances(name, df, horizon=horizon, run_tag="first_tune")