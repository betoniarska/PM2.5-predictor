import pandas as pd
from src.features import (
    PROCESSED_DIR, HORIZON_HOURS, LAG_HOURS, ROLLING_WINDOWS,
    drop_metadata_cols, add_target, add_time_features,
)

def add_lags_for(df, pollutant_cols):
    for col in pollutant_cols:
        for lag in LAG_HOURS:
            df[f"{col}_lag{lag}h"] = df[col].shift(lag)
        for window in ROLLING_WINDOWS:
            df[f"{col}_roll{window}h_mean"] = df[col].shift(1).rolling(window).mean()
    return df

def build(pollutant_cols):
    df = pd.read_parquet(PROCESSED_DIR / "helsinki_hourly_filled.parquet")
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    df = drop_metadata_cols(df)
    df = add_target(df, HORIZON_HOURS)
    df = add_lags_for(df, pollutant_cols)
    df = add_time_features(df)
    return df

target_col = f"pm25_target_{HORIZON_HOURS}h"

# WITH o3
df_with = build(["pm25", "pm10", "no2", "o3"])
df_with = df_with.dropna(subset=[target_col])
feat_cols_with = [c for c in df_with.columns if c not in ("datetime_utc", target_col)]
df_with = df_with.dropna(subset=feat_cols_with)

# WITHOUT o3
df_without = build(["pm25", "pm10", "no2"])
df_without = df_without.dropna(subset=[target_col])
feat_cols_without = [c for c in df_without.columns if c not in ("datetime_utc", target_col)]
df_without = df_without.dropna(subset=feat_cols_without)

print("With o3:   ", len(df_with), "rows")
print("Without o3:", len(df_without), "rows")