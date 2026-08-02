# Helsinki PM2.5 Predictor

A hobby ML project forecasting PM2.5 concentration in Helsinki using historical
air quality data (OpenAQ), weather observations (FMI) and historical forecast data (Open-Meteo).

**Status**: 2.8.2026: early/iterating: pipeline works end-to-end, diagnosed plateauing results, forecast features implementation begun

## Pipeline

```
data/raw/            <- untouched API responses (ELT: extract, load, transform later)
data/processed/      <- FMI resampled, joined w/ OpenAQ, gap-filled, feature-engineered (.parquet)
models/              <- trained model artifacts (.joblib)

src/
  ingestion/
    openaq.py             OpenAQ API client (PM2.5, PM10, NO2, O3)
    fmi.py                FMI open data client (weather observations)
    openmeteo.py          OpenMeteo data client (historical forecast data)
  load_raw.py             orchestrates the raw pull for both sources
  resample_fmi.py         10-min weather observations -> hourly
  aggregate_openaq.py     multiple sensors per pollutant -> one hourly value for Helsinki area
  join_hourly.py          outer-joins weather + all pollutants on datetime_utc
  handle_gaps.py          capped linear interpolation for small gaps (larger caps quite consistent across sensors --> drop NaN rows)
  features.py             lag/rolling features, target construction, train/test split
  train.py                fits Ridge / Random Forest / XGBoost
  evaluate.py             compares models against a naive persistence baseline
  importance.py           report top-n feature importances
  logger.py               log the results from evaluate.py and importance.py into results/ from which they can be visualized in results.ipynb
  diagnose.py             Initially to diagnose the plateauing effect of adding more data, later the current best result (1 yr XGBoost R2: 0.306)
```

Each script under `src/` does one job and can be run standalone
(`python -m src.<module>`), which made debugging each pipeline stage in
isolation much easier than building one large script.

## Key design decisions

- **ELT** Raw API responses are saved untouched; all cleaning/
  filtering/feature choices happen downstream and are cheap to redo without
  re-hitting the APIs.
- **Time-based train/test split** Lag features make rows
  temporally dependent, and a real forecast never sees the future.
- **Rolling/lag features are computed with `.shift(1)` first**, so a feature
  never includes the value it's implicitly trying to predict.
- **No live weather forecast source yet** Weather features use the most
  recent observation, not a forecast for the target hour.
- **Sensor averaging**: pollutants with multiple nearby sensors (PM2.5, PM10,
  NO2) are averaged per hour; a `_sensor_count` column is kept so an hour
  backed by fewer sensors is distinguishable from a fully-covered one. O3 has
  only one sensor in range, so its "average" is really a single station. This is
  for the Helsinki area currently.

## Engineering notes

- **OpenAQ pagination silently failed on a full year of hourly data.** Root
  cause turned out to be server-side: deep pagination over a wide date range
  returned an explicit HTTP 408 ("try a smaller time frame"), which a longer
  client timeout couldn't fix. Solved by chunking each sensor's request into
  ~45-day windows instead of one year-long paginated request. Same fix
  pattern already used for FMI's weather pull, just for a different underlying
  reason (FMI limits response size; OpenAQ apparently degrades on deep offset
  scans over wide ranges).
- **FMI reports at 10-minute resolution**, not hourly — required resampling
  before it could be joined with OpenAQ's hourly pollutant data. Precipitation
  is summed across the hour; everything else is averaged.

## Known limitations / next steps

- O3 has only one sensor and noticeably worse uptime than the other
  pollutants; still deciding whether to keep it as a feature or drop it.
  (might change if adding a larger measurement area e.g. whole Helsinki metropolitan)
- Forecast data integration resolved via open-meteo's (https://www.meteomatics.com/)
  historical forecast data for the Helsinki area. Currently on a 24h horizon.
- More yearly data shown in diagnosis.py (2 years vs 1 year) to not have meaningful prediction power on pm2.5,
  this is mostl likely due to the fact that historical pm2.5 features (rolling 24h pm2.5,
  pm2.5, rolling 6h pm2.5 etc (shown in results.ipynb on the importance chart) are completely
  dominating the feature importance metrics. However, Open-Meteos wind_direction_10m_previous_day_1
  has made it into the top 15 features in importance, beating out all of FMI's ground thruth weather
  features and a considerable chunk of OpenAQ's historical pollutants, showing that the forecast features
  indeed do represent promising value for predicting pm2.5.
- Non-stationarity likely cause for the 2 years of data achieving worse results

## Current best prediction accuracy (diagnosis.py)

  XGBoost via horizon=24h and 1 year of data: R2=0.306
  
  This could be classified as moderate predictive power for the pm2.5 levels in the Helsinki area,
  with the XGBoost explaining around 30% of the variation in the pm2.5 levels.

  The full latest run of diagnose:
  
  === Full 2yr training window ===
  ridge           MAE=2.349 RMSE=3.258 R2=0.170
  random_forest   MAE=2.248 RMSE=3.129 R2=0.234
  xgboost         MAE=2.171 RMSE=3.096 R2=0.250
  
  === Recent 1yr training window (same test set) ===
  ridge           MAE=2.461 RMSE=3.385 R2=0.103
  random_forest   MAE=2.389 RMSE=3.220 R2=0.189
  xgboost         MAE=2.125 RMSE=2.978 R2=0.306
