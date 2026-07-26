# Helsinki PM2.5 Predictor

A hobby ML project forecasting PM2.5 concentration in Helsinki using historical
air quality data (OpenAQ) and weather observations (FMI).

**Status**: 26.7.2026: early/iterating: pipeline works end-to-end, first tuned models, no forecast weather data yet

## Pipeline

```
data/raw/            <- untouched API responses (ELT: extract, load, transform later)
data/processed/      <- FMI resampled, joined w/ OpenAQ, gap-filled, feature-engineered (.parquet)
models/              <- trained model artifacts (.joblib)

src/
  ingestion/
    openaq.py             OpenAQ API client (PM2.5, PM10, NO2, O3)
    fmi.py                FMI open data client (weather observations)
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
```

Each script under `src/` does one job and can be run standalone
(`python -m src.<module>`), which made debugging each pipeline stage in
isolation much easier than building one large script.

## Key design decisions

- **ELT, not ETL.** Raw API responses are saved untouched; all cleaning/
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
- No real weather forecast integration yet: needed for the 24h horizon to
  reach its full potential.
- Major pitfall: to use forecast data for the t+horizon timestamp the models needs to
  have historical data of forecasts on the same horizon, FMI most likely does not serve
  historical forecasts. Other way is to start collecting forecasts from once the predictor
  goes live, but that will take months to gather any meaningful forecast data.
  Update: found Open-meteo at https://www.meteomatics.com/ which servers daily historical
  forecasts for specified regions. Seems like a convenient solution as of now:
  for a 24h-ahead target, join _previous_day1 on my existing datetime_utc; for 48h-ahead,
  join _previous_day2, etc etc just swapping the offset depending on the desired horizon
