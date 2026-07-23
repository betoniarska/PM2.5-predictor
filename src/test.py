import pandas as pd
df = pd.read_parquet('data/processed/helsinki_hourly_joined.parquet')
gaps = df[df['pm25'].isna()]['datetime_utc'].reset_index(drop=True)

# group into contiguous runs

is_new_block = gaps.diff() != pd.Timedelta(hours=1)
block_id = is_new_block.cumsum()
blocks = gaps.groupby(block_id).agg(['min', 'max', 'count'])
print(blocks.sort_values(('count'), ascending=False))

raw = pd.read_parquet('data/raw/openaq_pm25.parquet')
raw['datetime_utc'] = pd.to_datetime(raw['period.datetime_from.utc'], utc=True)

# check which sensors have data during one of the known gap windows from `blocks` above