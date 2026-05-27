# Dataset Placement

Large raw and processed datasets are intentionally excluded from git.

## Expected Raw Paths

Place public benchmark files under these paths:

```text
data/raw/etth1/ETTh1.csv
data/raw/electricity/electricity.csv
data/raw/exchange_rate/exchange_rate.csv
data/raw/traffic/traffic.csv
data/raw/weather/weather.csv
```

Each public benchmark CSV should be wide format: one timestamp column named `date`,
plus one or more target columns. The loader converts it to `[unique_id, ds, y]`.

The provided finance dataset should live outside the repo:

```text
../market_data_downloader
```

On A00 this path is satisfied by:

```text
/home/wanyi/zy_test/market_data_downloader
```

because the project checkout is:

```text
/home/wanyi/zy_test/tabpfn-ts-benchmark
```

## Materialization

To inspect registry entries:

```bash
python scripts/inspect_datasets.py
```

To fetch a public benchmark file:

```bash
python scripts/fetch_public_benchmark.py etth1
python scripts/materialize_dataset.py etth1
```

To materialize a small finance sample:

```bash
bash scripts/materialize_stock_sample.sh 5
```

To materialize a complete registered dataset:

```bash
python scripts/materialize_dataset.py stock_provided
```

Outputs are written to `data/processed/<dataset>/`:

```text
series.parquet
summary.csv
spec.json
```
