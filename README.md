# TabPFN Time-Series Benchmark

## Project goal

This project benchmarks TabPFN-TS against statistical, deep learning, and foundation-model baselines across multiple time-series domains. Finance is one downstream application, not the whole study: the benchmark should cover public non-financial datasets first, then use a governed market-data pipeline to evaluate whether forecasts can become tradable stock signals.

## Claims being tested

1. TabPFN-TS achieves competitive zero-shot performance across diverse time-series benchmarks.
2. TabPFN-TS degrades less gracefully than trained baselines in low-data regimes.
3. TabPFN-TS generalizes consistently across energy, traffic, weather, and finance domains.
4. With a proper data governance pipeline, TabPFN-TS can produce tradable stock signals measured by IC, Rank IC, backtest Sharpe, and drawdown.

## Quick start

Experiments should be run on the least busy available remote GPU host (`A00` or
`4080s`), not on the local machine. Before launching any job, check GPU
utilization and avoid starting runs when another user is occupying the card.

```bash
ssh A00
nvidia-smi

cd /path/to/tabpfn-quant
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
bash scripts/run_smoke_test.sh
pytest tests/
```

From this machine, `bash scripts/check_experiment_hosts.sh` prints the current
remote GPU and project sync status without starting an experiment.

## Experiment reports

Generate a reusable HTML report from any run directory that contains
`forecast_metrics.csv` and optionally `forecast_predictions.csv`:

```bash
python scripts/generate_report.py \
  --run-dir results/tabpfn_ts_a100_c96_h12 \
  --baseline-dir results/pilot_baseline \
  --config configs/config.yaml \
  --out reports/tabpfn_ts_a100_c96_h12
```

The report writes `index.html`, `metrics.json`, `summary.csv`, and matplotlib
SVG figures under `assets/`. Missing metrics or logs are shown as N/A rather
than being fabricated.

For a presentation-grade dashboard demo with premium dark and clean themes:

```bash
python scripts/generate_fancy_demo_report.py
```

This writes:

- `reports/demo/fancy_dark/index.html`
- `reports/demo/fancy_clean/index.html`
- `reports/demo/fancy_v1/index.html`

The fancy dashboard uses the shared design tokens in `src/reporting/style_config.py`.
It reuses real metrics and predictions, while clearly marking the training curve
as a demo placeholder when no real training logs are available.

## Project layout

```text
tabpfn-quant/
├── configs/          # Hydra dataset, model, and experiment configs
├── src/              # Governance, model, data registry/loaders, evaluation, backtest, and utilities
├── experiments/      # Runnable experiment entry points
├── tests/            # Metric, interface, and smoke tests
├── data/             # Raw and processed data placeholders
├── results/          # Experiment outputs
├── notebooks/        # Exploratory analysis notebooks
└── scripts/          # Reproducible shell entry points
```

The provided market data lives one level above this project in `../market_data_downloader` and is referenced by `configs/dataset/stock_provided.yaml`. Public benchmark datasets such as ETT, electricity, exchange-rate, traffic, weather, and other non-financial panels are part of later phases.

To inspect the cross-domain dataset registry without launching experiments:

```bash
python scripts/inspect_datasets.py
python scripts/inspect_datasets.py --dataset stock_provided --load --max-files 3
```

Benchmark outputs should use the canonical schemas in `src/evaluation/results_schema.py`.
Planned dashboard figures are declared in `src/visualization/specs.py` so the report can be built from stable result tables.

Dataset placement and materialization are documented in `docs/DATASETS.md`.

## Roadmap

- Phase 1: ✅ scaffolding, interfaces, metrics, configs, smoke test
- Phase 2: ⏳ data registry, governance implementation, and multi-domain dataset ingestion
- Phase 3: ⏳ TabPFN-TS and baseline model wrappers
- Phase 4: ⏳ benchmark runs across domains and low-data regimes
- Phase 5: ⏳ quant backtest, ablations, and executive visual report
