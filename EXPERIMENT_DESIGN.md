# Multi-Domain Experiment Design

## Execution policy

Do not run experiments on the local machine. All smoke tests, benchmark runs, and model training jobs should run on `A00`.

Before each run:

```bash
ssh A00
nvidia-smi
```

Only launch jobs when GPU memory and utilization are low enough to avoid interfering with other users' experiments. From the local checkout, `bash scripts/check_a00_gpu.sh` can be used as a quick read-only status check.

## Available local dataset

The existing `../market_data_downloader` folder contains 1-hour and 1-day OHLCV CSV files for A-share Shanghai listings, Hong Kong stocks, and crypto assets. Phase 1 keeps the files in place and references them through `configs/dataset/stock_provided.yaml`; later phases should materialize governed parquet panels under `data/processed/`.

This dataset is useful for the finance application and governance/backtest phases, but it should not define the whole study. The main benchmark should also include non-financial datasets such as ETT, electricity, exchange rate, traffic, weather, and any additional public time-series panels selected in later phases.

## Proposed experiments

1. **E1 zero-shot smoke and forecasting sanity check**
   - Use synthetic sine data now, then small curated subsets from public benchmarks and `stock_provided`.
   - Compare Dummy Mean, Seasonal Naive, ARIMA, TabPFN-TS, and foundation baselines.
   - Metrics: MSE, MAE, RMSE, SMAPE, WAPE, MASE.

2. **E2 low-data regime**
   - Context windows should be dataset-aware, for example 24, 48, 96, 168, 240 bars for hourly data and 60, 120, 252 bars for daily data.
   - Measure degradation curves by domain, dataset, and model.
   - Key figure: performance versus context length with confidence ribbons.

3. **E3 cross-domain generalization**
   - Evaluate separately on energy, traffic, weather, exchange-rate, finance, and any added public domains.
   - Keep chronological walk-forward splits and never mix future information across series or assets.
   - Key table: per-domain rank of each model plus domain variance.

4. **E5 governance ablation**
   - For finance, compare full governance against missing-value only, no outlier treatment, no rank normalization, and no halt handling.
   - For non-financial datasets, use analogous ablations: missing handling, outlier handling, calendar alignment, and scaling.
   - Metrics: IC, Rank IC, ICIR, Sharpe, max drawdown.

5. **E6 tradable signal backtest**
   - Convert forecasts to next-period return signals.
   - Use top-bottom quantile long-short portfolios with transaction-cost sensitivity.
   - Report stability probability: `P(return > 0 and Sharpe > 0 and MaxDD <= 20%)`.

## Fancy visualization plan

Use Plotly for interactive HTML dashboards and export static PNGs for reports:

- **Model leaderboard ridge plot**: distribution of normalized error or relative improvement by model, colored by domain.
- **Context degradation fan chart**: x-axis context length, y-axis normalized error, ribbons for bootstrapped uncertainty.
- **Cross-domain heatmap matrix**: model by dataset/domain, annotated with relative rank and normalized metric values.
- **Forecast panel gallery**: small multiples of actual versus predicted trajectories with uncertainty bands where available.
- **Error topology map**: UMAP/t-SNE projection of datasets using seasonal/trend/volatility descriptors, colored by the best model.
- **Finance cumulative return arena**: synchronized cumulative return and drawdown panels with hoverable event windows.
- **Finance risk-return bubble map**: x-axis max drawdown, y-axis annualized return, bubble size as stability probability, color as model family.
- **Finance signal decay curve**: IC by forecast horizon to show whether predictions are tradable or merely contemporaneous.
- **Governance waterfall**: step-by-step change in forecast metrics and finance metrics from raw data to fully governed data.

The final report should open with one executive dashboard for cross-domain benchmark performance, then use dedicated drill-down tabs for domain diagnostics and a separate finance application section.
