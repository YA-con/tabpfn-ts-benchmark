#!/usr/bin/env python
"""Generate a temporal forecast-performance report from benchmark predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SHORT_NAMES = {
    "tabpfn_ts": "TabPFN-TS",
    "lightgbm_ar": "LightGBM",
    "xgboost_ar": "XGBoost",
    "hist_gradient_boosting_ar": "HistGBR",
    "ridge_ar": "Ridge-AR",
    "moving_average": "MovingAvg",
    "seasonal_naive": "Seasonal",
    "linear_trend": "Trend",
    "dummy_mean": "Mean",
}

MODEL_COLORS = {
    "tabpfn_ts": "#0f766e",
    "lightgbm_ar": "#2563eb",
    "xgboost_ar": "#7c3aed",
    "hist_gradient_boosting_ar": "#db2777",
    "ridge_ar": "#ea580c",
    "moving_average": "#64748b",
    "seasonal_naive": "#94a3b8",
    "linear_trend": "#b45309",
    "dummy_mean": "#475569",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _scenario_from_run(run: str) -> tuple[int, int, str]:
    context = int(run.split("_c")[1].split("_h")[0])
    horizon = int(run.split("_h")[1].split("_s")[0])
    return context, horizon, f"c{context}/h{horizon}"


def _load_predictions(results_glob: str) -> pd.DataFrame:
    frames = []
    for path in sorted(PROJECT_ROOT.glob(results_glob)):
        run = path.parent.name
        context, horizon, scenario = _scenario_from_run(run)
        frame = pd.read_csv(path)
        frame["matrix_run"] = run
        frame["context"] = context
        frame["horizon"] = horizon
        frame["scenario"] = scenario
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No prediction files matched: {results_glob}")
    predictions = pd.concat(frames, ignore_index=True)
    predictions["ds"] = pd.to_datetime(predictions["ds"], errors="coerce")
    predictions = predictions.sort_values(["matrix_run", "dataset", "model", "unique_id", "ds"])
    predictions["lead_step"] = (
        predictions.groupby(["matrix_run", "dataset", "model", "unique_id"]).cumcount() + 1
    )
    denom = predictions["y"].abs() + predictions["y_hat"].abs()
    predictions["smape_point"] = np.where(
        denom > 1e-12,
        200.0 * (predictions["y"] - predictions["y_hat"]).abs() / denom,
        0.0,
    )
    predictions["abs_error"] = (predictions["y"] - predictions["y_hat"]).abs()
    return predictions


def _configure_matplotlib() -> None:
    candidates = [
        "Heiti TC",
        "PingFang SC",
        "Songti SC",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": candidates,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d7dee8",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#e6edf5",
            "grid.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#e2e8f0",
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path.name


def _model_order(step_metrics: pd.DataFrame) -> list[str]:
    return (
        step_metrics.groupby("model")["smape_point"]
        .mean()
        .sort_values()
        .index.tolist()
    )


def _plot_temporal_curves(step_metrics: pd.DataFrame, out: Path) -> str:
    order = _model_order(step_metrics)
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    for model in order[:7]:
        data = step_metrics[step_metrics["model"] == model].sort_values("lead_step")
        ax.plot(
            data["lead_step"],
            data["smape_point"],
            label=SHORT_NAMES.get(model, model),
            color=MODEL_COLORS.get(model, "#2563eb"),
            linewidth=3.2 if model == "tabpfn_ts" else 2.0,
            alpha=1.0 if model == "tabpfn_ts" else 0.72,
            marker="o" if model == "tabpfn_ts" else None,
            markersize=4,
        )
    ax.set_title("预测越往后，谁退化得更快")
    ax.set_xlabel("预测步长 / lead step")
    ax.set_ylabel("逐步 SMAPE，越低越好")
    ax.grid(axis="y", alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=10)
    best = step_metrics.groupby("model")["smape_point"].mean().idxmin()
    ax.annotate(
        f"整体最低: {SHORT_NAMES.get(best, best)}",
        xy=(0.98, 0.94),
        xycoords="axes fraction",
        ha="right",
        color=MODEL_COLORS.get(best, "#0f766e"),
        fontweight="bold",
    )
    return _save(fig, out / "temporal_smape_curves.svg")


def _plot_rank_heatmap(step_metrics: pd.DataFrame, out: Path) -> str:
    ranked = step_metrics.copy()
    ranked["rank"] = ranked.groupby("lead_step")["smape_point"].rank(method="min", ascending=True)
    order = _model_order(step_metrics)
    pivot = ranked.pivot(index="model", columns="lead_step", values="rank").reindex(order)
    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis_r", vmin=1, vmax=len(order))
    ax.set_title("不同预测步长上的模型排名")
    ax.set_xlabel("预测步长 / lead step")
    ax.set_ylabel("")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT_NAMES.get(model, model) for model in order])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(step)) for step in pivot.columns])
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iat[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{int(value)}", ha="center", va="center", fontsize=8, color="white" if value <= 4 else "#0f172a")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Rank，越低越好")
    return _save(fig, out / "temporal_rank_heatmap.svg")


def _plot_tabpfn_delta(predictions: pd.DataFrame, out: Path) -> str | None:
    step_task = (
        predictions.groupby(["matrix_run", "dataset", "unique_id", "lead_step", "model"], as_index=False)[
            "smape_point"
        ].mean()
    )
    if "tabpfn_ts" not in set(step_task["model"]):
        return None
    tab = step_task[step_task["model"] == "tabpfn_ts"].rename(columns={"smape_point": "tabpfn_smape"})
    base = (
        step_task[step_task["model"] != "tabpfn_ts"]
        .groupby(["matrix_run", "dataset", "unique_id", "lead_step"], as_index=False)["smape_point"]
        .min()
        .rename(columns={"smape_point": "best_baseline_smape"})
    )
    delta = tab.merge(base, on=["matrix_run", "dataset", "unique_id", "lead_step"], how="inner")
    delta["delta"] = delta["tabpfn_smape"] - delta["best_baseline_smape"]
    curve = delta.groupby("lead_step", as_index=False)["delta"].mean()
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.axhline(0, color="#0f172a", linewidth=1.2)
    ax.fill_between(
        curve["lead_step"],
        0,
        curve["delta"],
        where=curve["delta"] <= 0,
        color="#0f766e",
        alpha=0.16,
        interpolate=True,
        label="TabPFN-TS 更好",
    )
    ax.fill_between(
        curve["lead_step"],
        0,
        curve["delta"],
        where=curve["delta"] > 0,
        color="#dc2626",
        alpha=0.14,
        interpolate=True,
        label="最佳 baseline 更好",
    )
    ax.plot(curve["lead_step"], curve["delta"], color="#0f766e", linewidth=3, marker="o", markersize=4)
    ax.set_title("TabPFN-TS 相对每个时刻最佳 baseline 的差距")
    ax.set_xlabel("预测步长 / lead step")
    ax.set_ylabel("SMAPE 差值：TabPFN - best baseline")
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10)
    return _save(fig, out / "tabpfn_vs_best_baseline_by_step.svg")


def _plot_early_late_shift(step_metrics: pd.DataFrame, out: Path) -> str:
    max_step = int(step_metrics["lead_step"].max())
    split = max(1, max_step // 2)
    early = (
        step_metrics[step_metrics["lead_step"] <= split]
        .groupby("model")["smape_point"]
        .mean()
        .rename("early_smape")
    )
    late = (
        step_metrics[step_metrics["lead_step"] > split]
        .groupby("model")["smape_point"]
        .mean()
        .rename("late_smape")
    )
    shift = pd.concat([early, late], axis=1).dropna()
    shift["degradation"] = shift["late_smape"] - shift["early_smape"]
    shift = shift.sort_values("degradation")
    colors = [MODEL_COLORS.get(model, "#2563eb") for model in shift.index]
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    ax.barh([SHORT_NAMES.get(model, model) for model in shift.index], shift["degradation"], color=colors, alpha=0.86)
    ax.axvline(0, color="#0f172a", linewidth=1.1)
    ax.set_title("后半段预测相比前半段多损失多少")
    ax.set_xlabel("Late SMAPE - Early SMAPE，越小越稳")
    ax.grid(axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out / "early_late_degradation.svg")


def build_report(results_glob: str, output: Path) -> Path:
    predictions = _load_predictions(results_glob)
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    _configure_matplotlib()

    step_metrics = (
        predictions.groupby(["model", "lead_step"], as_index=False)
        .agg(smape_point=("smape_point", "mean"), abs_error=("abs_error", "mean"))
        .sort_values(["model", "lead_step"])
    )
    scenario_step = (
        predictions.groupby(["scenario", "dataset", "model", "lead_step"], as_index=False)["smape_point"]
        .mean()
        .sort_values(["scenario", "dataset", "model", "lead_step"])
    )
    model_summary = (
        step_metrics.groupby("model")
        .agg(mean_smape=("smape_point", "mean"), final_smape=("smape_point", "last"))
        .sort_values("mean_smape")
    )
    model_summary["display"] = [SHORT_NAMES.get(model, model) for model in model_summary.index]
    model_summary.to_csv(output / "temporal_model_summary.csv")
    scenario_step.to_csv(output / "temporal_step_metrics.csv", index=False)

    figures = [
        {
            "title": "误差随预测步长变化",
            "file": _plot_temporal_curves(step_metrics, assets),
            "note": "看模型是否越预测越远越不稳定。曲线越低越好，越平越稳。",
        },
        {
            "title": "逐步排名热力图",
            "file": _plot_rank_heatmap(step_metrics, assets),
            "note": "看每一个 lead step 上模型排名是否改变，避免只看平均值。",
        },
        {
            "title": "TabPFN-TS 与最佳 baseline 的逐步差距",
            "file": _plot_tabpfn_delta(predictions, assets),
            "note": "低于 0 表示 TabPFN-TS 在该步长平均优于当时最佳 baseline。",
        },
        {
            "title": "前半段到后半段的退化幅度",
            "file": _plot_early_late_shift(step_metrics, assets),
            "note": "衡量模型从短期预测走向较长期预测时，误差增加得有多快。",
        },
    ]
    figures = [item for item in figures if item["file"]]
    best_model = model_summary.index[0]
    payload = _json_safe(
        {
            "models": [
                {
                    "id": model,
                    "display": SHORT_NAMES.get(model, model),
                    "color": MODEL_COLORS.get(model, "#2563eb"),
                }
                for model in model_summary.index
            ],
            "stepMetrics": step_metrics.to_dict(orient="records"),
            "scenarioStepMetrics": scenario_step.to_dict(orient="records"),
        }
    )
    cards = "\n".join(
        f"<div class=\"kpi\"><span>{row.display}</span><b>{row.mean_smape:.3f}</b><em>final {row.final_smape:.3f}</em></div>"
        for _, row in model_summary.head(5).iterrows()
    )
    figure_html = "\n".join(
        f"""
        <article class="figure-card">
          <div class="figure-copy"><h2>{item['title']}</h2><p>{item['note']}</p><a href="assets/{item['file']}">打开 SVG</a></div>
          <img src="assets/{item['file']}" alt="{item['title']}"/>
        </article>
        """
        for item in figures
    )
    table_rows = "\n".join(
        f"<tr><td>{SHORT_NAMES.get(model, model)}</td><td>{model}</td><td>{row.mean_smape:.4f}</td><td>{row.final_smape:.4f}</td></tr>"
        for model, row in model_summary.iterrows()
    )
    data_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Temporal Performance Report</title>
<style>
:root {{ --bg:#f6f8fb; --panel:#fff; --text:#0f172a; --muted:#64748b; --line:#dbe4ef; --accent:#0f766e; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter,-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif; }}
main {{ max-width:1480px; margin:0 auto; padding:34px; }}
.hero {{ display:grid; grid-template-columns:1.2fr .8fr; gap:22px; padding:26px 0 24px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0; font-size:42px; line-height:1.08; letter-spacing:0; }}
.sub {{ color:var(--muted); line-height:1.7; }}
.panel,.figure-card,.kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; box-shadow:0 14px 42px rgba(15,23,42,.055); }}
.panel {{ padding:18px; }}
.kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin:22px 0; }}
.kpi {{ padding:16px; display:grid; gap:8px; }}
.kpi span {{ color:var(--muted); font-size:12px; font-weight:800; }}
.kpi b {{ font-size:30px; }}
.kpi em {{ color:var(--muted); font-style:normal; font-size:12px; }}
.grid {{ display:grid; gap:22px; }}
.figure-card {{ display:grid; grid-template-columns:minmax(260px,.34fr) minmax(0,.66fr); gap:18px; padding:18px; align-items:start; }}
h2 {{ margin:0 0 10px; font-size:18px; }}
p {{ color:var(--muted); line-height:1.65; }}
a {{ color:var(--accent); font-weight:900; text-decoration:none; font-size:13px; }}
img {{ width:100%; display:block; border-radius:12px; background:white; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:10px; border-bottom:1px solid #e7edf5; text-align:right; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
th {{ color:var(--muted); }}
@media(max-width:960px) {{ main {{ padding:16px; }} .hero,.figure-card {{ grid-template-columns:1fr; }} .kpis {{ grid-template-columns:1fr 1fr; }} h1 {{ font-size:31px; }} }}
</style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <h1>时序步长上的性能差别</h1>
      <p class="sub">这份报告不只看平均分，而是把每个预测步长拆开：短期谁准、越往后谁退化、TabPFN-TS 在哪些步长相对 baseline 更占优。</p>
    </div>
    <div class="panel">
      <b>当前整体最低</b>
      <p>{SHORT_NAMES.get(best_model, best_model)} 的逐步平均 SMAPE 最低。所有数值来自真实 forecast_predictions.csv 重新聚合。</p>
    </div>
  </section>
  <section class="kpis">{cards}</section>
  <section class="grid">{figure_html}</section>
  <section class="panel" style="margin-top:22px">
    <h2>模型逐步误差摘要</h2>
    <table><thead><tr><th>Display</th><th>Raw model</th><th>Mean step SMAPE</th><th>Final step SMAPE</th></tr></thead><tbody>{table_rows}</tbody></table>
  </section>
</main>
<script id="temporal-payload" type="application/json">{data_json}</script>
</body>
</html>"""
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "temporal_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output / "index.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-glob", default="results/real_benchmark_v1_c*_h*_s3/forecast_predictions.csv")
    parser.add_argument("--out", default="reports/real_benchmark_v1_temporal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_report(args.results_glob, PROJECT_ROOT / args.out)
    print(path)


if __name__ == "__main__":
    main()
