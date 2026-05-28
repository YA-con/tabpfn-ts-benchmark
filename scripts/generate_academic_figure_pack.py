#!/usr/bin/env python
"""Generate publication-style figures with the academic-figures skill."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPT = PROJECT_ROOT / "skills" / "academic-figures" / "scripts" / "gen_figure.py"

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


def _scenario_from_run(run: str) -> tuple[int, int, str]:
    context = int(run.split("_c")[1].split("_h")[0])
    horizon = int(run.split("_h")[1].split("_s")[0])
    return context, horizon, f"c{context}/h{horizon}"


def _load_metrics(results_glob: str) -> pd.DataFrame:
    frames = []
    for path in sorted(PROJECT_ROOT.glob(results_glob)):
        run = path.parent.name
        context, horizon, scenario = _scenario_from_run(run)
        frame = pd.read_csv(path)
        frame["matrix_run"] = run
        frame["context"] = context
        frame["horizon"] = horizon
        frame["scenario"] = scenario
        frame["task"] = run + "|" + frame["dataset"].astype(str)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No metrics matched {results_glob}")
    return pd.concat(frames, ignore_index=True)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _run_figure(kind: str, data: Path, out: Path, *, title: str, xlabel: str = "", ylabel: str = "", width: float = 10, height: float = 6, extra: list[str] | None = None) -> None:
    cmd = [
        sys.executable,
        str(SKILL_SCRIPT),
        "--type",
        kind,
        "--data",
        str(data),
        "--out",
        str(out),
        "--theme",
        "nature",
        "--cjk",
        "--width",
        str(width),
        "--height",
        str(height),
        "--title",
        title,
    ]
    if xlabel:
        cmd += ["--xlabel", xlabel]
    if ylabel:
        cmd += ["--ylabel", ylabel]
    if extra:
        cmd += extra
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _render_both(kind: str, data: Path, assets: Path, stem: str, **kwargs) -> str:
    svg = assets / f"{stem}.svg"
    png = assets / f"{stem}.png"
    _run_figure(kind, data, svg, **kwargs)
    _run_figure(kind, data, png, **kwargs)
    return svg.name


def build_pack(results_glob: str, out_dir: Path) -> Path:
    metrics = _load_metrics(results_glob)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    unit = (
        metrics.groupby(["task", "matrix_run", "scenario", "context", "horizon", "dataset", "model"], as_index=False)
        .agg(
            smape=("smape", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mase=("mase", "mean"),
            inference_time_s=("inference_time_s", "mean"),
        )
    )
    unit["rank"] = unit.groupby("task")["smape"].rank(method="min", ascending=True).astype(int)
    summary = (
        unit.groupby("model")
        .agg(
            mean_smape=("smape", "mean"),
            std_smape=("smape", "std"),
            avg_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            median_rank=("rank", "median"),
            wins=("rank", lambda s: int((s == 1).sum())),
            top3=("rank", lambda s: int((s <= 3).sum())),
            inference_time_s=("inference_time_s", "mean"),
        )
        .sort_values(["avg_rank", "mean_smape"])
    )
    order = summary.index.tolist()
    labels = [SHORT_NAMES.get(model, model) for model in order]
    unit["model_short"] = unit["model"].map(lambda x: SHORT_NAMES.get(x, x))
    unit.to_csv(out_dir / "task_level_metrics.csv", index=False)
    summary.to_csv(out_dir / "model_summary.csv")

    # 1. Grouped bar: mean SMAPE and average rank in one publication-friendly comparison.
    bar_json = _write_json(
        data_dir / "01_model_bar.json",
        {
            "labels": labels,
            "series": {"Mean SMAPE": summary.loc[order, "mean_smape"].round(4).tolist()},
            "errors": {"Mean SMAPE": summary.loc[order, "std_smape"].fillna(0).round(4).tolist()},
        },
    )
    fig_bar = _render_both(
        "bar",
        bar_json,
        assets,
        "01_model_mean_smape_bar",
        title="模型整体误差对比 / Mean SMAPE",
        ylabel="Mean SMAPE (lower is better)",
        width=11.5,
        height=5.8,
    )

    # 2. Heatmap: scenario x model ranks.
    scenario_order = (
        unit[["matrix_run", "scenario", "context", "horizon"]]
        .drop_duplicates()
        .sort_values(["context", "horizon"])["scenario"]
        .tolist()
    )
    rank_scenario = unit.groupby(["scenario", "model"], as_index=False)["smape"].mean()
    rank_scenario["rank"] = rank_scenario.groupby("scenario")["smape"].rank(method="min", ascending=True).astype(int)
    heat = rank_scenario.pivot(index="model", columns="scenario", values="rank").loc[order, scenario_order]
    heat_json = _write_json(
        data_dir / "02_rank_heatmap.json",
        {
            "matrix": heat.to_numpy(dtype=float).tolist(),
            "row_labels": labels,
            "col_labels": scenario_order,
            "cbar_label": "Rank",
        },
    )
    fig_heat = _render_both(
        "heatmap",
        heat_json,
        assets,
        "02_scenario_rank_heatmap",
        title="不同实验场景下的模型排名 / Scenario-wise Rank",
        width=9.5,
        height=6.4,
        extra=["--cmap", "viridis_r", "--vmin", "1", "--vmax", str(len(order))],
    )

    # 3. Scatter: inference time vs average rank.
    scatter_json = _write_json(
        data_dir / "03_pareto_scatter.json",
        {
            "x": (summary.loc[order, "inference_time_s"] + 1e-4).round(6).tolist(),
            "y": summary.loc[order, "avg_rank"].round(4).tolist(),
            "groups": labels,
        },
    )
    fig_scatter = _render_both(
        "scatter",
        scatter_json,
        assets,
        "03_inference_rank_scatter",
        title="推理成本与平均排名 / Inference Cost vs Rank",
        xlabel="Mean inference time (s)",
        ylabel="Average rank (lower is better)",
        width=8.8,
        height=6.0,
        extra=["--no-trend"],
    )

    # 4. Line: rank trajectory across context/horizon scenarios.
    line_series = {}
    line_errors = {}
    for model in order[:6]:
        model_rows = (
            unit[unit["model"] == model]
            .groupby("scenario")
            .agg(mean_rank=("rank", "mean"), std_rank=("rank", "std"))
            .reindex(scenario_order)
        )
        line_series[SHORT_NAMES.get(model, model)] = model_rows["mean_rank"].round(4).tolist()
        line_errors[SHORT_NAMES.get(model, model)] = model_rows["std_rank"].fillna(0).round(4).tolist()
    line_json = _write_json(
        data_dir / "04_rank_trajectory.json",
        {"labels": scenario_order, "series": line_series, "errors": line_errors},
    )
    fig_line = _render_both(
        "line",
        line_json,
        assets,
        "04_rank_trajectory_line",
        title="模型排名轨迹 / Rank Trajectory",
        ylabel="Average rank (lower is better)",
        width=10.8,
        height=6.0,
    )

    # 5. Box plot: task-level ranks.
    box_json = _write_json(
        data_dir / "05_rank_box.json",
        {
            "labels": labels,
            "series": {
                SHORT_NAMES.get(model, model): unit.loc[unit["model"] == model, "rank"].astype(float).tolist()
                for model in order
            },
        },
    )
    fig_box = _render_both(
        "box",
        box_json,
        assets,
        "05_task_rank_boxplot",
        title="Task-level 排名分布 / Rank Distribution",
        ylabel="Rank (lower is better)",
        width=11.2,
        height=5.8,
    )

    # 6. Violin: task-level SMAPE distribution for top six models.
    top6 = order[:6]
    violin_json = _write_json(
        data_dir / "06_smape_violin.json",
        {
            "labels": [SHORT_NAMES.get(model, model) for model in top6],
            "series": {
                SHORT_NAMES.get(model, model): unit.loc[unit["model"] == model, "smape"].astype(float).round(6).tolist()
                for model in top6
            },
        },
    )
    fig_violin = _render_both(
        "violin",
        violin_json,
        assets,
        "06_top_model_smape_violin",
        title="Top 模型 SMAPE 分布 / SMAPE Distribution",
        ylabel="SMAPE",
        width=9.6,
        height=5.8,
    )

    # 7. Forest plot: average rank with bootstrap CI.
    rng = np.random.default_rng(42)
    estimates, ci_low, ci_high = [], [], []
    for model in order:
        values = unit.loc[unit["model"] == model, "rank"].to_numpy(dtype=float)
        boots = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(2000)]
        estimates.append(float(values.mean()))
        ci_low.append(float(np.percentile(boots, 2.5)))
        ci_high.append(float(np.percentile(boots, 97.5)))
    forest_json = _write_json(
        data_dir / "07_average_rank_forest.json",
        {
            "labels": labels,
            "estimates": estimates,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "overall": {
                "estimate": float(np.mean(estimates)),
                "ci_low": float(np.mean(ci_low)),
                "ci_high": float(np.mean(ci_high)),
            },
            "ref_line": 1.0,
        },
    )
    fig_forest = _render_both(
        "forest",
        forest_json,
        assets,
        "07_average_rank_forest",
        title="平均排名森林图 / Average Rank with 95% CI",
        xlabel="Average rank (lower is better)",
        width=10.5,
        height=6.8,
    )

    figures = [
        ("Mean SMAPE bar", fig_bar, "带标准差的整体误差对比。"),
        ("Scenario rank heatmap", fig_heat, "不同 context/horizon 场景中的模型排名。"),
        ("Inference-rank scatter", fig_scatter, "推理成本与平均排名的关系。"),
        ("Rank trajectory line", fig_line, "Top 模型排名随实验场景变化。"),
        ("Task rank boxplot", fig_box, "每个模型在 25 个 task 上的排名分布。"),
        ("Top model SMAPE violin", fig_violin, "Top 6 模型 task-level SMAPE 分布。"),
        ("Average rank forest", fig_forest, "平均排名及 bootstrap 95% CI。"),
    ]
    figure_html = "\n".join(
        f'<article><div><h2>{title}</h2><a href="assets/{file}">SVG</a><a href="assets/{Path(file).with_suffix(".png").name}">PNG</a></div><p>{caption}</p><img src="assets/{file}" alt="{title}"/></article>'
        for title, file, caption in figures
    )
    table_rows = "\n".join(
        f"<tr><td>{SHORT_NAMES.get(model, model)}</td><td>{model}</td><td>{row.avg_rank:.3f}</td><td>{row.mean_smape:.4f}</td><td>{int(row.wins)}</td><td>{int(row.top3)}/25</td></tr>"
        for model, row in summary.iterrows()
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Academic Figure Pack · Full Benchmark v1</title>
<style>
body {{ margin:0; background:#f6f8fb; color:#0f172a; font-family:Inter,-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif; }}
main {{ max-width:1440px; margin:0 auto; padding:34px; }}
header {{ padding:28px 0 22px; border-bottom:1px solid #dbe4ef; }}
h1 {{ margin:0 0 10px; font-size:44px; letter-spacing:0; }}
.sub {{ color:#64748b; line-height:1.7; max-width:980px; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px; margin-top:24px; }}
article, section {{ background:#fff; border:1px solid #dbe4ef; border-radius:16px; padding:18px; box-shadow:0 14px 42px rgba(15,23,42,.055); }}
article div {{ display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap; }}
h2 {{ margin:0; font-size:17px; }}
p {{ color:#64748b; line-height:1.6; }}
a {{ color:#0f766e; font-size:12px; font-weight:800; text-decoration:none; margin-left:8px; }}
img {{ width:100%; display:block; border-radius:10px; background:white; }}
section {{ margin-top:22px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ border-bottom:1px solid #e5ebf3; padding:10px; text-align:right; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
th {{ color:#64748b; }}
@media(max-width:900px) {{ main {{ padding:16px; }} .grid {{ grid-template-columns:1fr; }} h1 {{ font-size:32px; }} }}
</style>
</head>
<body>
<main>
<header>
<h1>Academic Figure Pack</h1>
<p class="sub">使用 SkillHub academic-figures 生成的出版风格图集。所有图基于 full_benchmark_v1 真实 metrics；输出同时包含 SVG 和 300DPI PNG。</p>
</header>
<div class="grid">{figure_html}</div>
<section>
<h2>Model summary</h2>
<table><thead><tr><th>Display</th><th>Raw model</th><th>Avg rank</th><th>Mean SMAPE</th><th>Wins</th><th>Top-3</th></tr></thead><tbody>{table_rows}</tbody></table>
</section>
</main>
</body>
</html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    return out_dir / "index.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-glob", default="results/full_benchmark_v1_c*_h*_s2/forecast_metrics.csv")
    parser.add_argument("--out", default="reports/full_benchmark_v1_academic_figures")
    return parser.parse_args()


def main() -> None:
    if not SKILL_SCRIPT.exists():
        raise FileNotFoundError(f"academic-figures skill script not found: {SKILL_SCRIPT}")
    args = parse_args()
    path = build_pack(args.results_glob, PROJECT_ROOT / args.out)
    print(path)


if __name__ == "__main__":
    main()
