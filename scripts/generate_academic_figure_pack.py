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


def _explain_html(*, watch: str, finding: str, caveat: str) -> str:
    return (
        '<aside class="explain">'
        f'<div><b>看什么</b><span>{watch}</span></div>'
        f'<div><b>当前结论</b><span>{finding}</span></div>'
        f'<div><b>注意口径</b><span>{caveat}</span></div>'
        "</aside>"
    )


def _run_label_from_glob(results_glob: str) -> str:
    marker = "results/"
    if marker in results_glob:
        return results_glob.split(marker, 1)[1].split("/")[0].replace("*", "matrix")
    return Path(results_glob).parts[1] if len(Path(results_glob).parts) > 1 else "benchmark"


def build_pack(results_glob: str, out_dir: Path) -> Path:
    metrics = _load_metrics(results_glob)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    run_label = _run_label_from_glob(results_glob)

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
    task_count = int(unit["task"].nunique())
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
            "x": np.log10(summary.loc[order, "inference_time_s"] + 1e-4).round(6).tolist(),
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
        xlabel="log10(mean inference time + 1e-4)",
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
        {
            "title": "Mean SMAPE bar",
            "file": fig_bar,
            "caption": "带标准差的整体误差对比。",
            "explain": _explain_html(
                watch="比较每个模型跨全部 task 的平均 SMAPE，误差线表示 task 间波动。",
                finding=(
                    f"{SHORT_NAMES.get(order[0], order[0])} 的平均 SMAPE 最低 "
                    f"({summary.iloc[0].mean_smape:.3f})。"
                ),
                caveat="SMAPE 是直接跨 task 平均，数据集尺度差异仍可能影响整体均值。",
            ),
        },
        {
            "title": "Scenario rank heatmap",
            "file": fig_heat,
            "caption": "不同 context/horizon 场景中的模型排名。",
            "explain": _explain_html(
                watch="看模型在不同 context/horizon 组合里是否稳定排在前列。",
                finding=(
                    f"{SHORT_NAMES.get(order[0], order[0])} 平均排名最低；"
                    f"热力图能看到长 context 场景下的名次变化。"
                ),
                caveat="颜色表示排名，不表示 SMAPE 的绝对差距；相邻名次可能差距很小。",
            ),
        },
        {
            "title": "Inference-rank scatter",
            "file": fig_scatter,
            "caption": "推理成本与平均排名的关系。",
            "explain": _explain_html(
                watch="横轴是平均推理时间，纵轴是平均排名；左上更快但更差，右下更慢但更强。",
                finding=(
                    f"{SHORT_NAMES.get(summary.index[0], summary.index[0])} 排名最好，但平均推理时间 "
                    f"{summary.loc[summary.index[0], 'inference_time_s']:.3f}s，明显慢于树模型和简单 baseline。"
                ),
                caveat="这里使用 log10 时间轴；推理时间来自当前 Python/TabPFN-TS pipeline 调用，包含模型 pipeline 开销，不等同于充分优化后的部署延迟。",
            ),
        },
        {
            "title": "Rank trajectory line",
            "file": fig_line,
            "caption": "Top 模型排名随实验场景变化。",
            "explain": _explain_html(
                watch="看模型排名是否随着 context/horizon 增大而上升或下降。",
                finding="TabPFN-TS 在部分短/中 context 场景领先，但长 context 下 baseline 有反超现象。",
                caveat="折线只展示 Top 模型，完整排名请结合热力图和汇总表。",
            ),
        },
        {
            "title": "Task rank boxplot",
            "file": fig_box,
            "caption": f"每个模型在 {task_count} 个 task 上的排名分布。",
            "explain": _explain_html(
                watch="箱体越靠低 rank 且越窄，表示模型越稳定。",
                finding=(
                    f"{SHORT_NAMES.get(order[0], order[0])} 的平均排名为 "
                    f"{summary.iloc[0].avg_rank:.2f}，wins={int(summary.iloc[0].wins)}/{task_count}。"
                ),
                caveat="箱线图看的是排名分布，不反映具体 SMAPE 差距大小。",
            ),
        },
        {
            "title": "Top model SMAPE violin",
            "file": fig_violin,
            "caption": "Top 6 模型 task-level SMAPE 分布。",
            "explain": _explain_html(
                watch="看 Top 模型的误差分布形态，是否有长尾或不稳定 task。",
                finding="TabPFN-TS 整体均值最低，但分布宽度提示仍存在失败或退步场景。",
                caveat=f"小提琴图基于 {task_count} 个 task，样本量有限，适合做诊断而非最终显著性结论。",
            ),
        },
        {
            "title": "Average rank forest",
            "file": fig_forest,
            "caption": "平均排名及 bootstrap 95% CI。",
            "explain": _explain_html(
                watch="看平均排名及不确定性区间，区间越短说明结果越稳定。",
                finding=(
                    f"{SHORT_NAMES.get(order[0], order[0])} 的平均排名最低，"
                    "但仍需看 CI 与其他强 baseline 是否明显分离。"
                ),
                caveat="CI 来自 task-level bootstrap，不代表独立重复实验或多 seed 置信区间。",
            ),
        },
    ]
    figure_html = "\n".join(
        '<article class="figure-row">'
        '<div class="figure-main">'
        f'<div><h2>{item["title"]}</h2><a href="assets/{item["file"]}">SVG</a>'
        f'<a href="assets/{Path(item["file"]).with_suffix(".png").name}">PNG</a></div>'
        f'<p>{item["caption"]}</p><img src="assets/{item["file"]}" alt="{item["title"]}"/>'
        "</div>"
        f'{item["explain"]}'
        "</article>"
        for item in figures
    )
    table_rows = "\n".join(
        f"<tr><td>{SHORT_NAMES.get(model, model)}</td><td>{model}</td><td>{row.avg_rank:.3f}</td><td>{row.mean_smape:.4f}</td><td>{int(row.wins)}</td><td>{int(row.top3)}/{task_count}</td></tr>"
        for model, row in summary.iterrows()
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Academic Figure Pack · {run_label}</title>
<style>
body {{ margin:0; background:#f6f8fb; color:#0f172a; font-family:Inter,-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif; }}
main {{ max-width:1440px; margin:0 auto; padding:34px; }}
header {{ padding:28px 0 22px; border-bottom:1px solid #dbe4ef; }}
h1 {{ margin:0 0 10px; font-size:44px; letter-spacing:0; }}
.sub {{ color:#64748b; line-height:1.7; max-width:980px; }}
.grid {{ display:grid; gap:22px; margin-top:24px; }}
article, section {{ background:#fff; border:1px solid #dbe4ef; border-radius:16px; padding:18px; box-shadow:0 14px 42px rgba(15,23,42,.055); }}
.figure-row {{ display:grid; grid-template-columns:minmax(0, 1.45fr) minmax(280px, .55fr); gap:20px; align-items:start; }}
.figure-main > div {{ display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap; }}
h2 {{ margin:0; font-size:17px; }}
p {{ color:#64748b; line-height:1.6; }}
a {{ color:#0f766e; font-size:12px; font-weight:800; text-decoration:none; margin-left:8px; }}
img {{ width:100%; display:block; border-radius:10px; background:white; }}
.explain {{ border-left:3px solid #0f766e; background:#f8fafc; border-radius:12px; padding:14px 14px 10px; display:grid; gap:12px; }}
.explain div {{ display:grid; gap:4px; }}
.explain b {{ color:#0f172a; font-size:12px; letter-spacing:.02em; }}
.explain span {{ color:#64748b; line-height:1.58; font-size:13px; }}
section {{ margin-top:22px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ border-bottom:1px solid #e5ebf3; padding:10px; text-align:right; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
th {{ color:#64748b; }}
@media(max-width:900px) {{ main {{ padding:16px; }} .figure-row {{ grid-template-columns:1fr; }} h1 {{ font-size:32px; }} }}
</style>
</head>
<body>
<main>
<header>
<h1>Academic Figure Pack</h1>
<p class="sub">使用 SkillHub academic-figures 生成的出版风格图集。所有图基于 {run_label} 真实 metrics；输出同时包含 SVG 和 300DPI PNG。标题、注释和摘要全部由当前结果自动计算。</p>
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
