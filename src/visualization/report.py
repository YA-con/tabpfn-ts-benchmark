"""HTML report generation for pilot benchmark results."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


PALETTE = {
    "dummy_mean": "#7c3aed",
    "seasonal_naive": "#0f766e",
    "energy": "#2563eb",
    "traffic": "#f97316",
    "weather": "#0891b2",
    "economics": "#9333ea",
    "finance": "#16a34a",
}

MODEL_LABELS = {
    "dummy_mean": "Historical Mean",
    "seasonal_naive": "Seasonal Naive",
    "moving_average": "Moving Average",
    "linear_trend": "Linear Trend",
}

DOMAIN_LABELS = {
    "energy": "能源",
    "traffic": "交通",
    "weather": "天气",
    "economics": "汇率/经济",
    "finance": "金融",
}

DATASET_LABELS = {
    "synthetic_energy": "合成能源",
    "synthetic_traffic": "合成交通",
    "synthetic_weather": "合成天气",
    "synthetic_exchange": "合成汇率",
    "stock_provided_sample": "现有金融样本",
    "stock_provided": "现有金融数据",
}

METRIC_LABELS = {
    "smape": "SMAPE",
    "mse": "MSE",
    "mae": "MAE",
    "rmse": "RMSE",
    "wape": "WAPE",
    "mase": "MASE",
}


def _label_model(model: object) -> str:
    """Return a Chinese display label for a model."""

    return MODEL_LABELS.get(str(model), str(model))


def _label_domain(domain: object) -> str:
    """Return a Chinese display label for a domain."""

    return DOMAIN_LABELS.get(str(domain), str(domain))


def _label_dataset(dataset: object) -> str:
    """Return a Chinese display label for a dataset."""

    return DATASET_LABELS.get(str(dataset), str(dataset))


def _scale(value: float, min_value: float, max_value: float, width: int) -> float:
    """Scale a numeric value into a pixel coordinate."""

    if max_value == min_value:
        return width / 2
    return (value - min_value) / (max_value - min_value) * width


def _bar_chart(metrics: pd.DataFrame, metric: str) -> str:
    """Render an SVG grouped bar chart for one metric."""

    grouped = (
        metrics.groupby("model", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=True)
        .reset_index(drop=True)
    )
    width = 760
    row_h = 44
    left = 170
    right = 40
    chart_w = width - left - right
    height = 70 + row_h * len(grouped)
    max_value = float(grouped[metric].max())
    rows = []
    for idx, row in grouped.iterrows():
        y = 46 + idx * row_h
        bar_w = _scale(float(row[metric]), 0.0, max_value, chart_w)
        color = PALETTE.get(str(row["model"]), "#334155")
        rows.append(
            f'<text x="20" y="{y + 21}" class="axis">{escape(_label_model(row["model"]))}</text>'
            f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="26" rx="4" fill="{color}">'
            f"</rect>"
            f'<text x="{left + bar_w + 8:.1f}" y="{y + 19}" class="value">'
            f"{float(row[metric]):.4f}</text>"
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart">'
        f'<text x="20" y="24" class="title">各模型平均{escape(METRIC_LABELS.get(metric, metric.upper()))}</text>'
        + "".join(rows)
        + "</svg>"
    )


def _heatmap(metrics: pd.DataFrame, metric: str) -> str:
    """Render a model-by-domain heatmap."""

    pivot = metrics.pivot_table(index="model", columns="domain", values=metric, aggfunc="mean")
    models = list(pivot.index)
    domains = list(pivot.columns)
    values = pivot.to_numpy(dtype=float)
    min_v = float(pd.DataFrame(values).min().min())
    max_v = float(pd.DataFrame(values).max().max())
    cell_w = 126
    cell_h = 58
    left = 136
    top = 58
    width = left + cell_w * len(domains) + 28
    height = top + cell_h * len(models) + 36
    pieces = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    pieces.append(
        f'<text x="20" y="24" class="title">{escape(METRIC_LABELS.get(metric, metric.upper()))} 领域热力图</text>'
    )
    for c, domain in enumerate(domains):
        pieces.append(
            f'<text x="{left + c * cell_w + cell_w / 2}" y="48" class="axis center">'
            f"{escape(_label_domain(domain))}</text>"
        )
    for r, model in enumerate(models):
        pieces.append(
            f'<text x="20" y="{top + r * cell_h + 34}" class="axis">{escape(_label_model(model))}</text>'
        )
        for c, domain in enumerate(domains):
            value = float(pivot.loc[model, domain])
            intensity = _scale(value, min_v, max_v, 1.0)
            red = int(248 - intensity * 210)
            green = int(250 - intensity * 96)
            blue = int(252 - intensity * 38)
            x = left + c * cell_w
            y = top + r * cell_h
            pieces.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 8}" '
                f'rx="6" fill="rgb({red},{green},{blue})"></rect>'
                f'<text x="{x + (cell_w - 8) / 2}" y="{y + 31}" class="cell">'
                f"{value:.3f}</text>"
            )
    pieces.append("</svg>")
    return "".join(pieces)


def _forecast_gallery(predictions: pd.DataFrame, max_panels: int = 4) -> str:
    """Render compact actual-vs-predicted SVG panels."""

    panels = []
    keys = predictions[["dataset", "unique_id", "model"]].drop_duplicates().head(max_panels)
    for _, key in keys.iterrows():
        panel = predictions[
            (predictions["dataset"] == key["dataset"])
            & (predictions["unique_id"] == key["unique_id"])
            & (predictions["model"] == key["model"])
        ].sort_values("ds")
        if panel.empty:
            continue
        y_values = pd.concat([panel["y"], panel["y_hat"]]).astype(float)
        min_y = float(y_values.min())
        max_y = float(y_values.max())
        width = 360
        height = 190
        pad = 32

        def points(column: str) -> str:
            coords = []
            for idx, value in enumerate(panel[column].astype(float)):
                x = pad + _scale(idx, 0, max(1, len(panel) - 1), width - 2 * pad)
                y = height - pad - _scale(value, min_y, max_y, height - 2 * pad)
                coords.append(f"{x:.1f},{y:.1f}")
            return " ".join(coords)

        color = PALETTE.get(str(key["model"]), "#334155")
        panels.append(
            f'<svg viewBox="0 0 {width} {height}" class="mini">'
            f'<text x="18" y="22" class="mini-title">{escape(_label_dataset(key["dataset"]))} | '
            f'{escape(_label_model(key["model"]))}</text>'
            f'<text x="18" y="40" class="mini-sub">{escape(str(key["unique_id"]))}</text>'
            f'<polyline points="{points("y")}" fill="none" stroke="#111827" stroke-width="2.2"/>'
            f'<polyline points="{points("y_hat")}" fill="none" stroke="{color}" stroke-width="2.2"/>'
            f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" '
            f'stroke="#cbd5e1"/>'
            f"</svg>"
        )
    return '<div class="gallery">' + "".join(panels) + "</div>"


def build_pilot_report(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Build a self-contained HTML pilot report."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = metrics.sort_values("smape").iloc[0]
    domains = "、".join(_label_domain(domain) for domain in sorted(metrics["domain"].unique()))
    display_metrics = metrics.copy()
    display_metrics["model"] = display_metrics["model"].map(_label_model)
    display_metrics["dataset"] = display_metrics["dataset"].map(_label_dataset)
    display_metrics["domain"] = display_metrics["domain"].map(_label_domain)
    display_metrics = display_metrics.rename(
        columns={
            "run_id": "运行ID",
            "model": "模型",
            "dataset": "数据集",
            "domain": "领域",
            "horizon": "预测步长",
            "context_length": "上下文长度",
            "mse": "MSE",
            "mae": "MAE",
            "rmse": "RMSE",
            "smape": "SMAPE",
            "wape": "WAPE",
            "mase": "MASE",
            "train_time_s": "训练时间(秒)",
            "inference_time_s": "推理时间(秒)",
            "gpu_memory_mb": "GPU显存(MB)",
        }
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>TabPFN-TS 初步实验结果</title>
<style>
body {{ margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f8fafc; color: #0f172a; }}
header {{ padding: 34px 44px 22px; background: #0f172a; color: white; }}
h1 {{ margin: 0; font-size: 34px; letter-spacing: 0; }}
.subtitle {{ margin-top: 8px; color: #cbd5e1; max-width: 880px; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; padding: 20px 44px; }}
.stat {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
.stat b {{ display: block; font-size: 24px; margin-top: 6px; }}
section {{ padding: 12px 44px 26px; }}
.panel {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
.chart {{ width: 100%; height: auto; }}
.title {{ font-size: 18px; font-weight: 700; fill: #0f172a; }}
.axis {{ font-size: 13px; fill: #475569; }}
.center {{ text-anchor: middle; }}
.value {{ font-size: 13px; fill: #334155; }}
.cell {{ text-anchor: middle; font-size: 14px; font-weight: 700; fill: #0f172a; }}
.gallery {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 14px; }}
.mini {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; width: 100%; }}
.mini-title {{ font-size: 14px; font-weight: 700; fill: #0f172a; }}
.mini-sub {{ font-size: 11px; fill: #64748b; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 9px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: #475569; }}
</style>
</head>
<body>
<header>
  <h1>TabPFN-TS 初步实验结果</h1>
  <div class="subtitle">CPU-only 初步实验，覆盖跨领域合成 sanity 数据集与现有金融样本。当前页面是后续完整 benchmark 的结果展示模板。</div>
</header>
<div class="grid">
  <div class="stat">数据集数量<b>{metrics["dataset"].nunique()}</b></div>
  <div class="stat">覆盖领域<b>{escape(domains)}</b></div>
  <div class="stat">最佳 SMAPE<b>{escape(_label_model(best["model"]))} | {float(best["smape"]):.3f}</b></div>
</div>
<section>
  <div class="panel">{_bar_chart(metrics, "smape")}</div>
  <div class="panel">{_heatmap(metrics, "smape")}</div>
  <div class="panel"><h2>预测曲线样例</h2>{_forecast_gallery(predictions)}</div>
  <div class="panel"><h2>指标明细表</h2>{display_metrics.round(5).to_html(index=False)}</div>
</section>
</body>
</html>"""
    output.write_text(html, encoding="utf-8")
