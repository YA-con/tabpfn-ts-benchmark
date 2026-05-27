"""HTML report generation for pilot benchmark results."""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

import pandas as pd


PALETTE = {
    "dummy_mean": "#7c3aed",
    "seasonal_naive": "#0f766e",
    "moving_average": "#2563eb",
    "linear_trend": "#dc2626",
    "ridge_ar": "#ea580c",
    "hist_gradient_boosting_ar": "#0ea5e9",
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
    "ridge_ar": "Ridge AR",
    "hist_gradient_boosting_ar": "Gradient Boosting AR",
}

DOMAIN_LABELS = {
    "energy": "能源",
    "traffic": "交通",
    "weather": "天气",
    "economics": "汇率/经济",
    "finance": "金融",
}

DATASET_LABELS = {
    "etth1": "ETTh1",
    "electricity": "Electricity",
    "exchange_rate": "Exchange Rate",
    "traffic": "Traffic",
    "weather": "Weather",
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


def _model_order(metrics: pd.DataFrame) -> list[str]:
    """Order models by average SMAPE."""

    return [
        str(model)
        for model in metrics.groupby("model")["smape"].mean().sort_values(ascending=True).index
    ]


def _model_legend(models: list[str], x: float, y: float, columns: int = 3) -> str:
    """Render a compact model color legend."""

    pieces = []
    col_w = 190
    row_h = 20
    for idx, model in enumerate(models):
        col = idx % columns
        row = idx // columns
        lx = x + col * col_w
        ly = y + row * row_h
        color = PALETTE.get(model, "#334155")
        pieces.append(f'<circle cx="{lx}" cy="{ly}" r="5" fill="{color}"/>')
        pieces.append(f'<text x="{lx + 12}" y="{ly + 4}" class="axis">{escape(_label_model(model))}</text>')
    return "".join(pieces)


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
    legend_y = top + cell_h * len(models) + 22
    height = legend_y + 46
    pieces = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    pieces.append(
        f'<text x="20" y="24" class="title">{escape(METRIC_LABELS.get(metric, metric.upper()))} 领域热力图</text>'
    )
    pieces.append('<text x="20" y="44" class="axis">颜色越深表示误差越高</text>')
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
    legend_x = left
    legend_w = min(260, cell_w * max(1, len(domains)) - 8)
    steps = 10
    step_w = legend_w / steps
    for idx in range(steps):
        intensity = idx / max(1, steps - 1)
        red = int(248 - intensity * 210)
        green = int(250 - intensity * 96)
        blue = int(252 - intensity * 38)
        pieces.append(
            f'<rect x="{legend_x + idx * step_w:.1f}" y="{legend_y}" width="{step_w + 0.5:.1f}" '
            f'height="12" fill="rgb({red},{green},{blue})"></rect>'
        )
    pieces.append(f'<text x="{legend_x}" y="{legend_y + 30}" class="axis">低误差 {min_v:.3f}</text>')
    pieces.append(
        f'<text x="{legend_x + legend_w}" y="{legend_y + 30}" class="axis end">高误差 {max_v:.3f}</text>'
    )
    pieces.append("</svg>")
    return "".join(pieces)


def _rank_bump_chart(metrics: pd.DataFrame) -> str:
    """Render a model rank bump chart across datasets."""

    pivot = metrics.pivot_table(index="model", columns="dataset", values="smape", aggfunc="mean")
    datasets = list(pivot.columns)
    ranks = pivot.rank(axis=0, method="min", ascending=True)
    models = list(ranks.index)
    width = 880
    height = 300
    left = 110
    right = 44
    top = 58
    bottom = 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_rank = max(1, len(models))

    def x_pos(idx: int) -> float:
        return left + _scale(idx, 0, max(1, len(datasets) - 1), plot_w)

    def y_pos(rank: float) -> float:
        return top + _scale(rank, 1, max_rank, plot_h)

    pieces = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    pieces.append('<text x="20" y="25" class="title">模型排名变化图（按数据集 SMAPE 排名）</text>')
    pieces.append(f'<text x="{width - 104}" y="25" class="axis">1 = 最好</text>')
    for idx, dataset in enumerate(datasets):
        x = x_pos(idx)
        pieces.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="#e2e8f0"/>')
        pieces.append(
            f'<text x="{x:.1f}" y="{height - 18}" class="axis center">{escape(_label_dataset(dataset))}</text>'
        )
    for model in models:
        coords = []
        for idx, dataset in enumerate(datasets):
            coords.append(f"{x_pos(idx):.1f},{y_pos(float(ranks.loc[model, dataset])):.1f}")
        color = PALETTE.get(str(model), "#334155")
        pieces.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2.8"/>')
        first_rank = float(ranks.loc[model, datasets[0]])
        last_rank = float(ranks.loc[model, datasets[-1]])
        pieces.append(
            f'<text x="20" y="{y_pos(first_rank) + 4:.1f}" class="axis">{escape(_label_model(model))}</text>'
        )
        pieces.append(
            f'<text x="{width - 34}" y="{y_pos(last_rank) + 4:.1f}" class="axis">{int(last_rank)}</text>'
        )
        for idx, dataset in enumerate(datasets):
            rank = float(ranks.loc[model, dataset])
            pieces.append(
                f'<circle cx="{x_pos(idx):.1f}" cy="{y_pos(rank):.1f}" r="4.2" fill="{color}"/>'
            )
    pieces.append("</svg>")
    return "".join(pieces)


def _error_scatter(metrics: pd.DataFrame) -> str:
    """Render MAE-vs-SMAPE scatter with domain colors and model labels."""

    width = 780
    height = 420
    left = 64
    right = 36
    top = 48
    bottom = 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x = float(metrics["mae"].min())
    max_x = float(metrics["mae"].max())
    min_y = float(metrics["smape"].min())
    max_y = float(metrics["smape"].max())
    pieces = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    pieces.append('<text x="20" y="25" class="title">误差空间图（MAE × SMAPE）</text>')
    pieces.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#94a3b8"/>')
    pieces.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#94a3b8"/>')
    pieces.append(f'<text x="{width / 2}" y="{height - 18}" class="axis center">MAE</text>')
    pieces.append(f'<text x="18" y="{top - 10}" class="axis">SMAPE</text>')
    for _, row in metrics.iterrows():
        x = left + _scale(float(row["mae"]), min_x, max_x, plot_w)
        y = height - bottom - _scale(float(row["smape"]), min_y, max_y, plot_h)
        color = PALETTE.get(str(row["domain"]), "#334155")
        pieces.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" opacity="0.76">'
            f'<title>{escape(_label_dataset(row["dataset"]))} | {escape(_label_model(row["model"]))} '
            f'SMAPE={float(row["smape"]):.3f}, MAE={float(row["mae"]):.3f}</title></circle>'
        )
    legend_x = left + 12
    legend_y = top + 16
    pieces.append(f'<text x="{legend_x - 2}" y="{legend_y - 14}" class="axis">领域图例</text>')
    for idx, domain in enumerate(sorted(metrics["domain"].unique())):
        y = legend_y + idx * 20
        color = PALETTE.get(str(domain), "#334155")
        pieces.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}"/>')
        pieces.append(f'<text x="{legend_x + 12}" y="{y + 4}" class="axis">{escape(_label_domain(domain))}</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def _dataset_cards(metrics: pd.DataFrame) -> str:
    """Render compact dataset cards with best model and model spread."""

    cards = []
    for dataset, group in metrics.groupby("dataset"):
        best = group.sort_values("smape").iloc[0]
        worst = group.sort_values("smape").iloc[-1]
        spread = float(worst["smape"] - best["smape"])
        cards.append(
            '<div class="dataset-card">'
            f'<div class="dataset-name">{escape(_label_dataset(dataset))}</div>'
            f'<div class="dataset-domain">{escape(_label_domain(best["domain"]))}</div>'
            f'<div class="dataset-best">{escape(_label_model(best["model"]))}</div>'
            f'<div class="dataset-metric">SMAPE {float(best["smape"]):.3f}</div>'
            f'<div class="dataset-spread">模型差距 {spread:.3f}</div>'
            "</div>"
        )
    return '<div class="cards">' + "".join(cards) + "</div>"


def _winner_table(metrics: pd.DataFrame) -> str:
    """Render a compact winner table by dataset."""

    winners = (
        metrics.sort_values("smape")
        .groupby("dataset", as_index=False)
        .first()[["dataset", "domain", "model", "smape", "mae", "rmse"]]
        .sort_values(["domain", "dataset"])
    )
    rows = []
    for _, row in winners.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(_label_dataset(row['dataset']))}</td>"
            f"<td>{escape(_label_domain(row['domain']))}</td>"
            f"<td>{escape(_label_model(row['model']))}</td>"
            f"<td>{float(row['smape']):.3f}</td>"
            f"<td>{float(row['mae']):.3f}</td>"
            f"<td>{float(row['rmse']):.3f}</td>"
            "</tr>"
        )
    return (
        '<table><thead><tr><th>数据集</th><th>领域</th><th>最佳模型</th>'
        "<th>SMAPE</th><th>MAE</th><th>RMSE</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _forecast_gallery(predictions: pd.DataFrame, metrics: pd.DataFrame, max_panels: int = 6) -> str:
    """Render compact actual-vs-predicted SVG panels."""

    panels = []
    best_models = metrics.sort_values("smape").groupby("dataset", as_index=False).first()
    keys = []
    for _, best in best_models.iterrows():
        subset = predictions[
            (predictions["dataset"] == best["dataset"]) & (predictions["model"] == best["model"])
        ]
        if subset.empty:
            continue
        first_id = subset["unique_id"].iloc[0]
        keys.append({"dataset": best["dataset"], "unique_id": first_id, "model": best["model"]})
    keys_df = pd.DataFrame(keys).head(max_panels)
    if keys_df.empty:
        return '<div class="gallery"></div>'
    for _, key in keys_df.iterrows():
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
        height = 214
        pad = 32
        chart_bottom = height - 56
        legend_y = height - 18

        def points(column: str) -> str:
            coords = []
            for idx, value in enumerate(panel[column].astype(float)):
                x = pad + _scale(idx, 0, max(1, len(panel) - 1), width - 2 * pad)
                y = chart_bottom - _scale(value, min_y, max_y, chart_bottom - pad)
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
            f'<line x1="{pad}" y1="{chart_bottom}" x2="{width - pad}" y2="{chart_bottom}" '
            f'stroke="#cbd5e1"/>'
            f'<line x1="18" y1="{legend_y}" x2="48" y2="{legend_y}" stroke="#111827" stroke-width="2.2"/>'
            f'<text x="54" y="{legend_y + 4}" class="mini-sub">真实值</text>'
            f'<line x1="112" y1="{legend_y}" x2="142" y2="{legend_y}" stroke="{color}" stroke-width="2.2"/>'
            f'<text x="148" y="{legend_y + 4}" class="mini-sub">预测值</text>'
            f"</svg>"
        )
    return '<div class="gallery">' + "".join(panels) + "</div>"


def _actual_predicted_fit(predictions: pd.DataFrame, metrics: pd.DataFrame) -> str:
    """Render actual-vs-predicted scatter with one fitted line per model."""

    frame = predictions[["model", "y", "y_hat"]].dropna().copy()
    if frame.empty:
        return ""
    if len(frame) > 900:
        frame = frame.sample(n=900, random_state=42)
    width = 820
    height = 480
    left = 70
    right = 32
    top = 58
    bottom = 104
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_v = float(pd.concat([frame["y"], frame["y_hat"]]).min())
    max_v = float(pd.concat([frame["y"], frame["y_hat"]]).max())
    models = _model_order(metrics)
    pieces = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    pieces.append('<text x="20" y="25" class="title">真实值-预测值散点图（含拟合线）</text>')
    pieces.append('<text x="20" y="45" class="axis">灰色虚线为理想预测线；彩色线为模型拟合线</text>')
    pieces.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#94a3b8"/>')
    pieces.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#94a3b8"/>')
    pieces.append(f'<text x="{width / 2}" y="{height - 64}" class="axis center">真实值</text>')
    pieces.append(f'<text x="18" y="{top - 10}" class="axis">预测值</text>')
    pieces.append(
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{top}" '
        f'stroke="#64748b" stroke-dasharray="5 5"/>'
    )
    for model in models:
        subset = frame[frame["model"] == model]
        if subset.empty:
            continue
        color = PALETTE.get(model, "#334155")
        for _, row in subset.iterrows():
            x = left + _scale(float(row["y"]), min_v, max_v, plot_w)
            y = height - bottom - _scale(float(row["y_hat"]), min_v, max_v, plot_h)
            pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="{color}" opacity="0.26"/>')
        if subset["y"].nunique() > 1:
            slope = float(subset["y"].cov(subset["y_hat"]) / subset["y"].var())
            intercept = float(subset["y_hat"].mean() - slope * subset["y"].mean())
            y1 = intercept + slope * min_v
            y2 = intercept + slope * max_v
            x1p = left
            x2p = width - right
            y1p = height - bottom - _scale(y1, min_v, max_v, plot_h)
            y2p = height - bottom - _scale(y2, min_v, max_v, plot_h)
            pieces.append(
                f'<line x1="{x1p}" y1="{y1p:.1f}" x2="{x2p}" y2="{y2p:.1f}" '
                f'stroke="{color}" stroke-width="2.2"/>'
            )
    pieces.append(_model_legend(models, left, height - 38, columns=3))
    pieces.append("</svg>")
    return "".join(pieces)


def _residual_scatter(predictions: pd.DataFrame, metrics: pd.DataFrame) -> str:
    """Render residuals against predicted values."""

    frame = predictions[["model", "y", "y_hat"]].dropna().copy()
    if frame.empty:
        return ""
    frame["residual"] = frame["y"] - frame["y_hat"]
    if len(frame) > 900:
        frame = frame.sample(n=900, random_state=7)
    width = 820
    height = 430
    left = 70
    right = 32
    top = 54
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x = float(frame["y_hat"].min())
    max_x = float(frame["y_hat"].max())
    max_abs = float(frame["residual"].abs().max())
    models = _model_order(metrics)
    zero_y = top + plot_h / 2
    pieces = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    pieces.append('<text x="20" y="25" class="title">残差图（真实值 - 预测值）</text>')
    pieces.append('<text x="20" y="45" class="axis">点越接近零线，预测偏差越小</text>')
    pieces.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#94a3b8"/>')
    pieces.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#94a3b8"/>')
    pieces.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}" stroke="#0f172a" stroke-dasharray="5 5"/>')
    pieces.append(f'<text x="{width / 2}" y="{height - 52}" class="axis center">预测值</text>')
    pieces.append(f'<text x="18" y="{top - 8}" class="axis">残差</text>')
    for _, row in frame.iterrows():
        color = PALETTE.get(str(row["model"]), "#334155")
        x = left + _scale(float(row["y_hat"]), min_x, max_x, plot_w)
        y = zero_y - _scale(float(row["residual"]), -max_abs, max_abs, plot_h) + plot_h / 2
        pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{color}" opacity="0.34"/>')
    pieces.append(_model_legend(models, left, height - 28, columns=3))
    pieces.append("</svg>")
    return "".join(pieces)


def _residual_boxplot(predictions: pd.DataFrame, metrics: pd.DataFrame) -> str:
    """Render model-wise absolute residual boxplots."""

    frame = predictions[["model", "y", "y_hat"]].dropna().copy()
    if frame.empty:
        return ""
    frame["abs_error"] = (frame["y"] - frame["y_hat"]).abs()
    models = _model_order(metrics)
    stats = []
    for model in models:
        values = frame.loc[frame["model"] == model, "abs_error"]
        if values.empty:
            continue
        stats.append(
            {
                "model": model,
                "q05": float(values.quantile(0.05)),
                "q25": float(values.quantile(0.25)),
                "q50": float(values.quantile(0.50)),
                "q75": float(values.quantile(0.75)),
                "q95": float(values.quantile(0.95)),
            }
        )
    if not stats:
        return ""
    width = 840
    row_h = 48
    left = 190
    right = 44
    top = 54
    bottom = 38
    plot_w = width - left - right
    height = top + row_h * len(stats) + bottom
    max_v = max(item["q95"] for item in stats)
    pieces = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    pieces.append('<text x="20" y="25" class="title">绝对残差箱线图</text>')
    pieces.append('<text x="20" y="45" class="axis">箱体为 25%-75%，横线为 5%-95%</text>')
    for idx, item in enumerate(stats):
        y = top + idx * row_h + 18
        color = PALETTE.get(item["model"], "#334155")
        q05 = left + _scale(item["q05"], 0, max_v, plot_w)
        q25 = left + _scale(item["q25"], 0, max_v, plot_w)
        q50 = left + _scale(item["q50"], 0, max_v, plot_w)
        q75 = left + _scale(item["q75"], 0, max_v, plot_w)
        q95 = left + _scale(item["q95"], 0, max_v, plot_w)
        pieces.append(f'<text x="20" y="{y + 5}" class="axis">{escape(_label_model(item["model"]))}</text>')
        pieces.append(f'<line x1="{q05:.1f}" y1="{y}" x2="{q95:.1f}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        pieces.append(f'<line x1="{q05:.1f}" y1="{y - 8}" x2="{q05:.1f}" y2="{y + 8}" stroke="{color}" stroke-width="2"/>')
        pieces.append(f'<line x1="{q95:.1f}" y1="{y - 8}" x2="{q95:.1f}" y2="{y + 8}" stroke="{color}" stroke-width="2"/>')
        pieces.append(f'<rect x="{q25:.1f}" y="{y - 13}" width="{max(q75 - q25, 1):.1f}" height="26" rx="4" fill="{color}" opacity="0.25" stroke="{color}"/>')
        pieces.append(f'<line x1="{q50:.1f}" y1="{y - 15}" x2="{q50:.1f}" y2="{y + 15}" stroke="{color}" stroke-width="3"/>')
        pieces.append(f'<text x="{q50 + 8:.1f}" y="{y + 5}" class="value">{item["q50"]:.3f}</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def _taylor_diagram(predictions: pd.DataFrame, metrics: pd.DataFrame) -> str:
    """Render a compact Taylor-style diagram by model."""

    frame = predictions[["model", "y", "y_hat"]].dropna().copy()
    if frame.empty:
        return ""
    models = _model_order(metrics)
    rows = []
    for model in models:
        subset = frame[frame["model"] == model]
        if len(subset) < 3:
            continue
        true_std = float(subset["y"].std())
        pred_std = float(subset["y_hat"].std())
        if true_std == 0.0 or math.isnan(true_std) or math.isnan(pred_std):
            continue
        corr = float(subset["y"].corr(subset["y_hat"]))
        if math.isnan(corr):
            corr = 0.0
        rows.append({"model": model, "corr": max(-1.0, min(1.0, corr)), "ratio": pred_std / true_std})
    if not rows:
        return ""
    width = 520
    height = 420
    cx = 96
    cy = 334
    radius = 260
    max_ratio = max(1.6, max(row["ratio"] for row in rows) * 1.12)
    pieces = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    pieces.append('<text x="20" y="25" class="title">泰勒图（相关性 × 标准差比）</text>')
    pieces.append('<text x="20" y="45" class="axis">越靠近右侧且半径接近 1，预测形态越接近真实值</text>')
    for ratio in [0.5, 1.0, 1.5]:
        r = radius * ratio / max_ratio
        pieces.append(f'<path d="M {cx:.1f} {cy - r:.1f} A {r:.1f} {r:.1f} 0 0 1 {cx + r:.1f} {cy:.1f}" fill="none" stroke="#e2e8f0"/>')
        pieces.append(f'<text x="{cx + r - 8:.1f}" y="{cy + 18}" class="axis">{ratio:.1f}</text>')
    for corr in [0.0, 0.5, 0.8, 0.95, 1.0]:
        angle = math.acos(corr)
        x = cx + radius * math.cos(angle)
        y = cy - radius * math.sin(angle)
        pieces.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        pieces.append(f'<text x="{x + 4:.1f}" y="{y:.1f}" class="axis">r={corr:.2f}</text>')
    pieces.append(f'<line x1="{cx}" y1="{cy}" x2="{cx + radius}" y2="{cy}" stroke="#94a3b8"/>')
    pieces.append(f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - radius}" stroke="#94a3b8"/>')
    for row in rows:
        angle = math.acos(max(0.0, min(1.0, row["corr"])))
        r = radius * min(row["ratio"], max_ratio) / max_ratio
        x = cx + r * math.cos(angle)
        y = cy - r * math.sin(angle)
        color = PALETTE.get(row["model"], "#334155")
        pieces.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}">'
            f'<title>{escape(_label_model(row["model"]))} corr={row["corr"]:.3f}, std ratio={row["ratio"]:.3f}</title></circle>'
        )
    pieces.append(_model_legend(models, 300, 88, columns=1))
    pieces.append("</svg>")
    return "".join(pieces)


def _radar_chart(metrics: pd.DataFrame) -> str:
    """Render normalized metric radar chart for model profiles."""

    metric_cols = ["smape", "mae", "rmse", "wape", "mase"]
    available = [col for col in metric_cols if col in metrics.columns]
    if len(available) < 3:
        return ""
    grouped = metrics.groupby("model")[available].mean()
    models = _model_order(metrics)
    width = 560
    height = 430
    cx = 210
    cy = 220
    radius = 142
    pieces = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    pieces.append('<text x="20" y="25" class="title">模型综合雷达图</text>')
    pieces.append('<text x="20" y="45" class="axis">所有指标已转成相对得分，越外圈越好</text>')
    for level in [0.25, 0.5, 0.75, 1.0]:
        coords = []
        for idx in range(len(available)):
            angle = -math.pi / 2 + 2 * math.pi * idx / len(available)
            coords.append(f"{cx + radius * level * math.cos(angle):.1f},{cy + radius * level * math.sin(angle):.1f}")
        pieces.append(f'<polygon points="{" ".join(coords)}" fill="none" stroke="#e2e8f0"/>')
    for idx, metric in enumerate(available):
        angle = -math.pi / 2 + 2 * math.pi * idx / len(available)
        x = cx + radius * 1.13 * math.cos(angle)
        y = cy + radius * 1.13 * math.sin(angle)
        pieces.append(f'<line x1="{cx}" y1="{cy}" x2="{cx + radius * math.cos(angle):.1f}" y2="{cy + radius * math.sin(angle):.1f}" stroke="#e2e8f0"/>')
        pieces.append(f'<text x="{x:.1f}" y="{y:.1f}" class="axis center">{escape(METRIC_LABELS.get(metric, metric.upper()))}</text>')
    for model in models:
        if model not in grouped.index:
            continue
        coords = []
        for idx, metric in enumerate(available):
            col = grouped[metric]
            min_v = float(col.min())
            max_v = float(col.max())
            score = 1.0 - _scale(float(grouped.loc[model, metric]), min_v, max_v, 1.0)
            angle = -math.pi / 2 + 2 * math.pi * idx / len(available)
            coords.append(f"{cx + radius * score * math.cos(angle):.1f},{cy + radius * score * math.sin(angle):.1f}")
        color = PALETTE.get(model, "#334155")
        pieces.append(f'<polygon points="{" ".join(coords)}" fill="{color}" opacity="0.10" stroke="{color}" stroke-width="2"/>')
    pieces.append(_model_legend(models, 382, 92, columns=1))
    pieces.append("</svg>")
    return "".join(pieces)


def _three_d_error_bars(metrics: pd.DataFrame) -> str:
    """Render pseudo-3D bars for average SMAPE."""

    grouped = (
        metrics.groupby("model", as_index=False)["smape"]
        .mean()
        .sort_values("smape", ascending=False)
        .reset_index(drop=True)
    )
    width = 880
    height = 420
    left = 62
    base = 342
    bar_w = 72
    gap = 48
    depth = 18
    max_v = float(grouped["smape"].max())
    pieces = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    pieces.append('<text x="20" y="25" class="title">3D 风格平均误差柱图</text>')
    pieces.append('<text x="20" y="45" class="axis">柱越低表示平均 SMAPE 越小</text>')
    pieces.append(f'<line x1="{left - 20}" y1="{base}" x2="{width - 30}" y2="{base}" stroke="#94a3b8"/>')
    for idx, row in grouped.iterrows():
        x = left + idx * (bar_w + gap)
        h = _scale(float(row["smape"]), 0, max_v, 240)
        y = base - h
        color = PALETTE.get(str(row["model"]), "#334155")
        pieces.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}" opacity="0.86"/>')
        pieces.append(
            f'<polygon points="{x},{y:.1f} {x + depth},{y - depth:.1f} {x + bar_w + depth},{y - depth:.1f} '
            f'{x + bar_w},{y:.1f}" fill="{color}" opacity="0.62"/>'
        )
        pieces.append(
            f'<polygon points="{x + bar_w},{y:.1f} {x + bar_w + depth},{y - depth:.1f} '
            f'{x + bar_w + depth},{base - depth} {x + bar_w},{base}" fill="{color}" opacity="0.38"/>'
        )
        pieces.append(f'<text x="{x + bar_w / 2}" y="{y - 24:.1f}" class="value center">{float(row["smape"]):.2f}</text>')
        pieces.append(
            f'<text x="{x + bar_w / 2}" y="{base + 22}" class="axis center">'
            f'{escape(_label_model(row["model"]))}</text>'
        )
    pieces.append("</svg>")
    return "".join(pieces)


def _radial_error_bars(metrics: pd.DataFrame) -> str:
    """Render radial grouped bars with standard-error whiskers."""

    grouped = metrics.groupby(["domain", "model"])["smape"].agg(["mean", "std", "count"]).reset_index()
    if grouped.empty:
        return ""
    domains = sorted(metrics["domain"].unique())
    models = _model_order(metrics)
    width = 720
    height = 720
    cx = width / 2
    cy = 360
    inner = 92
    outer = 276
    max_v = float((grouped["mean"] + grouped["std"].fillna(0.0)).max())
    pieces = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    pieces.append('<text x="24" y="28" class="title">环形分组柱状图（带误差棒）</text>')
    pieces.append('<text x="24" y="48" class="axis">半径表示平均 SMAPE，误差棒为跨数据集标准差</text>')
    total_slots = len(domains) * len(models)
    slot_angle = 2 * math.pi / max(1, total_slots)
    for ring in [0.25, 0.5, 0.75, 1.0]:
        r = inner + (outer - inner) * ring
        pieces.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none" stroke="#e2e8f0"/>')
    for d_idx, domain in enumerate(domains):
        mid_slot = d_idx * len(models) + len(models) / 2
        angle = -math.pi / 2 + mid_slot * slot_angle
        lx = cx + (outer + 34) * math.cos(angle)
        ly = cy + (outer + 34) * math.sin(angle)
        pieces.append(f'<text x="{lx:.1f}" y="{ly:.1f}" class="axis center">{escape(_label_domain(domain))}</text>')
    for d_idx, domain in enumerate(domains):
        for m_idx, model in enumerate(models):
            row = grouped[(grouped["domain"] == domain) & (grouped["model"] == model)]
            if row.empty:
                continue
            mean_v = float(row["mean"].iloc[0])
            std_v = float(row["std"].fillna(0.0).iloc[0])
            angle = -math.pi / 2 + (d_idx * len(models) + m_idx + 0.5) * slot_angle
            r = inner + _scale(mean_v, 0, max_v, outer - inner)
            err_r = inner + _scale(mean_v + std_v, 0, max_v, outer - inner)
            x1 = cx + inner * math.cos(angle)
            y1 = cy + inner * math.sin(angle)
            x2 = cx + r * math.cos(angle)
            y2 = cy + r * math.sin(angle)
            xe = cx + err_r * math.cos(angle)
            ye = cy + err_r * math.sin(angle)
            color = PALETTE.get(model, "#334155")
            pieces.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="7" opacity="0.78"/>')
            pieces.append(f'<line x1="{x2:.1f}" y1="{y2:.1f}" x2="{xe:.1f}" y2="{ye:.1f}" stroke="{color}" stroke-width="1.5"/>')
            cap_angle = angle + math.pi / 2
            cap = 5
            pieces.append(
                f'<line x1="{xe - cap * math.cos(cap_angle):.1f}" y1="{ye - cap * math.sin(cap_angle):.1f}" '
                f'x2="{xe + cap * math.cos(cap_angle):.1f}" y2="{ye + cap * math.sin(cap_angle):.1f}" '
                f'stroke="{color}" stroke-width="1.5"/>'
            )
    pieces.append(_model_legend(models, 110, 652, columns=3))
    pieces.append("</svg>")
    return "".join(pieces)


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
.end {{ text-anchor: end; }}
.value {{ font-size: 13px; fill: #334155; }}
.cell {{ text-anchor: middle; font-size: 14px; font-weight: 700; fill: #0f172a; }}
.gallery {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 14px; }}
.advanced-grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; }}
.advanced-grid .panel {{ margin-bottom: 0; }}
.mini {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; width: 100%; }}
.mini-title {{ font-size: 14px; font-weight: 700; fill: #0f172a; }}
.mini-sub {{ font-size: 11px; fill: #64748b; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
.dataset-card {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; background: #f8fafc; }}
.dataset-name {{ font-weight: 800; font-size: 16px; }}
.dataset-domain {{ color: #64748b; margin-top: 2px; }}
.dataset-best {{ margin-top: 12px; font-weight: 700; }}
.dataset-metric {{ color: #0f766e; margin-top: 4px; font-weight: 700; }}
.dataset-spread {{ color: #64748b; margin-top: 4px; font-size: 12px; }}
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
  <div class="panel"><h2>数据集概览</h2>{_dataset_cards(metrics)}</div>
  <div class="panel">{_bar_chart(metrics, "smape")}</div>
  <div class="panel">{_rank_bump_chart(metrics)}</div>
  <div class="panel">{_heatmap(metrics, "smape")}</div>
  <div class="panel">{_error_scatter(metrics)}</div>
  <div class="panel"><h2>各数据集最佳模型</h2>{_winner_table(metrics)}</div>
  <div class="panel"><h2>预测曲线样例</h2>{_forecast_gallery(predictions, metrics)}</div>
  <h2>高级诊断图</h2>
  <div class="advanced-grid">
    <div class="panel">{_actual_predicted_fit(predictions, metrics)}</div>
    <div class="panel">{_residual_scatter(predictions, metrics)}</div>
    <div class="panel">{_residual_boxplot(predictions, metrics)}</div>
    <div class="panel">{_taylor_diagram(predictions, metrics)}</div>
    <div class="panel">{_radar_chart(metrics)}</div>
    <div class="panel">{_three_d_error_bars(metrics)}</div>
    <div class="panel">{_radial_error_bars(metrics)}</div>
  </div>
  <div class="panel"><h2>指标明细表</h2>{display_metrics.round(5).to_html(index=False)}</div>
</section>
</body>
</html>"""
    output.write_text(html, encoding="utf-8")
