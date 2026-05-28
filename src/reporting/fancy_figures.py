"""Premium Matplotlib figures for demo-grade experiment dashboards."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

from src.reporting.comparison import metric_direction
from src.reporting.style_config import ReportTheme


def _mpl_color(color: str, fallback: str = "#94a3b8") -> str | tuple[float, float, float, float]:
    """Convert a small CSS color subset into Matplotlib-compatible values."""

    text = str(color).strip()
    if text.startswith("rgba(") and text.endswith(")"):
        parts = [part.strip() for part in text[5:-1].split(",")]
        if len(parts) == 4:
            try:
                return (int(parts[0]) / 255, int(parts[1]) / 255, int(parts[2]) / 255, float(parts[3]))
            except ValueError:
                return fallback
    if text.startswith("rgb(") and text.endswith(")"):
        parts = [part.strip() for part in text[4:-1].split(",")]
        if len(parts) == 3:
            try:
                return (int(parts[0]) / 255, int(parts[1]) / 255, int(parts[2]) / 255, 1.0)
            except ValueError:
                return fallback
    try:
        to_rgba(text)
        return text
    except ValueError:
        return fallback


def _style(theme: ReportTheme) -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in theme.font_family if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "figure.facecolor": theme.chart_background,
            "axes.facecolor": theme.chart_background,
            "axes.edgecolor": _mpl_color(theme.border_color),
            "axes.labelcolor": theme.muted_text_color,
            "axes.titlecolor": theme.text_color,
            "axes.titleweight": 800,
            "axes.titlesize": theme.font_sizes["title"],
            "axes.labelsize": theme.font_sizes["label"],
            "xtick.color": theme.muted_text_color,
            "ytick.color": theme.muted_text_color,
            "xtick.labelsize": theme.font_sizes["tick"],
            "ytick.labelsize": theme.font_sizes["tick"],
            "font.size": theme.font_sizes["label"],
            "grid.color": _mpl_color(theme.grid_color),
            "grid.linewidth": 0.9,
            "grid.alpha": 0.85,
            "legend.frameon": True,
            "legend.facecolor": _mpl_color(theme.card_background_alt, theme.chart_background),
            "legend.edgecolor": _mpl_color(theme.border_color),
            "legend.labelcolor": theme.text_color,
            "savefig.bbox": "tight",
            "savefig.facecolor": theme.chart_background,
        }
    )


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path.name


def _palette(theme: ReportTheme, keys: list[str]) -> dict[str, str]:
    return {key: theme.chart_palette[i % len(theme.chart_palette)] for i, key in enumerate(keys)}


def _best_by_model(metrics: pd.DataFrame, metric: str) -> pd.Series:
    grouped = metrics.groupby("model")[metric].mean(numeric_only=True).dropna()
    return grouped.sort_values(ascending=metric_direction(metric) > 0)


def plot_premium_baseline_delta(comparison: pd.DataFrame, out: Path, theme: ReportTheme) -> str | None:
    if comparison.empty or not {"baseline", "absolute_difference", "relative_improvement_pct"}.issubset(comparison.columns):
        return None
    frame = comparison.dropna(subset=["absolute_difference", "relative_improvement_pct"]).copy()
    if frame.empty:
        return None
    _style(theme)
    fig, ax = plt.subplots(figsize=theme.chart_dimensions["compact"])
    y = np.arange(len(frame))
    values = frame["relative_improvement_pct"].astype(float).to_numpy()
    colors = [theme.success_color if value >= 0 else theme.danger_color for value in values]
    ax.barh(y, values, color=colors, alpha=0.86, height=0.48)
    ax.axvline(0, color=theme.muted_text_color, linewidth=1.2)
    ax.set_yticks(y, frame["baseline"].astype(str))
    ax.set_xlabel("相对变化（%）")
    ax.set_title("Baseline Gap / 相对变化")
    ax.grid(axis="x")
    for idx, row in frame.iterrows():
        value = float(row["relative_improvement_pct"])
        absolute = float(row["absolute_difference"])
        ax.text(
            value,
            idx,
            f" {value:+.2f}% | abs {absolute:+.3g}",
            va="center",
            color=theme.text_color,
            fontsize=theme.font_sizes["annotation"],
            fontweight=700,
        )
    ax.text(
        0.01,
        -0.18,
        "正值表示当前 run 相对 baseline 变好；负值表示退步。",
        transform=ax.transAxes,
        color=theme.muted_text_color,
        fontsize=theme.font_sizes["annotation"],
    )
    return _save(fig, out / "premium_baseline_delta.svg")


def plot_metric_lollipop(metrics: pd.DataFrame, metric: str, out: Path, theme: ReportTheme) -> str | None:
    if metrics.empty or metric not in metrics.columns or "model" not in metrics.columns:
        return None
    grouped = _best_by_model(metrics, metric)
    if grouped.empty:
        return None
    _style(theme)
    fig, ax = plt.subplots(figsize=theme.chart_dimensions["medium"])
    labels = grouped.index.astype(str).tolist()
    colors = _palette(theme, labels)
    y = np.arange(len(grouped))
    ax.hlines(y, grouped.min(), grouped.values, color=theme.grid_color, linewidth=5, alpha=0.9)
    ax.scatter(grouped.values, y, s=120, color=[colors[x] for x in labels], edgecolor=theme.chart_background, linewidth=1.5, zorder=3)
    best_idx = 0
    ax.scatter(grouped.values[best_idx], y[best_idx], s=220, facecolor="none", edgecolor=theme.accent_color, linewidth=2.4, zorder=4)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Model Ranking / {metric.upper()} 横向棒棒糖图")
    ax.grid(axis="x")
    for idx, value in enumerate(grouped.values):
        ax.text(value, idx - 0.22, f"{value:.4g}", color=theme.text_color, fontsize=theme.font_sizes["annotation"])
    ax.annotate(
        "当前最佳",
        xy=(grouped.values[best_idx], y[best_idx]),
        xytext=(18, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": theme.accent_color, "lw": 1.2},
        color=theme.accent_color,
        fontsize=theme.font_sizes["annotation"],
        fontweight=800,
    )
    return _save(fig, out / f"premium_metric_lollipop_{metric}.svg")


def plot_dataset_heatmap(metrics: pd.DataFrame, metric: str, out: Path, theme: ReportTheme) -> str | None:
    if metrics.empty or metric not in metrics.columns or not {"dataset", "model"}.issubset(metrics.columns):
        return None
    pivot = metrics.pivot_table(index="model", columns="dataset", values=metric, aggfunc="mean")
    if pivot.empty:
        return None
    _style(theme)
    fig_w = max(theme.chart_dimensions["wide"][0], 1.0 * len(pivot.columns) + 5.8)
    fig_h = max(5.4, 0.44 * len(pivot.index) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = "magma_r" if metric_direction(metric) < 0 else "viridis"
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
    ax.set_title(f"Dataset x Model Heatmap / {metric.upper()}")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=28, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            value = pivot.iloc[r, c]
            if pd.notna(value):
                ax.text(c, r, f"{value:.2f}", ha="center", va="center", color="#ffffff", fontsize=8, fontweight=700)
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label(metric.upper())
    return _save(fig, out / f"premium_heatmap_{metric}.svg")


def plot_radar(metrics: pd.DataFrame, out: Path, theme: ReportTheme) -> str | None:
    metric_cols = [col for col in ["smape", "mae", "rmse", "wape", "mase"] if col in metrics.columns]
    if metrics.empty or len(metric_cols) < 3 or "model" not in metrics.columns:
        return None
    grouped = metrics.groupby("model")[metric_cols].mean(numeric_only=True).dropna(how="all")
    if grouped.empty:
        return None
    _style(theme)
    angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=theme.chart_dimensions["square"])
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(theme.chart_background)
    ax.set_title("综合能力雷达图 / normalized score", pad=20)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1], [col.upper() for col in metric_cols])
    ax.set_ylim(0, 1)
    colors = _palette(theme, grouped.index.astype(str).tolist())
    for model, row in grouped.iterrows():
        scores = []
        for col in metric_cols:
            values = grouped[col].dropna()
            lo, hi = float(values.min()), float(values.max())
            scaled = 1.0 if hi == lo else (float(row[col]) - lo) / (hi - lo)
            scores.append(1 - scaled if metric_direction(col) < 0 else scaled)
        scores += scores[:1]
        ax.plot(angles, scores, label=str(model), color=colors[str(model)], linewidth=2.0)
        ax.fill(angles, scores, color=colors[str(model)], alpha=0.07)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8)
    return _save(fig, out / "premium_radar.svg")


def plot_residual_density(predictions: pd.DataFrame, out: Path, theme: ReportTheme, max_models: int = 5) -> str | None:
    required = {"model", "y", "y_hat"}
    if predictions.empty or not required.issubset(predictions.columns):
        return None
    frame = predictions[list(required)].dropna().copy()
    if frame.empty:
        return None
    frame["residual"] = frame["y"] - frame["y_hat"]
    order = frame.groupby("model")["residual"].apply(lambda s: s.abs().median()).sort_values().head(max_models).index.tolist()
    if not order:
        return None
    _style(theme)
    fig, ax = plt.subplots(figsize=theme.chart_dimensions["medium"])
    colors = _palette(theme, order)
    for model in order:
        values = frame.loc[frame["model"] == model, "residual"].astype(float).to_numpy()
        ax.hist(values, bins=36, density=True, histtype="step", linewidth=2.1, color=colors[str(model)], label=str(model), alpha=0.95)
    ax.axvline(0, color=theme.muted_text_color, linestyle="--", linewidth=1.1)
    ax.set_title("Residual Distribution / 残差分布")
    ax.set_xlabel("真实值 - 预测值")
    ax.set_ylabel("密度")
    ax.grid(axis="y")
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, out / "premium_residual_distribution.svg")


def plot_runtime_performance(metrics: pd.DataFrame, metric: str, out: Path, theme: ReportTheme) -> str | None:
    if metrics.empty or metric not in metrics.columns or "model" not in metrics.columns:
        return None
    time_col = "inference_time_s" if "inference_time_s" in metrics.columns else "train_time_s"
    if time_col not in metrics.columns:
        return None
    grouped = metrics.groupby("model")[[metric, time_col]].mean(numeric_only=True).dropna()
    if grouped.empty:
        return None
    _style(theme)
    fig, ax = plt.subplots(figsize=theme.chart_dimensions["compact"])
    labels = grouped.index.astype(str).tolist()
    colors = _palette(theme, labels)
    for model, row in grouped.iterrows():
        ax.scatter(row[time_col], row[metric], s=140, color=colors[str(model)], edgecolor=theme.chart_background, linewidth=1.4)
        ax.text(row[time_col], row[metric], f"  {model}", color=theme.text_color, fontsize=8, va="center")
    best = grouped[metric].idxmin() if metric_direction(metric) < 0 else grouped[metric].idxmax()
    ax.annotate(
        "best metric",
        xy=(grouped.loc[best, time_col], grouped.loc[best, metric]),
        xytext=(28, 22),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": theme.accent_color},
        color=theme.accent_color,
        fontsize=theme.font_sizes["annotation"],
        fontweight=800,
    )
    ax.set_title("Efficiency Frontier / 效率-性能散点")
    ax.set_xlabel("平均推理时间（秒）")
    ax.set_ylabel(metric.upper())
    ax.grid(True)
    return _save(fig, out / "premium_efficiency_frontier.svg")


def plot_slope_baseline_current(
    metrics: pd.DataFrame,
    baseline: pd.DataFrame | None,
    metric: str,
    out: Path,
    theme: ReportTheme,
) -> str | None:
    if metrics.empty or baseline is None or baseline.empty or metric not in metrics.columns or metric not in baseline.columns:
        return None
    cur = metrics.groupby("model")[metric].mean(numeric_only=True).dropna()
    base = baseline.groupby("model")[metric].mean(numeric_only=True).dropna()
    common = sorted(set(cur.index.astype(str)) & set(base.index.astype(str)))
    if not common:
        return None
    _style(theme)
    fig, ax = plt.subplots(figsize=theme.chart_dimensions["compact"])
    colors = _palette(theme, common)
    x = [0, 1]
    for model in common:
        y = [float(base[model]), float(cur[model])]
        ax.plot(x, y, color=colors[model], linewidth=2.0, marker="o", markersize=6, alpha=0.92)
        ax.text(-0.03, y[0], model, ha="right", va="center", color=theme.muted_text_color, fontsize=8)
        ax.text(1.03, y[1], f"{y[1]:.3g}", ha="left", va="center", color=theme.text_color, fontsize=8)
    ax.set_xticks(x, ["Baseline", "Current"])
    ax.set_ylabel(metric.upper())
    ax.set_title("Slope Chart / Baseline 到 Current 变化")
    ax.grid(axis="y")
    return _save(fig, out / "premium_slope_chart.svg")


def plot_training_curve_demo(out: Path, theme: ReportTheme) -> str:
    """Create a clearly labelled visual placeholder when real training curves are absent."""

    _style(theme)
    steps = np.arange(1, 61)
    train = 1.08 * np.exp(-steps / 18) + 0.145 + 0.018 * np.sin(steps / 3)
    val = 1.02 * np.exp(-steps / 21) + 0.18 + 0.024 * np.sin(steps / 4 + 0.6)
    fig, ax = plt.subplots(figsize=theme.chart_dimensions["medium"])
    ax.plot(steps, train, color=theme.primary_color, linewidth=2.5, label="train loss (demo placeholder)")
    ax.plot(steps, val, color=theme.accent_color, linewidth=2.5, label="validation loss (demo placeholder)")
    ax.fill_between(steps, train - 0.025, train + 0.025, color=theme.primary_color, alpha=0.10)
    ax.fill_between(steps, val - 0.03, val + 0.03, color=theme.accent_color, alpha=0.10)
    ax.annotate(
        "DEMO ONLY\n未发现真实训练日志",
        xy=(44, val[43]),
        xytext=(32, val[43] + 0.26),
        arrowprops={"arrowstyle": "->", "color": theme.warning_color},
        color=theme.warning_color,
        fontsize=theme.font_sizes["annotation"],
        fontweight=900,
        ha="center",
    )
    ax.set_title("Training Curves / 训练曲线占位示例")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True)
    ax.legend()
    return _save(fig, out / "premium_training_curve_demo.svg")


def generate_fancy_figures(
    metrics: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    baseline_metrics: pd.DataFrame | None,
    comparison: pd.DataFrame,
    assets_dir: Path,
    theme: ReportTheme,
    primary_metric: str = "smape",
) -> tuple[list[dict[str, str]], list[str]]:
    """Generate premium charts and return figure metadata plus warnings."""

    figures: list[dict[str, str]] = []
    warnings: list[str] = []
    if metrics is None or metrics.empty:
        return figures, ["缺少 metrics，无法生成 fancy 图表。"]
    chart_calls = [
        ("Baseline 相对变化", plot_premium_baseline_delta(comparison, assets_dir, theme)),
        ("指标排名棒棒糖图", plot_metric_lollipop(metrics, primary_metric, assets_dir, theme)),
        ("数据集-模型热力图", plot_dataset_heatmap(metrics, primary_metric, assets_dir, theme)),
        ("多指标雷达图", plot_radar(metrics, assets_dir, theme)),
        ("效率-性能散点图", plot_runtime_performance(metrics, primary_metric, assets_dir, theme)),
        ("Baseline 到 Current 斜率图", plot_slope_baseline_current(metrics, baseline_metrics, primary_metric, assets_dir, theme)),
    ]
    if predictions is not None and not predictions.empty:
        chart_calls.append(("残差分布图", plot_residual_density(predictions, assets_dir, theme)))
    else:
        warnings.append("缺少 predictions，残差分布图不可用。")
    for title, filename in chart_calls:
        if filename:
            figures.append({"title": title, "file": filename})
    figures.append({"title": "训练曲线（Demo 占位）", "file": plot_training_curve_demo(assets_dir, theme), "demo": "true"})
    warnings.append("未发现真实 loss/accuracy 训练日志；训练曲线仅为视觉 demo 占位，不参与实验结论。")
    return figures, warnings
