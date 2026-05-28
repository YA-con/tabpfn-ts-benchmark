"""Matplotlib figure generation for experiment reports."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from src.reporting.comparison import metric_direction

MODEL_COLORS = {
    "dummy_mean": "#7c3aed",
    "seasonal_naive": "#0f766e",
    "moving_average": "#2563eb",
    "linear_trend": "#dc2626",
    "ridge_ar": "#ea580c",
    "hist_gradient_boosting_ar": "#0ea5e9",
    "lightgbm_ar": "#65a30d",
    "xgboost_ar": "#be123c",
    "tabpfn_ts": "#db2777",
}


def _style() -> None:
    """Apply a compact, report-friendly matplotlib style."""

    preferred_fonts = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next((font for font in preferred_fonts if font in available_fonts), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": font_family,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfdff",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "font.size": 10,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path.name


def plot_metric_bar(metrics: pd.DataFrame, metric: str, out: Path) -> str | None:
    """Average metric by model bar chart."""

    if metrics.empty or metric not in metrics.columns or "model" not in metrics.columns:
        return None
    _style()
    grouped = metrics.groupby("model")[metric].mean(numeric_only=True).dropna()
    if grouped.empty:
        return None
    grouped = grouped.sort_values(ascending=metric_direction(metric) < 0)
    colors = [MODEL_COLORS.get(str(model), "#334155") for model in grouped.index]
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.barh([str(x) for x in grouped.index], grouped.values, color=colors, alpha=0.88)
    ax.invert_yaxis()
    ax.set_title(f"模型平均 {metric.upper()} 对比")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("模型")
    ax.grid(axis="x")
    for idx, value in enumerate(grouped.values):
        ax.text(value, idx, f" {value:.4g}", va="center", color="#334155")
    return _save(fig, out / f"metric_bar_{metric}.svg")


def plot_dataset_heatmap(metrics: pd.DataFrame, metric: str, out: Path) -> str | None:
    """Dataset by model heatmap for one metric."""

    if metrics.empty or metric not in metrics.columns or not {"dataset", "model"}.issubset(metrics.columns):
        return None
    pivot = metrics.pivot_table(index="model", columns="dataset", values=metric, aggfunc="mean")
    if pivot.empty:
        return None
    _style()
    fig_w = max(8.5, 1.05 * len(pivot.columns) + 3.5)
    fig_h = max(4.8, 0.42 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.values, cmap="magma_r" if metric_direction(metric) < 0 else "viridis")
    ax.set_title(f"{metric.upper()} 数据集-模型热力图")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            value = pivot.iloc[r, c]
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", color="#0f172a", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label(metric.upper())
    return _save(fig, out / f"dataset_heatmap_{metric}.svg")


def plot_radar(metrics: pd.DataFrame, out: Path) -> str | None:
    """Multi-metric normalized radar chart."""

    metric_cols = [col for col in ["smape", "mae", "rmse", "wape", "mase"] if col in metrics.columns]
    if metrics.empty or len(metric_cols) < 3 or "model" not in metrics.columns:
        return None
    grouped = metrics.groupby("model")[metric_cols].mean(numeric_only=True).dropna(how="all")
    if grouped.empty:
        return None
    angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]
    _style()
    fig = plt.figure(figsize=(7.6, 6.8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_title("多指标综合雷达图")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1], [col.upper() for col in metric_cols])
    ax.set_ylim(0, 1)
    for model, row in grouped.iterrows():
        scores = []
        for col in metric_cols:
            values = grouped[col].dropna()
            lo, hi = float(values.min()), float(values.max())
            if hi == lo:
                score = 1.0
            else:
                scaled = (float(row[col]) - lo) / (hi - lo)
                score = 1 - scaled if metric_direction(col) < 0 else scaled
            scores.append(score)
        scores += scores[:1]
        ax.plot(angles, scores, label=str(model), color=MODEL_COLORS.get(str(model), None), linewidth=1.8)
        ax.fill(angles, scores, color=MODEL_COLORS.get(str(model), "#334155"), alpha=0.06)
    ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=8)
    return _save(fig, out / "model_radar.svg")


def plot_actual_vs_predicted(predictions: pd.DataFrame, out: Path, max_points: int = 1200) -> str | None:
    """Actual-vs-predicted scatter with ideal line."""

    required = {"model", "y", "y_hat"}
    if predictions.empty or not required.issubset(predictions.columns):
        return None
    frame = predictions[list(required)].dropna().copy()
    if frame.empty:
        return None
    if len(frame) > max_points:
        frame = frame.sample(max_points, random_state=42)
    lo = float(pd.concat([frame["y"], frame["y_hat"]]).min())
    hi = float(pd.concat([frame["y"], frame["y_hat"]]).max())
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        return None
    _style()
    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    for model, group in frame.groupby("model"):
        ax.scatter(
            group["y"],
            group["y_hat"],
            s=14,
            alpha=0.32,
            label=str(model),
            color=MODEL_COLORS.get(str(model), None),
            edgecolors="none",
        )
    ax.plot([lo, hi], [lo, hi], "--", color="#0f172a", linewidth=1.4, label="理想预测线")
    ax.set_title("真实值 vs 预测值散点图")
    ax.set_xlabel("真实值")
    ax.set_ylabel("预测值")
    ax.grid(True)
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out / "actual_vs_predicted.svg")


def plot_residual_boxplot(predictions: pd.DataFrame, out: Path) -> str | None:
    """Absolute residual boxplot by model."""

    required = {"model", "y", "y_hat"}
    if predictions.empty or not required.issubset(predictions.columns):
        return None
    frame = predictions[list(required)].dropna().copy()
    if frame.empty:
        return None
    frame["absolute_error"] = (frame["y"] - frame["y_hat"]).abs()
    order = frame.groupby("model")["absolute_error"].median().sort_values().index.tolist()
    data = [frame.loc[frame["model"] == model, "absolute_error"].to_numpy() for model in order]
    _style()
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    parts = ax.boxplot(data, patch_artist=True, tick_labels=order, showfliers=False)
    for patch, model in zip(parts["boxes"], order, strict=True):
        patch.set_facecolor(MODEL_COLORS.get(str(model), "#64748b"))
        patch.set_alpha(0.28)
        patch.set_edgecolor(MODEL_COLORS.get(str(model), "#334155"))
    ax.set_title("绝对残差分布箱线图")
    ax.set_ylabel("|真实值 - 预测值|")
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y")
    return _save(fig, out / "residual_boxplot.svg")


def plot_runtime(metrics: pd.DataFrame, out: Path) -> str | None:
    """Runtime comparison by model."""

    cols = [col for col in ["train_time_s", "inference_time_s"] if col in metrics.columns]
    if metrics.empty or not cols or "model" not in metrics.columns:
        return None
    grouped = metrics.groupby("model")[cols].mean(numeric_only=True).fillna(0)
    if grouped.empty:
        return None
    grouped = grouped.sort_values(cols[-1])
    _style()
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    bottom = np.zeros(len(grouped))
    colors = {"train_time_s": "#38bdf8", "inference_time_s": "#db2777"}
    for col in cols:
        ax.bar(grouped.index.astype(str), grouped[col].values, bottom=bottom, label=col, color=colors[col])
        bottom += grouped[col].values
    ax.set_title("平均训练/推理耗时")
    ax.set_ylabel("秒")
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y")
    ax.legend()
    return _save(fig, out / "runtime.svg")


def plot_baseline_improvement(comparison: pd.DataFrame, out: Path) -> str | None:
    """Baseline improvement chart based on computed comparison rows."""

    required = {"baseline", "absolute_difference", "relative_improvement_pct", "metric"}
    if comparison.empty or not required.issubset(comparison.columns):
        return None
    frame = comparison.dropna(subset=["absolute_difference", "relative_improvement_pct"]).copy()
    if frame.empty:
        return None
    _style()
    labels = frame["baseline"].astype(str).tolist()
    absolute = frame["absolute_difference"].astype(float).to_numpy()
    relative = frame["relative_improvement_pct"].astype(float).to_numpy()
    y = np.arange(len(frame))
    fig, ax1 = plt.subplots(figsize=(9.6, max(3.8, 0.55 * len(frame) + 2.8)))
    colors = ["#16a34a" if value >= 0 else "#dc2626" for value in absolute]
    ax1.barh(y, absolute, color=colors, alpha=0.82, label="Absolute improvement")
    ax1.axvline(0, color="#0f172a", linewidth=1.1)
    ax1.set_yticks(y, labels)
    ax1.set_xlabel("Absolute improvement")
    ax1.set_title(f"Baseline 改变量对比（{frame['metric'].iloc[0].upper()}）")
    ax1.grid(axis="x")
    ax2 = ax1.twiny()
    ax2.plot(relative, y, color="#0ea5e9", marker="o", linewidth=2.0, label="Relative improvement (%)")
    ax2.set_xlabel("Relative improvement (%)")
    for idx, (abs_value, rel_value) in enumerate(zip(absolute, relative, strict=True)):
        ax1.text(
            abs_value,
            idx,
            f" {abs_value:+.3g} / {rel_value:+.2f}%",
            va="center",
            color="#334155",
        )
    return _save(fig, out / "baseline_improvement.svg")


def generate_matplotlib_figures(
    metrics: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    assets_dir: Path,
    primary_metric: str = "smape",
    comparison: pd.DataFrame | None = None,
) -> list[dict[str, str]]:
    """Generate available figures and return metadata for HTML rendering."""

    figures: list[dict[str, str]] = []
    if comparison is not None and not comparison.empty:
        filename = plot_baseline_improvement(comparison, assets_dir)
        if filename:
            figures.append({"title": "Baseline 改变量图", "file": filename})
    if metrics is not None and not metrics.empty:
        for title, filename in [
            ("Baseline/模型平均指标柱状图", plot_metric_bar(metrics, primary_metric, assets_dir)),
            ("数据集-模型热力图", plot_dataset_heatmap(metrics, primary_metric, assets_dir)),
            ("多指标雷达图", plot_radar(metrics, assets_dir)),
            ("运行时间图", plot_runtime(metrics, assets_dir)),
        ]:
            if filename:
                figures.append({"title": title, "file": filename})
    if predictions is not None and not predictions.empty:
        for title, filename in [
            ("真实值-预测值散点图", plot_actual_vs_predicted(predictions, assets_dir)),
            ("残差箱线图", plot_residual_boxplot(predictions, assets_dir)),
        ]:
            if filename:
                figures.append({"title": title, "file": filename})
    return figures
