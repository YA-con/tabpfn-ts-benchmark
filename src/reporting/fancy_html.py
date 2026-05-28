"""Fancy dashboard HTML renderer for experiment visualization demos."""

from __future__ import annotations

import json
import math
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from src.reporting.comparison import metric_direction
from src.reporting.style_config import ReportTheme


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _table(df: pd.DataFrame | None, empty: str = "暂无数据") -> str:
    if df is None or df.empty:
        return f'<div class="empty-state">{escape(empty)}</div>'
    display = df.copy()
    for col in display.select_dtypes(include="number").columns:
        display[col] = display[col].map(_fmt)
    display = display.fillna("N/A")
    return display.to_html(index=False, classes="data-table", border=0, escape=True, na_rep="N/A")


def _best_current(metrics: pd.DataFrame | None, primary_metric: str) -> dict[str, Any]:
    if metrics is None or metrics.empty or "model" not in metrics.columns or primary_metric not in metrics.columns:
        return {"model": "N/A", "value": None}
    grouped = metrics.groupby("model")[primary_metric].mean(numeric_only=True).dropna()
    if grouped.empty:
        return {"model": "N/A", "value": None}
    model = grouped.idxmin() if metric_direction(primary_metric) < 0 else grouped.idxmax()
    return {"model": str(model), "value": float(grouped[model])}


def _runtime(metrics: pd.DataFrame | None) -> str:
    if metrics is None or metrics.empty:
        return "N/A"
    cols = [col for col in ["train_time_s", "inference_time_s"] if col in metrics.columns]
    if not cols:
        return "N/A"
    total = metrics[cols].sum(numeric_only=True).sum()
    return f"{float(total):.3f}s" if pd.notna(total) else "N/A"


def _metric_cards(metrics: pd.DataFrame | None, comparison: pd.DataFrame, primary_metric: str) -> str:
    if metrics is None or metrics.empty or "model" not in metrics.columns:
        return '<div class="empty-state">没有可用于 KPI 卡片的 metrics。</div>'
    numeric = [
        col
        for col in ["smape", "mae", "rmse", "wape", "mase", "train_time_s", "inference_time_s", "gpu_memory_mb"]
        if col in metrics.columns
    ]
    cards = []
    comp = comparison.iloc[0].to_dict() if not comparison.empty else {}
    for idx, metric in enumerate(numeric[:8]):
        grouped = metrics.groupby("model")[metric].mean(numeric_only=True).dropna()
        if grouped.empty:
            value = baseline = absolute = relative = None
            model = "N/A"
        else:
            model = grouped.idxmin() if metric_direction(metric) < 0 else grouped.idxmax()
            value = float(grouped[model])
            if metric == comp.get("metric"):
                baseline = comp.get("baseline_value")
                absolute = comp.get("absolute_difference")
                relative = comp.get("relative_improvement_pct")
            else:
                baseline = absolute = relative = None
        state = "primary" if metric == primary_metric else "standard"
        delta_class = "neutral"
        if relative is not None and pd.notna(relative):
            delta_class = "good" if float(relative) >= 0 else "bad"
        trend = "↑" if delta_class == "good" else "↓" if delta_class == "bad" else "·"
        cards.append(
            f"""
<article class="kpi-card {state}">
  <div class="kpi-top"><span>{escape(metric.upper())}</span><b>{escape(model)}</b></div>
  <div class="kpi-value" data-count="{escape(_fmt(value, 6))}">{escape(_fmt(value))}</div>
  <div class="kpi-grid">
    <span>Baseline</span><strong>{escape(_fmt(baseline))}</strong>
    <span>Absolute</span><strong>{escape(_fmt(absolute))}</strong>
    <span>Relative</span><strong class="pill {delta_class}">{trend} {escape(_fmt(relative, 3))}{'%' if relative is not None and pd.notna(relative) else ''}</strong>
  </div>
</article>"""
        )
    return "".join(cards)


def _insights(
    metrics: pd.DataFrame | None,
    comparison: pd.DataFrame,
    warnings: list[str],
    primary_metric: str,
) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []
    best = _best_current(metrics, primary_metric)
    if best["value"] is not None:
        insights.append(
            {
                "label": "Best Model",
                "title": f"{best['model']} 在 {primary_metric.upper()} 上当前最优",
                "body": f"平均 {primary_metric.upper()} = {_fmt(best['value'])}，由真实 forecast_metrics.csv 聚合计算。",
                "tone": "good",
            }
        )
    if not comparison.empty and pd.notna(comparison.iloc[0].get("relative_improvement_pct")):
        row = comparison.iloc[0]
        rel = float(row["relative_improvement_pct"])
        tone = "good" if rel >= 0 else "bad"
        word = "提升" if rel >= 0 else "下降"
        insights.append(
            {
                "label": "Baseline Gap",
                "title": f"相对 {row['baseline']} {word} {abs(rel):.2f}%",
                "body": f"absolute difference = {float(row['absolute_difference']):+.4g}，正负方向按 {primary_metric.upper()} 的优化方向计算。",
                "tone": tone,
            }
        )
    if metrics is not None and "dataset" in metrics.columns and primary_metric in metrics.columns:
        spread = metrics.groupby("dataset")[primary_metric].mean(numeric_only=True).dropna()
        if len(spread) >= 2:
            hardest = spread.idxmax() if metric_direction(primary_metric) < 0 else spread.idxmin()
            easiest = spread.idxmin() if metric_direction(primary_metric) < 0 else spread.idxmax()
            insights.append(
                {
                    "label": "Dataset Spread",
                    "title": f"{hardest} 与 {easiest} 差异最大",
                    "body": f"跨数据集平均 {primary_metric.upper()} 范围为 {spread.min():.4g} - {spread.max():.4g}。",
                    "tone": "warn",
                }
            )
    missing = [w for w in warnings if "缺失" in w or "未发现" in w or "N/A" in w]
    if missing:
        insights.append(
            {
                "label": "Missing Signals",
                "title": f"{len(missing)} 项信号不可用",
                "body": "报告已保留 N/A/warning 状态；不会用占位数据混入真实实验结论。",
                "tone": "warn",
            }
        )
    return insights or [
        {
            "label": "Insight",
            "title": "当前数据不足以生成自动洞察",
            "body": "请补充 metrics/predictions 或训练日志后重新生成报告。",
            "tone": "warn",
        }
    ]


def _insight_cards(items: list[dict[str, str]]) -> str:
    return "".join(
        f"""
<article class="insight-card {escape(item['tone'])}">
  <span>{escape(item['label'])}</span>
  <h3>{escape(item['title'])}</h3>
  <p>{escape(item['body'])}</p>
</article>"""
        for item in items
    )


def _figure_cards(figures: list[dict[str, str]]) -> str:
    size_classes = ["span-8", "span-4", "span-6", "span-6", "span-4", "span-4", "span-4", "span-8"]
    cards = []
    for idx, fig in enumerate(figures):
        demo = '<span class="demo-badge">DEMO</span>' if fig.get("demo") else ""
        cards.append(
            f"""
<article class="chart-card {size_classes[idx % len(size_classes)]}" data-chart="{escape(fig['title'])}">
  <div class="chart-head"><h3>{escape(fig['title'])}</h3>{demo}<a href="assets/{escape(fig['file'])}">SVG</a></div>
  <img src="assets/{escape(fig['file'])}" alt="{escape(fig['title'])}"/>
</article>"""
        )
    return "".join(cards)


def _config_block(config: dict[str, Any]) -> str:
    if not config:
        return '<div class="empty-state">未发现 config.yaml/config.json。</div>'
    return f"<pre>{escape(json.dumps(config, ensure_ascii=False, indent=2, default=str))}</pre>"


def render_fancy_dashboard(
    output_path: Path,
    *,
    theme: ReportTheme,
    run_id: str,
    run_dir: Path,
    metrics: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    comparison: pd.DataFrame,
    figures: list[dict[str, str]],
    config: dict[str, Any],
    environment: dict[str, Any],
    warnings: list[str],
    primary_metric: str,
    is_demo: bool,
) -> None:
    best = _best_current(metrics, primary_metric)
    dataset_count = 0 if metrics is None or "dataset" not in metrics.columns else metrics["dataset"].nunique()
    model_count = 0 if metrics is None or "model" not in metrics.columns else metrics["model"].nunique()
    metric_rows = 0 if metrics is None else len(metrics)
    prediction_rows = 0 if predictions is None else len(predictions)
    runtime = _runtime(metrics)
    comp = comparison.iloc[0].to_dict() if not comparison.empty else {}
    rel = comp.get("relative_improvement_pct")
    rel_ok = rel is not None and pd.notna(rel)
    rel_class = "good" if rel_ok and float(rel) >= 0 else "bad" if rel_ok else "neutral"
    rel_text = f"{float(rel):+.2f}%" if rel_ok else "N/A"
    baseline_name = str(comp.get("baseline", "N/A"))
    insights = _insights(metrics, comparison, warnings, primary_metric)
    raw_metrics = metrics.head(240) if metrics is not None else None
    payload = {
        "theme": theme.name,
        "primaryMetric": primary_metric,
        "metrics": [] if metrics is None else metrics.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
    }
    payload_json = escape(json.dumps(_safe_json(payload), ensure_ascii=False, allow_nan=False), quote=False)
    css_palette = ", ".join(f'"{c}"' for c in theme.chart_palette)
    warning_html = "".join(f"<li>{escape(w)}</li>" for w in warnings) or "<li>无 warning。</li>"
    demo_mark = "Demo 可视化版本" if is_demo else "真实实验报告"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!doctype html>
<html lang="zh-CN" data-theme="{escape(theme.name)}">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Fancy ML Dashboard | {escape(run_id)}</title>
<style>
:root {{
  --primary: {theme.primary_color};
  --accent: {theme.accent_color};
  --success: {theme.success_color};
  --warning: {theme.warning_color};
  --danger: {theme.danger_color};
  --bg: {theme.background_color};
  --hero-bg: {theme.hero_background};
  --card: {theme.card_background};
  --card-alt: {theme.card_background_alt};
  --text: {theme.text_color};
  --muted: {theme.muted_text_color};
  --border: {theme.border_color};
  --chart-bg: {theme.chart_background};
  --shadow: {theme.shadow_style};
  --radius: {theme.border_radius}px;
  --gap: {theme.grid_spacing}px;
  --palette: {css_palette};
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
  background:
    radial-gradient(circle at 10% 0%, color-mix(in srgb, var(--primary) 18%, transparent), transparent 34%),
    radial-gradient(circle at 92% 14%, color-mix(in srgb, var(--accent) 14%, transparent), transparent 32%),
    var(--bg);
  color: var(--text);
}}
a {{ color: var(--primary); text-decoration: none; }}
.shell {{ display: grid; grid-template-columns: 236px minmax(0, 1fr); gap: 0; min-height: 100vh; }}
.sidebar {{
  position: sticky; top: 0; height: 100vh; padding: 24px 18px;
  border-right: 1px solid var(--border); background: color-mix(in srgb, var(--card) 84%, transparent);
  backdrop-filter: blur(22px);
}}
.brand {{ display: grid; gap: 6px; margin-bottom: 26px; }}
.brand b {{ font-size: 15px; letter-spacing: .02em; }}
.brand span {{ color: var(--muted); font-size: 12px; }}
.nav {{ display: grid; gap: 8px; }}
.nav a {{ padding: 10px 12px; border-radius: 999px; color: var(--muted); font-weight: 800; font-size: 13px; }}
.nav a.active, .nav a:hover {{ background: color-mix(in srgb, var(--primary) 14%, transparent); color: var(--text); }}
.content {{ padding: 24px clamp(18px, 3vw, 44px) 54px; min-width: 0; }}
.hero {{
  position: relative; overflow: hidden; border: 1px solid var(--border); border-radius: calc(var(--radius) + 10px);
  padding: 34px; min-height: 340px; background: var(--hero-bg); box-shadow: var(--shadow);
}}
.hero:after {{
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(120deg, rgba(255,255,255,.18), transparent 18%, transparent 70%, rgba(255,255,255,.06));
  opacity: .35;
}}
.hero-inner {{ position: relative; z-index: 1; display: grid; grid-template-columns: 1.3fr .7fr; gap: 28px; align-items: end; }}
.eyebrow {{ display: inline-flex; width: fit-content; gap: 8px; align-items: center; padding: 8px 12px; border: 1px solid var(--border); border-radius: 999px; background: color-mix(in srgb, var(--card) 72%, transparent); color: var(--muted); font-size: 12px; font-weight: 900; }}
h1 {{ margin: 18px 0 10px; font-size: clamp(34px, 5vw, 64px); line-height: 1.02; letter-spacing: 0; }}
.hero-copy {{ color: var(--muted); line-height: 1.7; max-width: 820px; font-size: 15px; }}
.hero-metric {{ display: grid; gap: 12px; justify-items: start; padding: 22px; border: 1px solid var(--border); border-radius: var(--radius); background: color-mix(in srgb, var(--card) 76%, transparent); backdrop-filter: blur(20px); }}
.hero-metric span {{ color: var(--muted); font-size: 12px; font-weight: 900; text-transform: uppercase; }}
.hero-metric strong {{ font-size: clamp(42px, 6vw, 78px); line-height: .9; color: var(--primary); }}
.hero-meta {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; margin-top: 24px; position: relative; z-index: 1; }}
.meta-card, .kpi-card, .insight-card, .chart-card, .panel {{
  border: 1px solid var(--border); border-radius: var(--radius); background: var(--card);
  box-shadow: 0 16px 38px color-mix(in srgb, #000 14%, transparent); backdrop-filter: blur(18px);
}}
.meta-card {{ padding: 14px; }}
.meta-card span, .kpi-top span, .insight-card span {{ color: var(--muted); font-size: 11px; font-weight: 900; text-transform: uppercase; }}
.meta-card b {{ display: block; margin-top: 7px; font-size: 18px; overflow-wrap: anywhere; }}
.pill {{ display: inline-flex; align-items: center; width: fit-content; border-radius: 999px; padding: 5px 9px; font-size: 12px; font-weight: 900; }}
.pill.good {{ color: var(--success); background: color-mix(in srgb, var(--success) 14%, transparent); }}
.pill.bad {{ color: var(--danger); background: color-mix(in srgb, var(--danger) 14%, transparent); }}
.pill.neutral {{ color: var(--muted); background: color-mix(in srgb, var(--muted) 12%, transparent); }}
.section {{ margin-top: 24px; }}
.section-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 14px; }}
.section-head h2 {{ margin: 0; font-size: 24px; letter-spacing: 0; }}
.section-head p {{ margin: 6px 0 0; color: var(--muted); line-height: 1.6; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(4, minmax(190px, 1fr)); gap: var(--gap); }}
.kpi-card {{ padding: 18px; transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease; }}
.kpi-card:hover, .chart-card:hover, .insight-card:hover {{ transform: translateY(-3px); border-color: color-mix(in srgb, var(--primary) 55%, var(--border)); box-shadow: var(--shadow); }}
.kpi-card.primary {{ background: linear-gradient(145deg, color-mix(in srgb, var(--primary) 16%, var(--card)), var(--card)); }}
.kpi-top {{ display: flex; justify-content: space-between; gap: 10px; align-items: center; }}
.kpi-top b {{ color: var(--muted); font-size: 11px; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.kpi-value {{ margin: 14px 0; font-size: 34px; font-weight: 950; color: var(--text); }}
.kpi-grid {{ font-size: 12px; color: var(--muted); }}
.kpi-card .kpi-grid {{ display: grid; grid-template-columns: 1fr auto; gap: 8px 12px; }}
.kpi-card .kpi-grid strong {{ color: var(--text); }}
.insight-grid {{ display: grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap: var(--gap); }}
.insight-card {{ padding: 18px; border-left: 4px solid var(--primary); transition: transform .22s ease, border-color .22s ease; }}
.insight-card.good {{ border-left-color: var(--success); }}
.insight-card.bad {{ border-left-color: var(--danger); }}
.insight-card.warn {{ border-left-color: var(--warning); }}
.insight-card h3 {{ margin: 10px 0 8px; font-size: 16px; }}
.insight-card p {{ margin: 0; color: var(--muted); line-height: 1.62; font-size: 13px; }}
.tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
.tab-btn {{ border: 1px solid var(--border); background: var(--card-alt); color: var(--muted); border-radius: 999px; padding: 10px 14px; font-weight: 900; cursor: pointer; }}
.tab-btn.active {{ color: var(--text); background: color-mix(in srgb, var(--primary) 17%, var(--card)); border-color: color-mix(in srgb, var(--primary) 55%, var(--border)); }}
.tab-pane {{ display: none; }}
.tab-pane.active {{ display: block; animation: rise .28s ease both; }}
.chart-grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: var(--gap); }}
.chart-card {{ padding: 16px; overflow: hidden; transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease; }}
.span-4 {{ grid-column: span 4; }}
.span-6 {{ grid-column: span 6; }}
.span-8 {{ grid-column: span 8; }}
.chart-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; }}
.chart-head h3 {{ margin: 0; font-size: 15px; }}
.chart-head a, .demo-badge {{ color: var(--muted); font-size: 11px; font-weight: 900; border: 1px solid var(--border); border-radius: 999px; padding: 5px 8px; }}
.demo-badge {{ color: var(--warning); border-color: color-mix(in srgb, var(--warning) 45%, var(--border)); }}
.chart-card img {{ display: block; width: 100%; height: auto; border-radius: calc(var(--radius) - 7px); background: var(--chart-bg); }}
.panel {{ padding: 18px; overflow: hidden; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }}
.toolbar input, .toolbar select {{ border: 1px solid var(--border); background: var(--card-alt); color: var(--text); border-radius: 12px; padding: 10px 12px; min-width: 220px; }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.data-table th, .data-table td {{ border-bottom: 1px solid var(--border); padding: 10px 9px; text-align: right; white-space: nowrap; }}
.data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
.data-table th {{ position: sticky; top: 0; z-index: 1; color: var(--muted); background: var(--card-alt); cursor: pointer; text-transform: uppercase; font-size: 11px; }}
.table-wrap {{ max-height: 480px; overflow: auto; border: 1px solid var(--border); border-radius: var(--radius); }}
details {{ border: 1px solid var(--border); border-radius: var(--radius); background: var(--card); padding: 16px; }}
summary {{ cursor: pointer; font-weight: 950; }}
pre {{ white-space: pre-wrap; color: var(--muted); line-height: 1.55; font-size: 12px; }}
.warning-panel {{ border: 1px solid color-mix(in srgb, var(--warning) 42%, var(--border)); background: color-mix(in srgb, var(--warning) 9%, var(--card)); color: var(--text); border-radius: var(--radius); padding: 16px 18px; }}
.warning-panel li {{ margin: 6px 0; color: var(--muted); }}
.empty-state {{ padding: 16px; border: 1px dashed var(--border); border-radius: var(--radius); color: var(--muted); background: var(--card-alt); }}
.mini-svg {{ width: 100%; height: 280px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--chart-bg); }}
.svg-axis {{ fill: var(--muted); font-size: 12px; }}
.svg-title {{ fill: var(--text); font-size: 15px; font-weight: 900; }}
.svg-label {{ fill: var(--text); font-size: 12px; font-weight: 800; }}
.tooltip {{ position: fixed; pointer-events: none; opacity: 0; z-index: 20; background: color-mix(in srgb, var(--card) 94%, #000); color: var(--text); border: 1px solid var(--border); border-radius: 12px; padding: 9px 10px; font-size: 12px; box-shadow: var(--shadow); transition: opacity .14s ease; }}
.tooltip.show {{ opacity: 1; }}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@media print {{
  .sidebar, .tabs, .toolbar {{ display: none; }}
  .shell {{ display: block; }}
  body {{ background: white; color: #0f172a; }}
  .chart-card, .panel, .kpi-card, .insight-card {{ break-inside: avoid; box-shadow: none; }}
}}
@media (max-width: 1180px) {{
  .shell {{ grid-template-columns: 1fr; }}
  .sidebar {{ position: relative; height: auto; }}
  .nav {{ grid-template-columns: repeat(3, 1fr); }}
  .hero-inner, .hero-meta, .kpi-grid, .insight-grid {{ grid-template-columns: 1fr 1fr; }}
  .span-4, .span-6, .span-8 {{ grid-column: span 12; }}
}}
@media (max-width: 720px) {{
  .content {{ padding: 14px; }}
  .hero {{ padding: 22px; }}
  .hero-inner, .hero-meta, .kpi-grid, .insight-grid {{ grid-template-columns: 1fr; }}
  .nav {{ grid-template-columns: 1fr 1fr; }}
  h1 {{ font-size: 34px; }}
}}
</style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="brand"><b>TabPFN-TS Dashboard</b><span>{escape(demo_mark)} · {escape(theme.name)}</span></div>
    <nav class="nav">
      <a href="#overview" class="active">Overview</a>
      <a href="#kpi">KPI</a>
      <a href="#insights">Insights</a>
      <a href="#charts">Charts</a>
      <a href="#tables">Tables</a>
      <a href="#config">Config</a>
    </nav>
  </aside>
  <main class="content">
    <section class="hero" id="overview">
      <div class="hero-inner">
        <div>
          <div class="eyebrow">AI Research Dashboard · {escape(timestamp)}</div>
          <h1>{escape(run_id)}</h1>
          <p class="hero-copy">基于真实实验产物自动生成的高级可视化 dashboard。训练曲线缺失时会明确标注 demo 占位，不会把占位曲线混入真实实验结论。</p>
          <span class="pill {rel_class}">Baseline: {escape(baseline_name)} · {escape(rel_text)}</span>
        </div>
        <div class="hero-metric">
          <span>Primary Metric · {escape(primary_metric.upper())}</span>
          <strong data-count="{escape(_fmt(best['value'], 6))}">{escape(_fmt(best['value']))}</strong>
          <span>Best current model: {escape(str(best['model']))}</span>
        </div>
      </div>
      <div class="hero-meta">
        <div class="meta-card"><span>Run ID</span><b>{escape(run_id)}</b></div>
        <div class="meta-card"><span>Datasets / Models</span><b>{dataset_count} / {model_count}</b></div>
        <div class="meta-card"><span>Rows</span><b>{metric_rows} metrics · {prediction_rows} preds</b></div>
        <div class="meta-card"><span>Runtime</span><b>{escape(runtime)}</b></div>
      </div>
    </section>

    <section class="section" id="kpi">
      <div class="section-head"><div><h2>KPI 指标卡片</h2><p>当前 run 与 baseline 的核心指标对比，缺失项保持 N/A。</p></div></div>
      <div class="kpi-grid">{_metric_cards(metrics, comparison, primary_metric)}</div>
    </section>

    <section class="section" id="insights">
      <div class="section-head"><div><h2>自动洞察</h2><p>只基于可用真实数据计算，不输出未经数据支持的结论。</p></div></div>
      <div class="insight-grid">{_insight_cards(insights)}</div>
    </section>

    <section class="section" id="charts">
      <div class="section-head"><div><h2>Dashboard 图表区</h2><p>Bento grid 布局，重点图表更大，辅助诊断图更紧凑。</p></div></div>
      <div class="tabs">
        <button class="tab-btn active" data-tab="overview-charts">Overview</button>
        <button class="tab-btn" data-tab="baseline-charts">Baselines</button>
        <button class="tab-btn" data-tab="diagnostics-charts">Diagnostics</button>
        <button class="tab-btn" data-tab="interactive-charts">Interactive</button>
      </div>
      <div id="overview-charts" class="tab-pane active"><div class="chart-grid">{_figure_cards(figures)}</div></div>
      <div id="baseline-charts" class="tab-pane"><div class="panel"><h3>Baseline 对比表</h3><div class="table-wrap">{_table(comparison, "没有 baseline 对比数据。")}</div></div></div>
      <div id="diagnostics-charts" class="tab-pane"><div class="chart-grid">{_figure_cards([f for f in figures if "残差" in f["title"] or "训练" in f["title"] or "效率" in f["title"]])}</div></div>
      <div id="interactive-charts" class="tab-pane">
        <div class="panel">
          <div class="toolbar"><select id="metric-select"></select><select id="dataset-select"></select></div>
          <svg id="mini-ranking" class="mini-svg" viewBox="0 0 980 280"></svg>
        </div>
      </div>
    </section>

    <section class="section" id="tables">
      <div class="section-head"><div><h2>原始指标与 Baseline</h2><p>可筛选、可点击表头排序。</p></div></div>
      <div class="panel">
        <div class="toolbar"><input id="table-filter" placeholder="筛选 metrics 表格..."/></div>
        <div class="table-wrap" id="metrics-table">{_table(raw_metrics, "未找到 forecast_metrics.csv。")}</div>
      </div>
    </section>

    <section class="section" id="config">
      <div class="section-head"><div><h2>配置、环境与 Warning</h2><p>所有缺失信号以 warning 保留。</p></div></div>
      <details open><summary>配置参数</summary>{_config_block(config)}</details>
      <br/>
      <details><summary>运行环境</summary><pre>{escape(json.dumps(environment, ensure_ascii=False, indent=2, default=str))}</pre></details>
      <br/>
      <details open><summary>Raw metrics snapshot</summary><div class="table-wrap">{_table(raw_metrics, "未找到 metrics。")}</div></details>
      <br/>
      <div class="warning-panel"><b>Warnings / Missing Metrics</b><ul>{warning_html}</ul></div>
      <p class="hero-copy">源 run 目录：{escape(str(run_dir))}</p>
      <p><a href="metrics_demo.json">metrics_demo.json</a> · <a href="summary.csv">summary.csv</a></p>
    </section>
  </main>
</div>
<script id="dashboard-payload" type="application/json">{payload_json}</script>
<script>
const payload = JSON.parse(document.getElementById("dashboard-payload").textContent);
const palette = [{css_palette}];
const lowerIsBetter = metric => ["loss","mse","mae","rmse","smape","wape","mase","runtime","train_time_s","inference_time_s","gpu_memory_mb"].includes(String(metric).toLowerCase());
function tip() {{
  let node = document.querySelector(".tooltip");
  if (!node) {{ node = document.createElement("div"); node.className = "tooltip"; document.body.appendChild(node); }}
  return node;
}}
function showTip(event, html) {{
  const node = tip();
  node.innerHTML = html;
  node.style.left = `${{event.clientX + 14}}px`;
  node.style.top = `${{event.clientY + 14}}px`;
  node.classList.add("show");
}}
function hideTip() {{ tip().classList.remove("show"); }}
document.querySelectorAll(".tab-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".tab-btn").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(x => x.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
    drawMiniRanking();
  }});
}});
document.querySelectorAll(".data-table th").forEach(th => {{
  th.addEventListener("click", () => {{
    const table = th.closest("table");
    const tbody = table.querySelector("tbody");
    const index = Array.from(th.parentElement.children).indexOf(th);
    const asc = th.dataset.asc !== "true";
    th.dataset.asc = String(asc);
    Array.from(tbody.querySelectorAll("tr")).sort((a,b) => {{
      const av = a.children[index].textContent.trim();
      const bv = b.children[index].textContent.trim();
      const an = Number(av), bn = Number(bv);
      const cmp = Number.isFinite(an) && Number.isFinite(bn) ? an - bn : av.localeCompare(bv);
      return asc ? cmp : -cmp;
    }}).forEach(row => tbody.appendChild(row));
  }});
}});
const filter = document.getElementById("table-filter");
if (filter) filter.addEventListener("input", () => {{
  const q = filter.value.toLowerCase();
  document.querySelectorAll("#metrics-table tbody tr").forEach(row => row.style.display = row.textContent.toLowerCase().includes(q) ? "" : "none");
}});
document.querySelectorAll("[data-count]").forEach(node => {{
  const target = Number(node.dataset.count);
  if (!Number.isFinite(target)) return;
  const end = target;
  const start = performance.now();
  const raw = node.textContent;
  function step(now) {{
    const t = Math.min(1, (now - start) / 780);
    node.textContent = (end * (1 - Math.pow(1 - t, 3))).toPrecision(4);
    if (t < 1) requestAnimationFrame(step); else node.textContent = raw;
  }}
  requestAnimationFrame(step);
}});
const metrics = payload.metrics || [];
const metricSelect = document.getElementById("metric-select");
const datasetSelect = document.getElementById("dataset-select");
const metricCols = ["smape","mae","rmse","wape","mase","train_time_s","inference_time_s"].filter(col => metrics.some(row => Number.isFinite(Number(row[col]))));
const datasets = [...new Set(metrics.map(row => row.dataset).filter(Boolean))].sort();
if (metricSelect) metricSelect.innerHTML = metricCols.map(m => `<option value="${{m}}">${{m.toUpperCase()}}</option>`).join("");
if (datasetSelect) datasetSelect.innerHTML = datasets.map(d => `<option value="${{d}}">${{d}}</option>`).join("");
if (metricSelect) metricSelect.value = payload.primaryMetric;
function svgEl(name, attrs = {{}}, text = "") {{
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([k,v]) => node.setAttribute(k,v));
  if (text) node.textContent = text;
  return node;
}}
function drawMiniRanking() {{
  const svg = document.getElementById("mini-ranking");
  if (!svg) return;
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const metric = metricSelect.value || payload.primaryMetric;
  const dataset = datasetSelect.value || datasets[0];
  const rows = metrics.filter(row => row.dataset === dataset && Number.isFinite(Number(row[metric])));
  const grouped = new Map();
  rows.forEach(row => {{
    if (!grouped.has(row.model)) grouped.set(row.model, []);
    grouped.get(row.model).push(Number(row[metric]));
  }});
  const values = [...grouped.entries()].map(([model, vals]) => [model, vals.reduce((a,b)=>a+b,0)/vals.length]).sort((a,b) => lowerIsBetter(metric) ? a[1]-b[1] : b[1]-a[1]);
  svg.append(svgEl("text", {{x: 24, y: 32, class: "svg-title"}}, `${{dataset}} · ${{metric.toUpperCase()}} ranking`));
  if (!values.length) {{
    svg.append(svgEl("text", {{x: 24, y: 72, class: "svg-axis"}}, "当前筛选下没有数据。"));
    return;
  }}
  const max = Math.max(...values.map(x => x[1]), 1);
  values.forEach(([model, value], idx) => {{
    const y = 62 + idx * 25;
    const width = Math.max(4, value / max * 640);
    const color = palette[idx % palette.length];
    svg.append(svgEl("text", {{x: 24, y: y + 15, class: "svg-axis"}}, `${{idx + 1}}. ${{model}}`));
    const bar = svgEl("rect", {{x: 240, y, width, height: 17, rx: 9, fill: color, opacity: .88}});
    bar.addEventListener("mousemove", e => showTip(e, `<b>${{model}}</b><br>${{metric.toUpperCase()}}: ${{value.toFixed(6)}}`));
    bar.addEventListener("mouseleave", hideTip);
    svg.append(bar);
    svg.append(svgEl("text", {{x: 248 + width, y: y + 14, class: "svg-label"}}, value.toPrecision(4)));
  }});
}}
if (metricSelect) metricSelect.addEventListener("change", drawMiniRanking);
if (datasetSelect) datasetSelect.addEventListener("change", drawMiniRanking);
drawMiniRanking();
</script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
