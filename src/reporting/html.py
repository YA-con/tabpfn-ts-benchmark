"""HTML rendering for reusable experiment reports."""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from src.reporting.comparison import MetricSummary


def _fmt(value: object, digits: int = 4) -> str:
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


def _table(df: pd.DataFrame | None, empty: str = "暂无数据") -> str:
    if df is None or df.empty:
        return f'<div class="empty">{escape(empty)}</div>'
    display = df.copy()
    numeric_cols = display.select_dtypes(include="number").columns
    for col in numeric_cols:
        display[col] = display[col].map(_fmt)
    display = display.fillna("N/A")
    return display.to_html(index=False, classes="data-table", border=0, escape=True, na_rep="N/A")


def _metric_cards(summaries: list[MetricSummary]) -> str:
    if not summaries:
        return '<div class="empty">未找到可汇总的数值指标。</div>'
    cards = []
    for item in summaries[:8]:
        cards.append(
            '<div class="metric-card">'
            f'<div class="metric-name">{escape(item.metric.upper())}</div>'
            f'<div class="metric-value">{escape(_fmt(item.value))}</div>'
            f'<div class="metric-sub">最佳模型: {escape(item.model)}</div>'
            "</div>"
        )
    return '<div class="metric-grid">' + "".join(cards) + "</div>"


def _config_block(config: dict[str, Any]) -> str:
    if not config:
        return '<div class="empty">未发现 config.yaml/config.json；该项记为 N/A。</div>'
    text = json.dumps(config, ensure_ascii=False, indent=2, default=str)
    return f"<pre>{escape(text)}</pre>"


def _json_safe(value: Any) -> Any:
    """Convert non-JSON float values into null for browser-side payloads."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _interactive_section(
    metrics: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    primary_metric: str,
) -> str:
    """Render offline interactive SVG charts backed by embedded real data."""

    if metrics is None or metrics.empty:
        return '<div class="empty">没有 metrics 数据，无法生成动态交互图表。</div>'
    metric_cols = [
        col
        for col in ["smape", "mae", "rmse", "wape", "mase", "train_time_s", "inference_time_s"]
        if col in metrics.columns
    ]
    metric_payload = metrics.copy()
    if predictions is not None and not predictions.empty:
        pred_cols = [col for col in ["dataset", "domain", "model", "unique_id", "ds", "y", "y_hat"] if col in predictions.columns]
        pred_payload = predictions[pred_cols].copy()
        pred_payload["ds"] = pred_payload["ds"].astype(str)
    else:
        pred_payload = pd.DataFrame()
    payload = {
        "primaryMetric": primary_metric,
        "metricColumns": metric_cols,
        "metrics": metric_payload.to_dict(orient="records"),
        "predictions": pred_payload.to_dict(orient="records"),
    }
    payload_json = escape(
        json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False),
        quote=False,
    )
    return f"""
<div class="dynamic-report">
  <div class="dynamic-toolbar">
    <label>指标<select id="dyn-metric"></select></label>
    <label>数据集<select id="dyn-dataset"></select></label>
    <label>模型<select id="dyn-model"></select></label>
    <label>序列<select id="dyn-series"></select></label>
    <button type="button" id="dyn-play">播放</button>
    <input id="dyn-step" type="range" min="1" value="1"/>
  </div>
  <div class="dynamic-progress"><span id="dyn-progress"></span></div>
  <div class="dynamic-grid">
    <article class="dynamic-card dynamic-wide">
      <h3>预测轨迹播放</h3>
      <svg id="dyn-forecast" viewBox="0 0 1100 420" class="dynamic-svg"></svg>
    </article>
    <article class="dynamic-card">
      <h3>指标排名动态图</h3>
      <svg id="dyn-ranking" viewBox="0 0 620 420" class="dynamic-svg"></svg>
    </article>
    <article class="dynamic-card">
      <h3>残差直方图</h3>
      <svg id="dyn-residual" viewBox="0 0 620 420" class="dynamic-svg"></svg>
    </article>
    <article class="dynamic-card">
      <h3>效率-精度散点图</h3>
      <svg id="dyn-efficiency" viewBox="0 0 620 420" class="dynamic-svg"></svg>
    </article>
    <article class="dynamic-card">
      <h3>数据集排名轨迹</h3>
      <svg id="dyn-bump" viewBox="0 0 620 420" class="dynamic-svg"></svg>
    </article>
  </div>
</div>
<script id="interactive-payload" type="application/json">{payload_json}</script>
"""


def render_html_report(
    output_path: Path,
    *,
    run_id: str,
    run_dir: Path,
    summaries: list[MetricSummary],
    comparison: pd.DataFrame,
    metrics: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    figures: list[dict[str, str]],
    config: dict[str, Any],
    environment: dict[str, Any],
    warnings: list[str],
    primary_metric: str,
    conclusions: list[str] | None = None,
) -> None:
    """Write the final HTML report."""

    metric_rows = 0 if metrics is None else len(metrics)
    prediction_rows = 0 if predictions is None else len(predictions)
    dataset_count = 0 if metrics is None or "dataset" not in metrics.columns else metrics["dataset"].nunique()
    model_count = 0 if metrics is None or "model" not in metrics.columns else metrics["model"].nunique()
    warning_html = "".join(f"<li>{escape(item)}</li>" for item in warnings) or "<li>无</li>"
    figure_html = "".join(
        '<article class="figure-card">'
        f'<h3>{escape(fig["title"])}</h3>'
        f'<a href="assets/{escape(fig["file"])}" class="figure-link">'
        f'<img src="assets/{escape(fig["file"])}" alt="{escape(fig["title"])}"/>'
        "</a>"
        "</article>"
        for fig in figures
    )
    raw_metrics = metrics.head(200) if metrics is not None else None
    conclusion_html = "".join(
        f"<li>{escape(item)}</li>" for item in (conclusions or ["暂无自动结论。"])
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>实验报告 | {escape(run_id)}</title>
<style>
:root {{
  --bg: #eef2f7;
  --ink: #0f172a;
  --muted: #64748b;
  --line: #dbe3ee;
  --panel: #ffffff;
  --accent: #db2777;
  --accent2: #0ea5e9;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
header {{ background: radial-gradient(circle at 14% 20%, rgba(219,39,119,.26), transparent 28%), linear-gradient(135deg, #101827, #172033 62%, #223044); color: white; }}
.hero {{ max-width: 1440px; margin: 0 auto; padding: 34px 36px 28px; }}
.kicker {{ color: #93c5fd; font-weight: 800; letter-spacing: .08em; font-size: 12px; text-transform: uppercase; }}
h1 {{ margin: 8px 0 8px; font-size: 34px; letter-spacing: 0; }}
.subtitle {{ color: #cbd5e1; line-height: 1.6; max-width: 980px; }}
main {{ max-width: 1440px; margin: 0 auto; padding: 22px 36px 44px; }}
.overview {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 18px; }}
.stat, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 1px 2px rgba(15,23,42,.05); }}
.stat {{ padding: 16px; }}
.stat span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 700; }}
.stat b {{ display: block; margin-top: 6px; font-size: 24px; }}
section {{ padding: 20px; margin-bottom: 18px; }}
h2 {{ margin: 0 0 14px; font-size: 20px; }}
h3 {{ margin: 0 0 10px; font-size: 15px; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; }}
.metric-card {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; background: #f8fafc; }}
.metric-name {{ color: var(--muted); font-size: 12px; font-weight: 800; }}
.metric-value {{ margin-top: 6px; font-size: 28px; font-weight: 850; color: var(--accent); }}
.metric-sub {{ margin-top: 6px; color: #475569; font-size: 12px; }}
.conclusion-list {{ display: grid; gap: 10px; margin: 0; padding: 0; list-style: none; }}
.conclusion-list li {{ border-left: 4px solid var(--accent); background: #f8fafc; padding: 12px 14px; border-radius: 6px; line-height: 1.55; }}
.figure-grid {{ display: grid; grid-template-columns: repeat(2, minmax(420px, 1fr)); gap: 16px; }}
.figure-card {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; background: #fbfdff; overflow-x: auto; }}
.figure-link {{ display: block; }}
.figure-card img {{ width: 100%; min-width: 420px; height: auto; display: block; }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.data-table th, .data-table td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 9px; text-align: right; }}
.data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
.data-table td {{ max-width: 260px; overflow-wrap: anywhere; }}
.data-table th {{ color: #475569; position: sticky; top: 0; background: #fff; cursor: pointer; user-select: none; }}
.data-table th::after {{ content: " ↕"; color: #94a3b8; font-weight: 400; }}
.table-wrap {{ max-height: 460px; overflow: auto; border: 1px solid #e2e8f0; border-radius: 8px; }}
.empty {{ padding: 14px; border: 1px dashed #cbd5e1; border-radius: 8px; color: var(--muted); background: #f8fafc; }}
details {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; background: #fbfdff; }}
summary {{ cursor: pointer; font-weight: 800; }}
pre {{ margin: 12px 0 0; white-space: pre-wrap; font-size: 12px; line-height: 1.5; color: #1e293b; }}
.warnings {{ color: #92400e; background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 12px 14px; }}
.toolbar {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px; }}
.toolbar input {{ border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px 10px; min-width: 260px; }}
.dynamic-report {{ display: grid; gap: 16px; }}
.dynamic-toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: end; padding: 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; }}
.dynamic-toolbar label {{ display: grid; gap: 5px; color: var(--muted); font-size: 12px; font-weight: 800; }}
.dynamic-toolbar select, .dynamic-toolbar button, .dynamic-toolbar input {{ height: 34px; border: 1px solid #cbd5e1; border-radius: 6px; background: white; color: var(--ink); }}
.dynamic-toolbar select {{ min-width: 150px; padding: 0 8px; }}
.dynamic-toolbar button {{ padding: 0 16px; background: var(--ink); color: white; font-weight: 800; cursor: pointer; }}
.dynamic-toolbar button.is-playing {{ background: var(--accent); box-shadow: 0 0 0 4px rgba(219,39,119,.14); }}
.dynamic-toolbar input {{ min-width: 220px; accent-color: var(--accent); }}
.dynamic-progress {{ height: 6px; border-radius: 999px; overflow: hidden; background: #e2e8f0; }}
.dynamic-progress span {{ display: block; width: 0%; height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width .42s ease; }}
.dynamic-grid {{ display: grid; grid-template-columns: repeat(2, minmax(420px, 1fr)); gap: 16px; }}
.dynamic-card {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; background: #fbfdff; overflow-x: auto; position: relative; }}
.dynamic-card:hover {{ border-color: #bfdbfe; box-shadow: 0 10px 26px rgba(15, 23, 42, .08); }}
.dynamic-wide {{ grid-column: 1 / -1; }}
.dynamic-svg {{ width: 100%; min-width: 460px; height: auto; display: block; background: #ffffff; border-radius: 6px; }}
.dynamic-svg .animated {{ animation: fadeSlide .38s ease both; transform-origin: center; }}
.dynamic-svg .draw-line {{ stroke-dasharray: 1200; stroke-dashoffset: 1200; animation: drawLine .8s ease forwards; }}
.dynamic-svg .hoverable {{ transition: opacity .18s ease, transform .18s ease; cursor: default; }}
.dynamic-svg .hoverable:hover {{ opacity: 1; transform: scale(1.025); }}
.tooltip {{ position: fixed; pointer-events: none; opacity: 0; transform: translate(10px, 10px); transition: opacity .12s ease; z-index: 50; background: rgba(15,23,42,.92); color: white; padding: 8px 10px; border-radius: 6px; font-size: 12px; max-width: 280px; box-shadow: 0 10px 28px rgba(15,23,42,.25); }}
.tooltip.show {{ opacity: 1; }}
.svg-axis {{ font-size: 12px; fill: #475569; }}
.svg-title {{ font-size: 14px; font-weight: 800; fill: #0f172a; }}
.svg-value {{ font-size: 12px; fill: #334155; }}
@keyframes fadeSlide {{ from {{ opacity: 0; transform: translateY(7px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes drawLine {{ to {{ stroke-dashoffset: 0; }} }}
@media (max-width: 980px) {{
  .hero, main {{ padding-left: 16px; padding-right: 16px; }}
  .overview, .metric-grid, .figure-grid, .dynamic-grid {{ grid-template-columns: 1fr; }}
  .figure-card img {{ min-width: 360px; }}
  .dynamic-svg {{ min-width: 360px; }}
}}
</style>
</head>
<body>
<header>
  <div class="hero">
    <div class="kicker">Experiment Visualization Report</div>
    <h1>{escape(run_id)}</h1>
    <div class="subtitle">自动收集真实实验产物生成的可视化报告。缺失的训练曲线、分类指标或日志不会被伪造，会在报告中以 N/A 或 warning 标出。</div>
  </div>
</header>
<main>
  <div class="overview">
    <div class="stat"><span>数据集</span><b>{dataset_count}</b></div>
    <div class="stat"><span>模型</span><b>{model_count}</b></div>
    <div class="stat"><span>指标行数</span><b>{metric_rows}</b></div>
    <div class="stat"><span>预测行数</span><b>{prediction_rows}</b></div>
  </div>
  <section><h2>结论摘要</h2><ul class="conclusion-list">{conclusion_html}</ul></section>
  <section><h2>核心指标卡片</h2>{_metric_cards(summaries)}</section>
  <section><h2>Baseline 对比</h2><div class="table-wrap">{_table(comparison, "没有提供 baseline，或 baseline 指标不可用。")}</div></section>
  <section><h2>动态交互图表</h2>{_interactive_section(metrics, predictions, primary_metric)}</section>
  <section><h2>Matplotlib 可视化图表</h2><div class="figure-grid">{figure_html or '<div class="empty">没有足够数据生成图表。</div>'}</div></section>
  <section><h2>训练过程曲线</h2><div class="empty">N/A：当前 run 目录未发现 loss/accuracy/TensorBoard/wandb 训练曲线文件。</div></section>
  <section><h2>错误分析与样本分析</h2><div class="empty">当前支持预测任务的残差与真实-预测散点分析；分类 confusion matrix / ROC / PR 曲线需要真实分类输出后自动启用。</div></section>
  <section>
    <h2>原始 Metrics</h2>
    <div class="toolbar"><input id="metric-filter" placeholder="筛选表格..."/></div>
    <div class="table-wrap" id="metrics-table">{_table(raw_metrics, "未找到 forecast_metrics.csv。")}</div>
  </section>
  <section><h2>配置参数</h2><details open><summary>展开/折叠配置</summary>{_config_block(config)}</details></section>
  <section><h2>运行环境</h2><pre>{escape(json.dumps(environment, ensure_ascii=False, indent=2, default=str))}</pre></section>
  <section><h2>Warnings</h2><ul class="warnings">{warning_html}</ul></section>
  <section><h2>产物链接</h2>
    <ul>
      <li><a href="metrics.json">metrics.json</a></li>
      <li><a href="summary.csv">summary.csv</a></li>
      <li>源 run 目录: {escape(str(run_dir))}</li>
      <li>主指标: {escape(primary_metric)}</li>
    </ul>
  </section>
</main>
<script>
document.querySelectorAll(".data-table th").forEach((th, idx) => {{
  th.addEventListener("click", () => {{
    const table = th.closest("table");
    const tbody = table.querySelector("tbody");
    const headers = Array.from(th.parentElement.children);
    const colIndex = headers.indexOf(th);
    const rows = Array.from(tbody.querySelectorAll("tr"));
    const asc = th.dataset.asc !== "true";
    th.dataset.asc = String(asc);
    rows.sort((a, b) => {{
      const av = a.children[colIndex].textContent.trim();
      const bv = b.children[colIndex].textContent.trim();
      const an = Number(av), bn = Number(bv);
      const cmp = Number.isFinite(an) && Number.isFinite(bn) ? an - bn : av.localeCompare(bv);
      return asc ? cmp : -cmp;
    }});
    rows.forEach(row => tbody.appendChild(row));
  }});
}});
const filter = document.getElementById("metric-filter");
if (filter) {{
  filter.addEventListener("input", () => {{
    const q = filter.value.toLowerCase();
    document.querySelectorAll("#metrics-table tbody tr").forEach(row => {{
      row.style.display = row.textContent.toLowerCase().includes(q) ? "" : "none";
    }});
  }});
}}
(() => {{
  const payloadNode = document.getElementById("interactive-payload");
  if (!payloadNode) return;
  const payload = JSON.parse(payloadNode.textContent);
  const metrics = (payload.metrics || []).map(row => {{
    const out = {{...row}};
    for (const key of payload.metricColumns || []) out[key] = Number(out[key]);
    return out;
  }});
  const predictions = (payload.predictions || []).map(row => ({{
    ...row,
    ds: String(row.ds),
    y: Number(row.y),
    y_hat: Number(row.y_hat)
  }})).filter(row => Number.isFinite(row.y) && Number.isFinite(row.y_hat));
  const colors = ["#db2777", "#0ea5e9", "#ea580c", "#16a34a", "#7c3aed", "#be123c", "#0f766e", "#2563eb", "#64748b"];
  const colorMap = new Map();
  function color(key) {{
    if (!colorMap.has(key)) colorMap.set(key, colors[colorMap.size % colors.length]);
    return colorMap.get(key);
  }}
  function esc(value) {{
    return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll('"', "&quot;");
  }}
  function svg(id) {{ return document.getElementById(id); }}
  function clear(node) {{ while (node.firstChild) node.removeChild(node.firstChild); }}
  function el(name, attrs = {{}}, text = null) {{
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== null) node.textContent = text;
    return node;
  }}
  function tooltip() {{
    let tip = document.querySelector(".tooltip");
    if (!tip) {{
      tip = document.createElement("div");
      tip.className = "tooltip";
      document.body.appendChild(tip);
    }}
    return tip;
  }}
  function attachTooltip(node, text) {{
    node.classList.add("hoverable");
    node.addEventListener("mousemove", event => {{
      const tip = tooltip();
      tip.innerHTML = text;
      tip.style.left = `${{event.clientX + 14}}px`;
      tip.style.top = `${{event.clientY + 14}}px`;
      tip.classList.add("show");
    }});
    node.addEventListener("mouseleave", () => tooltip().classList.remove("show"));
    return node;
  }}
  function extent(values, fallback = [0, 1]) {{
    const clean = values.map(Number).filter(Number.isFinite);
    if (!clean.length) return fallback;
    let lo = Math.min(...clean);
    let hi = Math.max(...clean);
    if (lo === hi) {{ lo -= 1; hi += 1; }}
    return [lo, hi];
  }}
  function scale(value, lo, hi, span) {{
    return (Number(value) - lo) / (hi - lo || 1) * span;
  }}
  function mean(values) {{
    const clean = values.filter(Number.isFinite);
    return clean.length ? clean.reduce((a, b) => a + b, 0) / clean.length : NaN;
  }}
  function groupMean(rows, key, metric) {{
    const groups = new Map();
    rows.forEach(row => {{
      if (!groups.has(row[key])) groups.set(row[key], []);
      groups.get(row[key]).push(Number(row[metric]));
    }});
    return [...groups.entries()].map(([name, vals]) => ({{name, value: mean(vals)}})).filter(d => Number.isFinite(d.value));
  }}
  function lowerIsBetter(metric) {{
    return ["loss", "mse", "mae", "rmse", "smape", "wape", "mase", "runtime", "train_time_s", "inference_time_s", "gpu_memory_mb"].includes(String(metric).toLowerCase());
  }}
  function addTitle(node, text) {{
    node.append(el("text", {{x: 18, y: 24, class: "svg-title"}}, text));
  }}
  function addEmpty(node, text) {{
    clear(node);
    node.append(el("text", {{x: 24, y: 42, class: "svg-axis"}}, text));
  }}

  const metricSelect = document.getElementById("dyn-metric");
  const datasetSelect = document.getElementById("dyn-dataset");
  const modelSelect = document.getElementById("dyn-model");
  const seriesSelect = document.getElementById("dyn-series");
  const stepSlider = document.getElementById("dyn-step");
  const playButton = document.getElementById("dyn-play");
  const progressBar = document.getElementById("dyn-progress");
  let timer = null;

  const metricOptions = payload.metricColumns || [];
  const datasets = [...new Set(metrics.map(row => row.dataset).filter(Boolean))].sort();
  const models = [...new Set(metrics.map(row => row.model).filter(Boolean))].sort();
  metricSelect.innerHTML = metricOptions.map(m => `<option value="${{esc(m)}}">${{esc(m.toUpperCase())}}</option>`).join("");
  metricSelect.value = metricOptions.includes(payload.primaryMetric) ? payload.primaryMetric : metricOptions[0] || "";
  datasetSelect.innerHTML = datasets.map(d => `<option value="${{esc(d)}}">${{esc(d)}}</option>`).join("");
  modelSelect.innerHTML = models.map(m => `<option value="${{esc(m)}}">${{esc(m)}}</option>`).join("");
  if (models.includes("tabpfn_ts")) modelSelect.value = "tabpfn_ts";

  function preparePredictionSteps() {{
    const grouped = new Map();
    predictions.forEach(row => {{
      const key = `${{row.dataset}}|${{row.model}}|${{row.unique_id}}`;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(row);
    }});
    grouped.forEach(rows => {{
      rows.sort((a, b) => a.ds.localeCompare(b.ds));
      rows.forEach((row, idx) => row.step = idx + 1);
    }});
  }}
  preparePredictionSteps();

  function refreshSeries() {{
    const dataset = datasetSelect.value;
    const model = modelSelect.value;
    const series = [...new Set(predictions.filter(row => row.dataset === dataset && row.model === model).map(row => row.unique_id))].sort();
    seriesSelect.innerHTML = series.slice(0, 120).map(s => `<option value="${{esc(s)}}">${{esc(s)}}</option>`).join("");
    const steps = predictions.filter(row => row.dataset === dataset && row.model === model && (!seriesSelect.value || row.unique_id === seriesSelect.value)).map(row => row.step || 1);
    const maxStep = Math.max(1, ...steps);
    stepSlider.max = String(maxStep);
    stepSlider.value = String(Math.min(Number(stepSlider.value || 1), maxStep));
    progressBar.style.width = `${{Math.round(Number(stepSlider.value) / maxStep * 100)}}%`;
  }}

  function drawRanking() {{
    const node = svg("dyn-ranking");
    clear(node);
    const metric = metricSelect.value;
    const dataset = datasetSelect.value;
    const rows = metrics.filter(row => row.dataset === dataset && Number.isFinite(Number(row[metric])));
    if (!rows.length) return addEmpty(node, "当前数据集没有该指标。");
    const grouped = groupMean(rows, "model", metric).sort((a, b) => lowerIsBetter(metric) ? a.value - b.value : b.value - a.value);
    const maxValue = Math.max(...grouped.map(d => d.value), 1);
    const left = 188, top = 48, rowH = 34, barW = 350;
    addTitle(node, `${{dataset}} | ${{metric.toUpperCase()}} 排名`);
    grouped.forEach((item, idx) => {{
      const y = top + idx * rowH;
      const width = Math.max(3, scale(item.value, 0, maxValue, barW));
      node.append(el("text", {{x: 18, y: y + 20, class: "svg-axis animated"}}, `${{idx + 1}}. ${{item.name}}`));
      const bar = el("rect", {{x: left, y: y + 5, width, height: 21, rx: 4, fill: color(item.name), opacity: 0.86, class: "animated"}});
      node.append(attachTooltip(bar, `<b>${{esc(item.name)}}</b><br>${{metric.toUpperCase()}}: ${{item.value.toFixed(6)}}<br>排名: #${{idx + 1}}`));
      node.append(el("text", {{x: left + width + 8, y: y + 21, class: "svg-value animated"}}, item.value.toFixed(4)));
    }});
  }}

  function drawForecast() {{
    const node = svg("dyn-forecast");
    clear(node);
    const dataset = datasetSelect.value;
    const model = modelSelect.value;
    const series = seriesSelect.value;
    const step = Number(stepSlider.value || 1);
    const rows = predictions.filter(row => row.dataset === dataset && row.unique_id === series);
    const current = rows.filter(row => row.model === model).sort((a, b) => a.ds.localeCompare(b.ds));
    if (!current.length) return addEmpty(node, "当前组合没有预测明细。");
    const w = 1100, h = 420, left = 70, right = 28, top = 44, bottom = 70;
    const [lo, hi] = extent(rows.flatMap(row => [row.y, row.y_hat]));
    const xAt = idx => left + scale(idx, 0, Math.max(1, current.length - 1), w - left - right);
    const yAt = value => h - bottom - scale(value, lo, hi, h - top - bottom);
    addTitle(node, `${{dataset}} | ${{series}} | step=${{step}}`);
    node.append(el("line", {{x1: left, y1: h - bottom, x2: w - right, y2: h - bottom, stroke: "#94a3b8"}}));
    node.append(el("line", {{x1: left, y1: top, x2: left, y2: h - bottom, stroke: "#94a3b8"}}));
    const actualPoints = current.map((row, idx) => `${{xAt(idx).toFixed(1)}},${{yAt(row.y).toFixed(1)}}`).join(" ");
    node.append(el("polyline", {{points: actualPoints, fill: "none", stroke: "#0f172a", "stroke-width": 3, class: "draw-line"}}));
    models.filter(m => rows.some(row => row.model === m)).forEach((m, mi) => {{
      const mr = rows.filter(row => row.model === m && row.step <= step).sort((a, b) => a.ds.localeCompare(b.ds));
      if (!mr.length) return;
      const pts = mr.map((row, idx) => `${{xAt(idx).toFixed(1)}},${{yAt(row.y_hat).toFixed(1)}}`).join(" ");
      node.append(el("polyline", {{points: pts, fill: "none", stroke: color(m), "stroke-width": m === model ? 2.8 : 1.8, opacity: m === model ? 0.95 : 0.34, class: "draw-line"}}));
      const lx = left + (mi % 4) * 230;
      const ly = h - 38 + Math.floor(mi / 4) * 16;
      node.append(el("circle", {{cx: lx, cy: ly, r: 5, fill: color(m), opacity: m === model ? 1 : 0.55, class: "animated"}}));
      node.append(el("text", {{x: lx + 10, y: ly + 4, class: "svg-axis"}}, m));
    }});
    node.append(el("line", {{x1: left, y1: h - 18, x2: left + 30, y2: h - 18, stroke: "#0f172a", "stroke-width": 3}}));
    node.append(el("text", {{x: left + 38, y: h - 14, class: "svg-axis"}}, "真实值"));
  }}

  function drawResidual() {{
    const node = svg("dyn-residual");
    clear(node);
    const dataset = datasetSelect.value;
    const model = modelSelect.value;
    const rows = predictions.filter(row => row.dataset === dataset && row.model === model);
    if (!rows.length) return addEmpty(node, "当前组合没有残差数据。");
    const residuals = rows.map(row => row.y - row.y_hat).filter(Number.isFinite);
    const [lo, hi] = extent(residuals);
    const bins = 18;
    const counts = Array.from({{length: bins}}, () => 0);
    residuals.forEach(value => {{
      const idx = Math.max(0, Math.min(bins - 1, Math.floor(scale(value, lo, hi, bins))));
      counts[idx] += 1;
    }});
    const w = 620, h = 420, left = 58, right = 28, top = 48, bottom = 58;
    const maxCount = Math.max(...counts, 1);
    addTitle(node, `${{dataset}} | ${{model}} 残差分布`);
    node.append(el("line", {{x1: left, y1: h - bottom, x2: w - right, y2: h - bottom, stroke: "#94a3b8"}}));
    node.append(el("line", {{x1: left, y1: top, x2: left, y2: h - bottom, stroke: "#94a3b8"}}));
    counts.forEach((count, idx) => {{
      const x = left + idx * ((w - left - right) / bins);
      const bw = (w - left - right) / bins - 3;
      const bh = scale(count, 0, maxCount, h - top - bottom);
      const bar = el("rect", {{x, y: h - bottom - bh, width: bw, height: bh, rx: 3, fill: color(model), opacity: 0.78, class: "animated"}});
      node.append(attachTooltip(bar, `<b>${{esc(model)}}</b><br>残差区间: ${{idx + 1}} / ${{bins}}<br>样本数: ${{count}}`));
    }});
    node.append(el("text", {{x: left, y: h - 20, class: "svg-axis"}}, lo.toFixed(3)));
    node.append(el("text", {{x: w - right - 46, y: h - 20, class: "svg-axis"}}, hi.toFixed(3)));
  }}

  function drawEfficiency() {{
    const node = svg("dyn-efficiency");
    clear(node);
    const metric = metricSelect.value;
    const rows = metrics.filter(row => Number.isFinite(Number(row[metric])));
    if (!rows.length) return addEmpty(node, "没有效率-精度数据。");
    const grouped = models.map(model => {{
      const subset = rows.filter(row => row.model === model);
      return {{
        model,
        x: mean(subset.map(row => Number(row.inference_time_s || 0))),
        y: mean(subset.map(row => Number(row[metric]))),
      }};
    }}).filter(row => Number.isFinite(row.x) && Number.isFinite(row.y));
    const w = 620, h = 420, left = 66, right = 32, top = 48, bottom = 66;
    const [xLo, xHi] = extent(grouped.map(row => row.x));
    const [yLo, yHi] = extent(grouped.map(row => row.y));
    const xAt = value => left + scale(value, xLo, xHi, w - left - right);
    const yAt = value => h - bottom - scale(value, yLo, yHi, h - top - bottom);
    addTitle(node, `效率-精度 | y=${{metric.toUpperCase()}}`);
    node.append(el("line", {{x1: left, y1: h - bottom, x2: w - right, y2: h - bottom, stroke: "#94a3b8"}}));
    node.append(el("line", {{x1: left, y1: top, x2: left, y2: h - bottom, stroke: "#94a3b8"}}));
    grouped.forEach(row => {{
      const x = xAt(row.x);
      const y = yAt(row.y);
      const point = el("circle", {{cx: x, cy: y, r: 8, fill: color(row.model), opacity: 0.82, class: "animated"}});
      node.append(attachTooltip(point, `<b>${{esc(row.model)}}</b><br>推理时间: ${{row.x.toFixed(4)}}s<br>${{metric.toUpperCase()}}: ${{row.y.toFixed(6)}}`));
      node.append(el("text", {{x: x + 10, y: y + 4, class: "svg-axis"}}, row.model));
    }});
    node.append(el("text", {{x: w / 2, y: h - 18, class: "svg-axis"}}, "平均推理时间（秒）"));
    node.append(el("text", {{x: 18, y: top - 12, class: "svg-axis"}}, metric.toUpperCase()));
  }}

  function drawBump() {{
    const node = svg("dyn-bump");
    clear(node);
    const metric = metricSelect.value;
    const datasetsWithMetric = datasets.filter(ds => metrics.some(row => row.dataset === ds && Number.isFinite(Number(row[metric]))));
    if (datasetsWithMetric.length < 2) return addEmpty(node, "至少需要两个数据集才能绘制排名轨迹。");
    const w = 620, h = 420, left = 84, right = 52, top = 48, bottom = 70;
    const maxRank = Math.max(1, models.length);
    const xAt = idx => left + scale(idx, 0, Math.max(1, datasetsWithMetric.length - 1), w - left - right);
    const yAt = rank => top + scale(rank, 1, maxRank, h - top - bottom);
    addTitle(node, `${{metric.toUpperCase()}} 数据集排名轨迹`);
    datasetsWithMetric.forEach((ds, idx) => {{
      const x = xAt(idx);
      node.append(el("line", {{x1: x, y1: top, x2: x, y2: h - bottom, stroke: "#e2e8f0"}}));
      node.append(el("text", {{x, y: h - 34, class: "svg-axis", "text-anchor": "middle"}}, ds.length > 10 ? ds.slice(0, 10) + "…" : ds));
    }});
    models.forEach(model => {{
      const coords = [];
      datasetsWithMetric.forEach((ds, idx) => {{
        const values = groupMean(metrics.filter(row => row.dataset === ds), "model", metric).sort((a, b) => lowerIsBetter(metric) ? a.value - b.value : b.value - a.value);
        const rank = values.findIndex(row => row.name === model) + 1;
        if (rank > 0) coords.push(`${{xAt(idx).toFixed(1)}},${{yAt(rank).toFixed(1)}}`);
      }});
      if (coords.length < 2) return;
      const line = el("polyline", {{points: coords.join(" "), fill: "none", stroke: color(model), "stroke-width": model === modelSelect.value ? 3 : 1.8, opacity: model === modelSelect.value ? 0.95 : 0.42, class: "draw-line"}});
      node.append(attachTooltip(line, `<b>${{esc(model)}}</b><br>跨数据集排名轨迹`));
    }});
    node.append(el("text", {{x: 20, y: yAt(1) + 4, class: "svg-axis"}}, "#1"));
  }}

  function drawAll() {{
    drawRanking();
    drawForecast();
    drawResidual();
    drawEfficiency();
    drawBump();
  }}
  metricSelect.addEventListener("change", drawAll);
  datasetSelect.addEventListener("change", () => {{ refreshSeries(); drawAll(); }});
  modelSelect.addEventListener("change", () => {{ refreshSeries(); drawAll(); }});
  seriesSelect.addEventListener("change", drawForecast);
  stepSlider.addEventListener("input", drawForecast);
  stepSlider.addEventListener("input", () => {{
    progressBar.style.width = `${{Math.round(Number(stepSlider.value || 1) / Number(stepSlider.max || 1) * 100)}}%`;
  }});
  playButton.addEventListener("click", () => {{
    if (timer) {{
      clearInterval(timer);
      timer = null;
      playButton.textContent = "播放";
      playButton.classList.remove("is-playing");
      return;
    }}
    playButton.textContent = "暂停";
    playButton.classList.add("is-playing");
    timer = setInterval(() => {{
      const next = Number(stepSlider.value || 1) + 1;
      stepSlider.value = next > Number(stepSlider.max || 1) ? 1 : next;
      progressBar.style.width = `${{Math.round(Number(stepSlider.value || 1) / Number(stepSlider.max || 1) * 100)}}%`;
      drawForecast();
    }}, 650);
  }});
  refreshSeries();
  drawAll();
}})();
</script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
