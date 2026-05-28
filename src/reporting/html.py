"""HTML rendering for reusable experiment reports."""

from __future__ import annotations

import json
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
@media (max-width: 980px) {{
  .hero, main {{ padding-left: 16px; padding-right: 16px; }}
  .overview, .metric-grid, .figure-grid {{ grid-template-columns: 1fr; }}
  .figure-card img {{ min-width: 360px; }}
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
  <section><h2>核心指标卡片</h2>{_metric_cards(summaries)}</section>
  <section><h2>Baseline 对比</h2><div class="table-wrap">{_table(comparison, "没有提供 baseline，或 baseline 指标不可用。")}</div></section>
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
</script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
