#!/usr/bin/env python
"""Generate an offline animated model-comparison dashboard."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

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


def _load_frames(results_glob: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    prediction_frames = []
    for metrics_path in sorted(PROJECT_ROOT.glob(results_glob)):
        run_dir = metrics_path.parent
        run = run_dir.name
        context, horizon, scenario = _scenario_from_run(run)
        metrics = pd.read_csv(metrics_path)
        metrics["matrix_run"] = run
        metrics["context"] = context
        metrics["horizon"] = horizon
        metrics["scenario"] = scenario
        metric_frames.append(metrics)
        pred_path = run_dir / "forecast_predictions.csv"
        if pred_path.exists():
            pred = pd.read_csv(pred_path)
            pred["matrix_run"] = run
            pred["context"] = context
            pred["horizon"] = horizon
            pred["scenario"] = scenario
            prediction_frames.append(pred)
    if not metric_frames:
        raise FileNotFoundError(f"No metrics matched: {results_glob}")
    return pd.concat(metric_frames, ignore_index=True), pd.concat(prediction_frames, ignore_index=True)


def _run_label_from_glob(results_glob: str) -> str:
    marker = "results/"
    if marker in results_glob:
        return results_glob.split(marker, 1)[1].split("/")[0].replace("*", "matrix")
    return Path(results_glob).parts[1] if len(Path(results_glob).parts) > 1 else "benchmark"


def _build_payload(metrics: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    metrics = metrics.copy()
    metrics["task"] = metrics["matrix_run"] + "|" + metrics["dataset"]
    scenario_meta = (
        metrics[["matrix_run", "scenario", "context", "horizon"]]
        .drop_duplicates()
        .sort_values(["context", "horizon"])
    )
    scenarios = scenario_meta.to_dict(orient="records")
    scenario_order = scenario_meta["matrix_run"].tolist()
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
    model_order = (
        unit.groupby("model")
        .agg(avg_rank=("rank", "mean"), mean_smape=("smape", "mean"))
        .sort_values(["avg_rank", "mean_smape"])
        .index.tolist()
    )
    scenario_rank = (
        unit.groupby(["matrix_run", "scenario", "context", "horizon", "model"], as_index=False)
        .agg(smape=("smape", "mean"), inference_time_s=("inference_time_s", "mean"))
    )
    scenario_rank["rank"] = scenario_rank.groupby("matrix_run")["smape"].rank(method="min", ascending=True).astype(int)
    rank_series = [
        {
            "model": model,
            "short": SHORT_NAMES.get(model, model),
            "color": MODEL_COLORS.get(model, "#2563eb"),
            "points": [
                {
                    "scenario": row["scenario"],
                    "matrix_run": row["matrix_run"],
                    "rank": int(row["rank"]),
                    "smape": float(row["smape"]),
                }
                for _, row in scenario_rank[scenario_rank["model"] == model]
                .set_index("matrix_run")
                .reindex(scenario_order)
                .reset_index()
                .dropna(subset=["model"])
                .iterrows()
            ],
        }
        for model in model_order
    ]

    best_non = (
        unit[unit["model"] != "tabpfn_ts"]
        .sort_values("smape")
        .groupby("task", as_index=False)
        .first()[["task", "model", "smape"]]
        .rename(columns={"model": "best_baseline_model", "smape": "best_baseline_smape"})
    )
    tab = unit[unit["model"] == "tabpfn_ts"][
        ["task", "matrix_run", "scenario", "context", "horizon", "dataset", "smape"]
    ].rename(columns={"smape": "tabpfn_smape"})
    delta = tab.merge(best_non, on="task")
    delta["delta"] = delta["tabpfn_smape"] - delta["best_baseline_smape"]
    paired_delta = delta.to_dict(orient="records")

    heatmap = (
        scenario_rank[["matrix_run", "scenario", "model", "rank", "smape"]]
        .sort_values(["matrix_run", "rank"])
        .to_dict(orient="records")
    )
    pareto = scenario_rank[["matrix_run", "scenario", "model", "rank", "smape", "inference_time_s"]].to_dict(
        orient="records"
    )

    winners = (
        unit.loc[unit.groupby("task")["smape"].idxmin()][["task", "matrix_run", "scenario", "dataset", "model", "smape"]]
        .sort_values(["matrix_run", "dataset"])
        .to_dict(orient="records")
    )

    residual_bins: dict[str, list[dict[str, Any]]] = {}
    forecast_rows: list[dict[str, Any]] = []
    if not predictions.empty:
        predictions = predictions.copy()
        predictions["residual"] = predictions["y"] - predictions["y_hat"]
        for scenario in scenario_order:
            sub = predictions[predictions["matrix_run"] == scenario]
            if sub.empty:
                continue
            values = sub["residual"].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            lo, hi = np.percentile(values, [1, 99])
            if lo == hi:
                lo, hi = values.min(), values.max()
            bins = np.linspace(lo, hi, 27)
            rows = []
            for model, group in sub.groupby("model"):
                counts, edges = np.histogram(group["residual"].dropna().to_numpy(dtype=float), bins=bins, density=True)
                rows.append(
                    {
                        "model": model,
                        "short": SHORT_NAMES.get(model, model),
                        "color": MODEL_COLORS.get(model, "#2563eb"),
                        "bins": [
                            {"x0": float(edges[i]), "x1": float(edges[i + 1]), "density": float(counts[i])}
                            for i in range(len(counts))
                        ],
                    }
                )
            residual_bins[scenario] = rows

        preferred_run = scenario_order[min(1, len(scenario_order) - 1)]
        sub = predictions[predictions["matrix_run"] == preferred_run]
        preferred_dataset = "synthetic_energy" if "synthetic_energy" in set(sub["dataset"]) else str(sub["dataset"].iloc[0])
        sub = sub[sub["dataset"] == preferred_dataset]
        preferred_series = str(sub["unique_id"].iloc[0])
        sub = sub[sub["unique_id"] == preferred_series].copy()
        sub["ds"] = sub["ds"].astype(str)
        forecast_rows = sub[["scenario", "dataset", "model", "unique_id", "ds", "y", "y_hat"]].to_dict(orient="records")

    return _json_safe(
        {
            "scenarios": scenarios,
            "models": [
                {"id": model, "short": SHORT_NAMES.get(model, model), "color": MODEL_COLORS.get(model, "#2563eb")}
                for model in model_order
            ],
            "rankSeries": rank_series,
            "pairedDelta": paired_delta,
            "heatmap": heatmap,
            "pareto": pareto,
            "winners": winners,
            "residualBins": residual_bins,
            "forecastRows": forecast_rows,
        }
    )


def _render_html(payload: dict[str, Any], output: Path, run_label: str) -> None:
    data_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{run_label} · Animated Model Dashboard</title>
<style>
:root {{
  --bg: #f6f8fb;
  --panel: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --line: #dbe4ef;
  --accent: #0f766e;
  --danger: #dc2626;
  --good: #059669;
  --blue: #2563eb;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif; }}
main {{ max-width: 1540px; margin: 0 auto; padding: 34px; }}
.hero {{ display: grid; grid-template-columns: 1.2fr .8fr; gap: 28px; align-items: end; padding: 28px 0 24px; border-bottom: 1px solid var(--line); }}
.badge {{ display: inline-flex; width: fit-content; padding: 6px 10px; border-radius: 999px; background: #ecfdf5; color: #14532d; font-weight: 900; font-size: 12px; margin-bottom: 12px; }}
h1 {{ margin: 0 0 12px; font-size: 46px; line-height: 1.05; letter-spacing: 0; }}
.sub {{ color: var(--muted); line-height: 1.72; font-size: 15px; }}
.note {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: 0 14px 42px rgba(15,23,42,.055); line-height: 1.7; color: var(--muted); }}
.controls {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 22px 0 4px; }}
button, select {{ border: 1px solid var(--line); background: var(--panel); color: var(--text); border-radius: 999px; height: 38px; padding: 0 14px; font-weight: 900; cursor: pointer; }}
button.primary {{ background: var(--accent); color: white; border-color: var(--accent); }}
.grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 22px; margin-top: 22px; }}
.card {{ grid-column: span 6; background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: 0 14px 42px rgba(15,23,42,.055); overflow: hidden; }}
.card.wide {{ grid-column: span 12; }}
.card-head {{ display: flex; align-items: start; justify-content: space-between; gap: 16px; margin-bottom: 12px; }}
h2 {{ margin: 0 0 5px; font-size: 18px; }}
.caption {{ margin: 0; color: var(--muted); font-size: 13px; line-height: 1.55; }}
svg {{ width: 100%; height: auto; display: block; background: #fbfcfe; border-radius: 12px; border: 1px solid #edf2f7; }}
.axis {{ stroke: #cbd5e1; stroke-width: 1; }}
.gridline {{ stroke: #e5ebf3; stroke-width: 1; }}
.tick {{ fill: #64748b; font-size: 12px; }}
.label {{ fill: #0f172a; font-size: 12px; font-weight: 800; }}
.small {{ fill: #64748b; font-size: 11px; }}
.tooltip {{ position: fixed; pointer-events: none; opacity: 0; background: rgba(15,23,42,.94); color: white; border-radius: 10px; padding: 9px 10px; font-size: 12px; z-index: 10; box-shadow: 0 18px 45px rgba(15,23,42,.24); transition: opacity .12s ease; max-width: 280px; }}
.tooltip.show {{ opacity: 1; }}
.progress {{ height: 8px; border-radius: 999px; background: #e2e8f0; overflow: hidden; margin-top: 10px; }}
.progress span {{ display: block; height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent), #38bdf8); transition: width .2s ease; }}
@media (max-width: 980px) {{ main {{ padding: 16px; }} .hero {{ grid-template-columns: 1fr; }} .card, .card.wide {{ grid-column: span 12; }} h1 {{ font-size: 32px; }} }}
</style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <span class="badge">animated · all charts</span>
      <h1>动态图表版模型对比</h1>
      <p class="sub">这里把适合动的 7 类图都做出来：排名轨迹、成对差值、预测播放、热力图 reveal、Pareto 轨迹、残差分布切换、胜场累积。动画只服务于“变化过程”，不做无意义装饰。</p>
    </div>
    <div class="note"><b>数据口径</b><br/>全部来自 {run_label} 的真实 metrics/predictions CSV。每个动画都可以暂停，避免为了动而牺牲读数。</div>
  </section>
  <div class="controls">
    <button class="primary" id="play">播放</button>
    <button id="reset">重置</button>
    <select id="scenario"></select>
  </div>
  <div class="progress"><span id="progress"></span></div>
  <section class="grid">
    <article class="card wide">
      <div class="card-head"><div><h2>1. Animated Rank Trajectory</h2><p class="caption">模型排名随 context/horizon 变化。线越靠上排名越好。</p></div></div>
      <svg id="rank" viewBox="0 0 1280 460"></svg>
    </article>
    <article class="card">
      <div class="card-head"><div><h2>2. Animated Paired Delta</h2><p class="caption">TabPFN-TS 与该 task 最强非 TabPFN baseline 的差值，低于 0 表示 TabPFN-TS 更好。</p></div></div>
      <svg id="delta" viewBox="0 0 620 420"></svg>
    </article>
    <article class="card">
      <div class="card-head"><div><h2>3. Forecast Playback</h2><p class="caption">真实曲线固定，预测线按 horizon step 展开。</p></div></div>
      <svg id="forecast" viewBox="0 0 620 420"></svg>
    </article>
    <article class="card">
      <div class="card-head"><div><h2>4. Heatmap Reveal</h2><p class="caption">Run × Model 排名热力图按场景逐步出现。</p></div></div>
      <svg id="heatmap" viewBox="0 0 620 420"></svg>
    </article>
    <article class="card">
      <div class="card-head"><div><h2>5. Efficiency / Performance Pareto</h2><p class="caption">点随场景移动，显示推理时间和排名的 trade-off。</p></div></div>
      <svg id="pareto" viewBox="0 0 620 420"></svg>
    </article>
    <article class="card">
      <div class="card-head"><div><h2>6. Residual Distribution Morph</h2><p class="caption">残差分布随场景切换，观察误差是否集中或偏移。</p></div></div>
      <svg id="residual" viewBox="0 0 620 420"></svg>
    </article>
    <article class="card">
      <div class="card-head"><div><h2>7. Cumulative Wins Race</h2><p class="caption">按 task 顺序累计第一名次数，展示谁是持续赢。</p></div></div>
      <svg id="wins" viewBox="0 0 620 420"></svg>
    </article>
  </section>
</main>
<script id="payload" type="application/json">{data_json}</script>
<script>
const data = JSON.parse(document.getElementById('payload').textContent);
const scenarios = data.scenarios;
const models = data.models;
const modelMap = new Map(models.map(m => [m.id, m]));
let frame = 0;
let playing = false;
let timer = null;
const totalFrames = Math.max(1, scenarios.length * 24);
const $ = id => document.getElementById(id);
const scenarioSelect = $('scenario');
scenarioSelect.innerHTML = scenarios.map((s,i) => `<option value="${{i}}">${{s.scenario}}</option>`).join('');

function esc(s) {{ return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;'); }}
function svgEl(name, attrs={{}}, text='') {{
  const n = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text) n.textContent = text;
  return n;
}}
function clear(node) {{ while (node.firstChild) node.removeChild(node.firstChild); }}
function extent(vals, fallback=[0,1]) {{
  const v = vals.map(Number).filter(Number.isFinite);
  if (!v.length) return fallback;
  let lo = Math.min(...v), hi = Math.max(...v);
  if (lo === hi) {{ lo -= 1; hi += 1; }}
  return [lo, hi];
}}
function scale(v, lo, hi, outLo, outHi) {{ return outLo + (v - lo) / ((hi - lo) || 1) * (outHi - outLo); }}
function tip() {{ let t=document.querySelector('.tooltip'); if(!t) {{t=document.createElement('div'); t.className='tooltip'; document.body.appendChild(t);}} return t; }}
function hover(node, html) {{
  node.addEventListener('mousemove', e => {{ const t=tip(); t.innerHTML=html; t.style.left=`${{e.clientX+14}}px`; t.style.top=`${{e.clientY+14}}px`; t.classList.add('show'); }});
  node.addEventListener('mouseleave', () => tip().classList.remove('show'));
  return node;
}}
function currentScenarioIndex() {{ return Math.min(scenarios.length - 1, Math.floor(frame / 24)); }}
function currentProgress() {{ return (frame % 24) / 23; }}
function drawAxes(node, left, top, width, height, xLabel='', yLabel='') {{
  node.append(svgEl('line', {{x1:left,y1:top+height,x2:left+width,y2:top+height,class:'axis'}}));
  node.append(svgEl('line', {{x1:left,y1:top,x2:left,y2:top+height,class:'axis'}}));
  if (xLabel) node.append(svgEl('text', {{x:left+width/2,y:top+height+42,class:'small','text-anchor':'middle'}}, xLabel));
  if (yLabel) node.append(svgEl('text', {{x:left-48,y:top-12,class:'small'}}, yLabel));
}}

function drawRank() {{
  const node = $('rank'); clear(node);
  const w=1280,h=460,left=150,right=44,top=48,bottom=82;
  const idx=currentScenarioIndex(), p=currentProgress();
  const xs=scenarios.map((s,i)=>scale(i,0,Math.max(1,scenarios.length-1),left,w-right));
  for (let r=1;r<=models.length;r++) {{
    const y=scale(r,1,models.length,top,h-bottom);
    node.append(svgEl('line',{{x1:left,y1:y,x2:w-right,y2:y,class:'gridline'}}));
    node.append(svgEl('text',{{x:left-18,y:y+4,class:'tick','text-anchor':'end'}}, '#'+r));
  }}
  scenarios.forEach((s,i)=>node.append(svgEl('text',{{x:xs[i],y:h-38,class:'tick','text-anchor':'middle'}}, s.scenario)));
  drawAxes(node,left,top,w-left-right,h-top-bottom,'scenario','rank');
  data.rankSeries.forEach(series => {{
    const pts=[];
    for (let i=0;i<=idx;i++) {{
      const pt=series.points[i]; if(!pt) continue;
      pts.push([xs[i], scale(pt.rank,1,models.length,top,h-bottom)]);
    }}
    if (idx < series.points.length - 1 && series.points[idx] && series.points[idx+1]) {{
      const a=series.points[idx], b=series.points[idx+1];
      pts.push([scale(idx+p,0,Math.max(1,scenarios.length-1),left,w-right), scale(a.rank+(b.rank-a.rank)*p,1,models.length,top,h-bottom)]);
    }}
    if (pts.length) {{
      const line=svgEl('polyline',{{points:pts.map(x=>x.join(',')).join(' '),fill:'none',stroke:series.color,'stroke-width':series.model==='tabpfn_ts'?4:2,opacity:series.model==='tabpfn_ts'?1:.55}});
      node.append(hover(line, `<b>${{esc(series.short)}}</b><br>排名轨迹`));
      const last=pts[pts.length-1];
      node.append(svgEl('circle',{{cx:last[0],cy:last[1],r:series.model==='tabpfn_ts'?6:4,fill:series.color,opacity:.95}}));
      if (series.model==='tabpfn_ts' || ['lightgbm_ar','xgboost_ar','ridge_ar'].includes(series.model)) {{
        node.append(svgEl('text',{{x:last[0]+8,y:last[1]+4,class:'label',fill:series.color}}, series.short));
      }}
    }}
  }});
}}

function drawDelta() {{
  const node=$('delta'); clear(node);
  const w=620,h=420,left=64,right=28,top=44,bottom=64, idx=currentScenarioIndex();
  const sc=scenarios[idx]?.matrix_run;
  const rows=data.pairedDelta.filter(d=>d.matrix_run===sc);
  const vals=data.pairedDelta.map(d=>d.delta);
  const [lo,hi]=extent(vals,[-1,1]);
  drawAxes(node,left,top,w-left-right,h-top-bottom,'datasets','delta');
  const zero=scale(0,lo,hi,h-bottom,top);
  node.append(svgEl('line',{{x1:left,y1:zero,x2:w-right,y2:zero,stroke:'#0f172a','stroke-width':1.2}}));
  node.append(svgEl('text',{{x:left,y:26,class:'label'}}, `${{scenarios[idx]?.scenario}} · TabPFN vs best baseline`));
  rows.forEach((r,i)=>{{
    const x=scale(i,0,Math.max(1,rows.length-1),left+20,w-right-20);
    const y=scale(r.delta,lo,hi,h-bottom,top);
    const c=r.delta<=0?'#059669':'#dc2626';
    const point=svgEl('circle',{{cx:x,cy:y,r:8,fill:c,opacity:.9,stroke:'#fff','stroke-width':1.2}});
    node.append(hover(point, `<b>${{esc(r.dataset)}}</b><br>delta: ${{r.delta.toFixed(4)}}<br>baseline: ${{esc(r.best_baseline_model)}}`));
    node.append(svgEl('text',{{x:x,y:h-34,class:'tick','text-anchor':'middle'}}, r.dataset.replace('synthetic_','').slice(0,8)));
  }});
}}

function drawForecast() {{
  const node=$('forecast'); clear(node);
  const rows=data.forecastRows || [];
  if (!rows.length) {{ node.append(svgEl('text',{{x:24,y:42,class:'label'}},'没有预测明细')); return; }}
  const w=620,h=420,left=58,right=24,top=44,bottom=66;
  const p=(frame % 48)/47;
  const byModel = new Map();
  rows.forEach(r=>{{ if(!byModel.has(r.model)) byModel.set(r.model,[]); byModel.get(r.model).push(r); }});
  byModel.forEach(v=>v.sort((a,b)=>String(a.ds).localeCompare(String(b.ds))));
  const base=[...byModel.values()][0];
  const maxStep=Math.max(2, Math.floor(base.length*p));
  const vals=rows.flatMap(r=>[r.y,r.y_hat]).map(Number);
  const [lo,hi]=extent(vals);
  const xAt=i=>scale(i,0,Math.max(1,base.length-1),left,w-right);
  const yAt=v=>scale(v,lo,hi,h-bottom,top);
  drawAxes(node,left,top,w-left-right,h-top-bottom,'horizon step','value');
  node.append(svgEl('text',{{x:left,y:26,class:'label'}}, `${{base[0].scenario}} · ${{base[0].dataset}} · ${{base[0].unique_id}}`));
  const actualPts=base.slice(0,maxStep).map((r,i)=>`${{xAt(i)}},${{yAt(Number(r.y))}}`).join(' ');
  node.append(svgEl('polyline',{{points:actualPts,fill:'none',stroke:'#0f172a','stroke-width':3}}));
  let mi=0;
  byModel.forEach((series,model)=>{{
    const m=modelMap.get(model) || {{short:model,color:'#2563eb'}};
    const pts=series.slice(0,maxStep).map((r,i)=>`${{xAt(i)}},${{yAt(Number(r.y_hat))}}`).join(' ');
    const line=svgEl('polyline',{{points:pts,fill:'none',stroke:m.color,'stroke-width':model==='tabpfn_ts'?2.8:1.6,opacity:model==='tabpfn_ts'?1:.38}});
    node.append(hover(line, `<b>${{esc(m.short)}}</b><br>forecast playback`));
    if (model==='tabpfn_ts' || mi<3) {{
      node.append(svgEl('circle',{{cx:382+(mi%3)*72,cy:22+Math.floor(mi/3)*16,r:4,fill:m.color}}));
      node.append(svgEl('text',{{x:390+(mi%3)*72,y:26+Math.floor(mi/3)*16,class:'small'}}, m.short));
    }}
    mi++;
  }});
}}

function drawHeatmap() {{
  const node=$('heatmap'); clear(node);
  const w=620,h=420,left=118,right=20,top=54,bottom=58, idx=currentScenarioIndex();
  const visible=scenarios.slice(0,idx+1);
  const cellW=(w-left-right)/models.length, cellH=(h-top-bottom)/scenarios.length;
  node.append(svgEl('text',{{x:left,y:28,class:'label'}}, 'Run × Model rank reveal'));
  models.forEach((m,c)=>node.append(svgEl('text',{{x:left+c*cellW+cellW/2,y:h-28,class:'tick','text-anchor':'middle',transform:`rotate(-28 ${{left+c*cellW+cellW/2}} ${{h-28}})`}}, m.short.slice(0,9))));
  visible.forEach((s,r)=>{{
    node.append(svgEl('text',{{x:left-10,y:top+r*cellH+cellH/2+4,class:'tick','text-anchor':'end'}}, s.scenario));
    models.forEach((m,c)=>{{
      const row=data.heatmap.find(x=>x.matrix_run===s.matrix_run && x.model===m.id);
      const rank=row ? Number(row.rank) : 9;
      const hue=155-(rank-1)*15;
      const rect=svgEl('rect',{{x:left+c*cellW+2,y:top+r*cellH+2,width:cellW-4,height:cellH-4,rx:7,fill:`hsl(${{hue}},62%,${{rank<=3?42:70}}%)`,opacity:.9}});
      node.append(hover(rect, `<b>${{esc(m.short)}}</b><br>${{s.scenario}}<br>rank #${{rank}}`));
      node.append(svgEl('text',{{x:left+c*cellW+cellW/2,y:top+r*cellH+cellH/2+4,class:'label','text-anchor':'middle',fill:rank<=3?'#fff':'#0f172a'}}, String(rank)));
    }});
  }});
}}

function drawPareto() {{
  const node=$('pareto'); clear(node);
  const w=620,h=420,left=64,right=30,top=46,bottom=62, idx=currentScenarioIndex();
  const sc=scenarios[idx]?.matrix_run;
  const rows=data.pareto.filter(d=>d.matrix_run===sc);
  const allX=data.pareto.map(d=>Math.log10(Number(d.inference_time_s)+1e-4));
  const [xLo,xHi]=extent(allX), yLo=1, yHi=models.length;
  drawAxes(node,left,top,w-left-right,h-top-bottom,'log inference time','rank');
  node.append(svgEl('text',{{x:left,y:28,class:'label'}}, `${{scenarios[idx]?.scenario}} · Pareto movement`));
  rows.forEach(r=>{{
    const m=modelMap.get(r.model) || {{short:r.model,color:'#2563eb'}};
    const x=scale(Math.log10(Number(r.inference_time_s)+1e-4),xLo,xHi,left,w-right);
    const y=scale(Number(r.rank),yLo,yHi,top,h-bottom);
    const point=svgEl('circle',{{cx:x,cy:y,r:r.model==='tabpfn_ts'?10:7,fill:m.color,opacity:.86,stroke:'#fff','stroke-width':1.2}});
    node.append(hover(point, `<b>${{esc(m.short)}}</b><br>rank #${{r.rank}}<br>SMAPE ${{Number(r.smape).toFixed(4)}}<br>time ${{Number(r.inference_time_s).toFixed(4)}}s`));
    if (['tabpfn_ts','lightgbm_ar','xgboost_ar','ridge_ar'].includes(r.model)) node.append(svgEl('text',{{x:x+9,y:y+4,class:'small'}}, m.short));
  }});
}}

function drawResidual() {{
  const node=$('residual'); clear(node);
  const w=620,h=420,left=58,right=28,top=48,bottom=62, idx=currentScenarioIndex();
  const sc=scenarios[idx]?.matrix_run;
  const rows=data.residualBins[sc] || [];
  const densities=rows.flatMap(r=>r.bins.map(b=>b.density));
  const xs=rows.flatMap(r=>r.bins.flatMap(b=>[b.x0,b.x1]));
  const [xLo,xHi]=extent(xs,[-1,1]), [_,yHi]=extent(densities,[0,1]);
  drawAxes(node,left,top,w-left-right,h-top-bottom,'residual','density');
  node.append(svgEl('text',{{x:left,y:28,class:'label'}}, `${{scenarios[idx]?.scenario}} · residual distributions`));
  rows.filter(r=>['tabpfn_ts','lightgbm_ar','xgboost_ar','ridge_ar','hist_gradient_boosting_ar'].includes(r.model)).forEach(r=>{{
    const pts=r.bins.map(b=>[scale((b.x0+b.x1)/2,xLo,xHi,left,w-right), scale(b.density,0,yHi,h-bottom,top)]);
    const d='M '+pts.map(p=>p.join(',')).join(' L ');
    const path=svgEl('path',{{d,fill:'none',stroke:r.color,'stroke-width':r.model==='tabpfn_ts'?3:1.8,opacity:r.model==='tabpfn_ts'?1:.65}});
    node.append(hover(path, `<b>${{esc(r.short)}}</b><br>residual density`));
  }});
}}

function drawWins() {{
  const node=$('wins'); clear(node);
  const w=620,h=420,left=142,right=34,top=48,bottom=54;
  const maxTasks=data.winners.length;
  const upto=Math.max(1, Math.floor(maxTasks * (frame / Math.max(1,totalFrames-1))));
  const counts=new Map(models.map(m=>[m.id,0]));
  data.winners.slice(0,upto).forEach(w=>counts.set(w.model,(counts.get(w.model)||0)+1));
  const ordered=[...counts.entries()].sort((a,b)=>b[1]-a[1]);
  const max=Math.max(1,...ordered.map(x=>x[1]));
  node.append(svgEl('text',{{x:left,y:28,class:'label'}}, `cumulative wins · ${{upto}} / ${{maxTasks}} tasks`));
  ordered.forEach(([model,count],i)=>{{
    const m=modelMap.get(model) || {{short:model,color:'#2563eb'}};
    const y=top+i*34, bw=scale(count,0,max,0,w-left-right);
    node.append(svgEl('text',{{x:left-10,y:y+20,class:'tick','text-anchor':'end'}}, m.short));
    const bar=svgEl('rect',{{x:left,y:y+4,width:bw,height:22,rx:11,fill:m.color,opacity:.88}});
    node.append(hover(bar, `<b>${{esc(m.short)}}</b><br>wins: ${{count}}`));
    node.append(svgEl('text',{{x:left+bw+8,y:y+20,class:'label'}}, String(count)));
  }});
}}

function drawAll() {{
  $('progress').style.width = `${{Math.round(frame/(totalFrames-1)*100)}}%`;
  scenarioSelect.value = String(currentScenarioIndex());
  drawRank(); drawDelta(); drawForecast(); drawHeatmap(); drawPareto(); drawResidual(); drawWins();
}}
function play() {{
  playing = !playing;
  $('play').textContent = playing ? '暂停' : '播放';
  if (timer) clearInterval(timer);
  if (playing) timer = setInterval(() => {{ frame = (frame + 1) % totalFrames; drawAll(); }}, 620);
}}
$('play').addEventListener('click', play);
$('reset').addEventListener('click', () => {{ frame=0; playing=false; if(timer) clearInterval(timer); $('play').textContent='播放'; drawAll(); }});
scenarioSelect.addEventListener('change', () => {{ frame = Number(scenarioSelect.value) * 24; drawAll(); }});
drawAll();
</script>
</body>
</html>"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "animated_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-glob", default="results/full_benchmark_v1_c*_h*_s2/forecast_metrics.csv")
    parser.add_argument("--out", default="reports/full_benchmark_v1_animated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, predictions = _load_frames(args.results_glob)
    payload = _build_payload(metrics, predictions)
    _render_html(payload, PROJECT_ROOT / args.out, _run_label_from_glob(args.results_glob))
    print(PROJECT_ROOT / args.out / "index.html")


if __name__ == "__main__":
    main()
