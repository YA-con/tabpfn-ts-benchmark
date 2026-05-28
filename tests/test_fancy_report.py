from pathlib import Path

from src.reporting.fancy_generator import generate_fancy_demo_report


def test_fancy_demo_report_generates_dashboard(tmp_path: Path) -> None:
    out = tmp_path / "fancy"
    report = generate_fancy_demo_report(
        "results/tabpfn_ts_a100_c96_h12",
        out,
        baseline_dirs=["results/pilot_baseline"],
        config_paths=["configs/config.yaml"],
        primary_metric="smape",
        theme_name="dark_premium",
    )

    html = report.read_text(encoding="utf-8")
    assert report.exists()
    assert (out / "metrics_demo.json").exists()
    assert (out / "summary.csv").exists()
    assert len(list((out / "assets").glob("*.svg"))) >= 5
    assert "AI Research Dashboard" in html
    assert "KPI 指标卡片" in html
    assert "Dashboard 图表区" in html
    assert "Demo 占位" in html
