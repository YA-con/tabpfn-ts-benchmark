"""Smoke experiment subprocess test."""

import subprocess
from pathlib import Path


def test_smoke_experiment_runs() -> None:
    """The smoke script exits cleanly and writes a metrics CSV."""

    project_root = Path(__file__).resolve().parents[1]
    output_csv = project_root / "results" / "e1_smoke" / "metrics.csv"
    if output_csv.exists():
        output_csv.unlink()

    result = subprocess.run(
        ["bash", "scripts/run_smoke_test.sh"],
        cwd=project_root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_csv.exists()
