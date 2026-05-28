#!/usr/bin/env python
"""Generate a reusable HTML report for one experiment run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting import generate_experiment_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Directory containing forecast_metrics.csv.")
    parser.add_argument("--out", required=True, help="Output report directory.")
    parser.add_argument(
        "--baseline-dir",
        action="append",
        default=[],
        help="Baseline run directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Optional config YAML/JSON file to embed. Can be passed multiple times.",
    )
    parser.add_argument("--primary-metric", default="smape", help="Metric used for best-model comparison.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = generate_experiment_report(
        run_dir=args.run_dir,
        out_dir=args.out,
        baseline_dirs=args.baseline_dir,
        config_paths=args.config,
        primary_metric=args.primary_metric,
    )
    print(path)


if __name__ == "__main__":
    main()
