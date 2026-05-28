#!/usr/bin/env python
"""Generate fancy dashboard demo reports from existing run artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.fancy_generator import generate_fancy_demo_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/tabpfn_ts_a100_c96_h12")
    parser.add_argument("--baseline-dir", action="append", default=["results/pilot_baseline"])
    parser.add_argument("--config", action="append", default=["configs/config.yaml"])
    parser.add_argument("--primary-metric", default="smape")
    parser.add_argument(
        "--theme",
        action="append",
        default=None,
        choices=["dark_premium", "clean_premium"],
        help="Theme(s) to render. Defaults to both themes.",
    )
    parser.add_argument("--out-root", default="reports/demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    themes = args.theme or ["dark_premium", "clean_premium"]
    for theme in themes:
        name = "fancy_dark" if theme == "dark_premium" else "fancy_clean"
        path = generate_fancy_demo_report(
            args.run_dir,
            f"{args.out_root}/{name}",
            baseline_dirs=args.baseline_dir,
            config_paths=args.config,
            primary_metric=args.primary_metric,
            theme_name=theme,
            is_demo=True,
        )
        print(path)
    dark_path = generate_fancy_demo_report(
        args.run_dir,
        f"{args.out_root}/fancy_v1",
        baseline_dirs=args.baseline_dir,
        config_paths=args.config,
        primary_metric=args.primary_metric,
        theme_name="dark_premium",
        is_demo=True,
    )
    print(dark_path)


if __name__ == "__main__":
    main()
