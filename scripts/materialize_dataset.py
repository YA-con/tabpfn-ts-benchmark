"""Materialize one registered dataset to data/processed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.materialize import materialize_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Registered dataset name.")
    parser.add_argument("--project-root", default=".", help="Project root used to resolve paths.")
    parser.add_argument("--output-root", default="data/processed", help="Processed output root.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files for quick checks.")
    return parser.parse_args()


def main() -> None:
    """Materialize the requested dataset and print a concise summary."""

    args = parse_args()
    summary = materialize_dataset(
        name=args.dataset,
        project_root=args.project_root,
        output_root=args.output_root,
        max_files=args.max_files,
    )
    print(
        f"{summary.name}: series={summary.n_series} rows={summary.n_observations} "
        f"range={summary.min_ds}..{summary.max_ds}"
    )


if __name__ == "__main__":
    main()
