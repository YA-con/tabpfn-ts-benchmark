"""Inspect registered datasets and optionally summarize loadable ones."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import list_dataset_specs, summarize_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project root used to resolve data paths.")
    parser.add_argument("--dataset", default=None, help="Inspect one dataset by registry name.")
    parser.add_argument("--load", action="store_true", help="Load datasets and print summaries.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files for provided market data.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a requested dataset cannot be loaded.",
    )
    return parser.parse_args()


def main() -> None:
    """Print dataset registry entries and optional summaries."""

    args = parse_args()
    project_root = Path(args.project_root)
    specs = list_dataset_specs()
    if args.dataset is not None:
        specs = [spec for spec in specs if spec.name == args.dataset]
        if not specs:
            raise KeyError(f"Unknown dataset {args.dataset!r}.")
    for spec in specs:
        print(f"{spec.name}\t{spec.domain}\t{spec.freq}\t{spec.storage}")
        if args.load:
            try:
                summary = summarize_dataset(spec.name, project_root=project_root, max_files=args.max_files)
            except FileNotFoundError as exc:
                if args.strict:
                    raise
                print(f"  skipped: {exc}")
                continue
            print(f"  series={summary.n_series} rows={summary.n_observations}")


if __name__ == "__main__":
    main()
