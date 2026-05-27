"""Fetch a public benchmark dataset into data/raw."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import PUBLIC_BENCHMARK_URLS, download_public_benchmark


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=sorted(PUBLIC_BENCHMARK_URLS),
        help="Public benchmark name to fetch.",
    )
    parser.add_argument("--project-root", default=".", help="Project root used to resolve data paths.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing raw file.")
    return parser.parse_args()


def main() -> None:
    """Fetch the requested public benchmark."""

    args = parse_args()
    path = download_public_benchmark(
        name=args.dataset,
        project_root=args.project_root,
        overwrite=args.overwrite,
    )
    print(path)


if __name__ == "__main__":
    main()
