"""Dataset acquisition helpers.

Large raw datasets should not be committed to git. This module records the public
benchmark targets and downloads them explicitly into data/raw when requested.
"""

from __future__ import annotations

from pathlib import Path
from time import sleep
from urllib.request import urlretrieve

from src.data.registry import get_dataset_spec, list_dataset_specs


PUBLIC_BENCHMARK_URLS = {
    "etth1": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh1.csv"
    ],
    "electricity": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv"
    ],
    "exchange_rate": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/exchange_rate/exchange_rate.csv",
        "https://huggingface.co/datasets/dunzane/time-series-dataset/resolve/main/exchange_rate/exchange_rate.csv",
        "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/exchange_rate.csv",
    ],
    "traffic": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv"
    ],
    "weather": [
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv",
        "https://huggingface.co/datasets/dunzane/time-series-dataset/resolve/main/weather/weather.csv",
        "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/weather.csv",
    ],
}


def print_dataset_manifest() -> None:
    """Print registered datasets and expected local storage locations."""

    for spec in list_dataset_specs():
        print(f"{spec.name}\t{spec.domain}\t{spec.storage}")


def download_public_benchmark(
    name: str,
    project_root: str | Path = ".",
    overwrite: bool = False,
    retries: int = 3,
) -> Path:
    """Download one registered public benchmark into its expected raw path."""

    if name not in PUBLIC_BENCHMARK_URLS:
        raise KeyError(f"No public download URL registered for {name!r}.")
    spec = get_dataset_spec(name)
    if spec.path is None:
        raise ValueError(f"{name} does not define a raw path.")

    output_path = Path(project_root).resolve() / spec.path
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in PUBLIC_BENCHMARK_URLS[name]:
        for attempt in range(1, retries + 1):
            try:
                urlretrieve(url, output_path)
                return output_path
            except Exception as exc:
                last_error = exc
                if output_path.exists():
                    output_path.unlink()
                if attempt < retries:
                    sleep(2 * attempt)
    if last_error is not None:
        raise last_error
    return output_path
