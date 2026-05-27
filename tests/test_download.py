"""Public benchmark download helper tests."""

from pathlib import Path

from src.data.download import PUBLIC_BENCHMARK_URLS, download_public_benchmark


def test_download_public_benchmark_reuses_existing_file(tmp_path: Path) -> None:
    """Existing raw files are reused unless overwrite is requested."""

    path = tmp_path / "data" / "raw" / "etth1" / "ETTh1.csv"
    path.parent.mkdir(parents=True)
    path.write_text("date,target\n2024-01-01,1\n", encoding="utf-8")

    output = download_public_benchmark("etth1", project_root=tmp_path)

    assert output == path
    assert output.read_text(encoding="utf-8").startswith("date,target")


def test_public_benchmark_urls_cover_registry_targets() -> None:
    """Downloader knows the public benchmark targets used by the registry."""

    assert {"etth1", "electricity", "exchange_rate", "traffic", "weather"}.issubset(
        PUBLIC_BENCHMARK_URLS
    )
