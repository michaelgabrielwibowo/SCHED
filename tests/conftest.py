import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

try:
    from ..sfjssp_model.instance import SFJSSPInstance
except ImportError:  # pragma: no cover - supports repo-root imports
    from sfjssp_model.instance import SFJSSPInstance


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "benchmarks"
CANONICAL_BENCHMARK_FIXTURE_DIRS = ("small", "medium")
CANONICAL_BENCHMARK_FIXTURE_FILES = {
    "small": BENCHMARK_ROOT / "small" / "SFJSSP_small_000.json",
    "medium": BENCHMARK_ROOT / "medium" / "SFJSSP_medium_000.json",
}


def _load_benchmark(path: Path) -> SFJSSPInstance:
    with path.open("r", encoding="utf-8") as handle:
        return SFJSSPInstance.from_dict(json.load(handle))


@pytest.fixture
def tmp_path() -> Path:
    """Repo-local temporary directory fixture that avoids Pytest's tmpdir plugin."""

    temp_root = REPO_ROOT / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    path = temp_root / f"pytest-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def small_benchmark_instance() -> SFJSSPInstance:
    """Load the canonical small benchmark fixture committed to the repo."""

    return _load_benchmark(CANONICAL_BENCHMARK_FIXTURE_FILES["small"])


@pytest.fixture(scope="session")
def medium_benchmark_instance() -> SFJSSPInstance:
    """Load the canonical medium benchmark fixture committed to the repo."""

    return _load_benchmark(CANONICAL_BENCHMARK_FIXTURE_FILES["medium"])
