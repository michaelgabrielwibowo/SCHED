import json
from pathlib import Path

import pytest

try:
    from ..sfjssp_model.instance import SFJSSPInstance
except ImportError:  # pragma: no cover - supports repo-root imports
    from sfjssp_model.instance import SFJSSPInstance


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "benchmarks"


def _load_benchmark(path: Path) -> SFJSSPInstance:
    with path.open("r", encoding="utf-8") as handle:
        return SFJSSPInstance.from_dict(json.load(handle))


@pytest.fixture(scope="session")
def small_benchmark_instance() -> SFJSSPInstance:
    return _load_benchmark(BENCHMARK_ROOT / "small" / "SFJSSP_small_000.json")


@pytest.fixture(scope="session")
def medium_benchmark_instance() -> SFJSSPInstance:
    return _load_benchmark(BENCHMARK_ROOT / "medium" / "SFJSSP_medium_000.json")
