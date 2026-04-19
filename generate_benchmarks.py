"""
Compatibility wrapper for the canonical benchmark generator CLI.

The authoritative public entrypoint is `experiments.generate_benchmarks`.
This module remains for backwards-compatible imports and scripts.
"""

from pathlib import Path
from typing import Optional

try:
    from .experiments.generate_benchmarks import (
        generate_example,
        generate_suite,
        main,
    )
    from .utils.benchmark_generator import InstanceSize
except ImportError:  # pragma: no cover - supports repo-root imports
    from experiments.generate_benchmarks import (
        generate_example,
        generate_suite,
        main,
    )
    from utils.benchmark_generator import InstanceSize


def generate_large_benchmarks(
    output_dir: str = "benchmarks/large",
    n_instances: int = 3,
    base_seed: int = 43,
) -> None:
    """Backwards-compatible large-slice helper routed to the canonical suite CLI."""
    output_path = Path(output_dir)
    suite_root = output_path.parent if output_path.name.lower() == InstanceSize.LARGE.value else output_path
    generate_suite(
        output_dir=str(suite_root),
        n_per_size=n_instances,
        sizes=[InstanceSize.LARGE],
        base_seed=base_seed,
        is_dynamic=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
