#!/usr/bin/env python
"""
Canonical benchmark generation CLI.

This is the single public entrypoint for generating benchmark instances and
suite metadata. The generation semantics live in `utils.benchmark_generator`.
"""

from pathlib import Path
from typing import Iterable, List, Optional

try:
    from ..utils.benchmark_generator import (
        BENCHMARK_DOCUMENT_SCHEMA,
        BENCHMARK_DOCUMENT_VERSION,
        DEFAULT_SUITE_SIZES,
        SUPPORTED_INSTANCE_SIZES,
        BenchmarkGenerator,
        GeneratorConfig,
        InstanceSize,
        coerce_instance_size,
        get_size_preset_table,
    )
except ImportError:  # pragma: no cover - supports repo-root imports
    from utils.benchmark_generator import (
        BENCHMARK_DOCUMENT_SCHEMA,
        BENCHMARK_DOCUMENT_VERSION,
        DEFAULT_SUITE_SIZES,
        SUPPORTED_INSTANCE_SIZES,
        BenchmarkGenerator,
        GeneratorConfig,
        InstanceSize,
        coerce_instance_size,
        get_size_preset_table,
    )


def _normalize_sizes(sizes: Optional[Iterable[object]]) -> List[InstanceSize]:
    """Return validated size presets in user-specified order."""
    requested = list(sizes) if sizes is not None else list(DEFAULT_SUITE_SIZES)
    normalized: List[InstanceSize] = []
    for value in requested:
        size = coerce_instance_size(value)
        if size not in normalized:
            normalized.append(size)
    return normalized


def parse_size_csv(size_csv: str) -> List[InstanceSize]:
    """Parse a comma-separated size list from the CLI."""
    values = [chunk.strip() for chunk in size_csv.split(",") if chunk.strip()]
    if not values:
        raise ValueError("At least one size must be provided")
    return _normalize_sizes(values)


def generate_suite(
    output_dir: str = "benchmarks",
    n_per_size: int = 5,
    sizes: Optional[Iterable[object]] = None,
    base_seed: int = 42,
    is_dynamic: bool = False,
):
    """Generate the published benchmark suite layout from the canonical generator."""
    selected_sizes = _normalize_sizes(sizes)
    generator = BenchmarkGenerator(
        GeneratorConfig(
            size=selected_sizes[0],
            seed=base_seed,
            is_dynamic=is_dynamic,
        )
    )
    instances = generator.generate_suite(
        sizes=selected_sizes,
        instances_per_size=n_per_size,
        base_seed=base_seed,
    )
    generator.save_suite(instances, output_dir)

    print("=" * 60)
    print("SFJSSP Benchmark Suite Generator")
    print("=" * 60)
    print(f"Schema: {BENCHMARK_DOCUMENT_SCHEMA} v{BENCHMARK_DOCUMENT_VERSION}")
    print(f"Supported sizes: {[size.value for size in SUPPORTED_INSTANCE_SIZES]}")
    print(f"Generated {len(instances)} instances into {Path(output_dir)}")
    return instances


def generate_example(
    output_dir: str = "benchmarks",
    seed: int = 42,
    is_dynamic: bool = False,
):
    """Generate a single example benchmark instance."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = GeneratorConfig(
        size=InstanceSize.SMALL,
        seed=seed,
        is_dynamic=is_dynamic,
        instance_id="SFJSSP_example_001",
    )
    generator = BenchmarkGenerator(config)
    instance = generator.generate()

    filepath = output_path / "example_001.json"
    generator.save_instance(instance, str(filepath))

    print("=" * 60)
    print("Generated Example Benchmark")
    print("=" * 60)
    print(f"Schema: {BENCHMARK_DOCUMENT_SCHEMA} v{BENCHMARK_DOCUMENT_VERSION}")
    print(f"Size presets: {get_size_preset_table()}")
    print(f"Saved to: {filepath}")
    return instance


def main(argv: Optional[List[str]] = None) -> int:
    """Run the canonical benchmark generator CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate canonical SFJSSP benchmarks")
    parser.add_argument("--mode", choices=["example", "suite"], default="example")
    parser.add_argument("--output", type=str, default="benchmarks")
    parser.add_argument("--n", type=int, default=5, help="Instances per size when --mode suite")
    parser.add_argument(
        "--sizes",
        type=str,
        default="small,medium",
        help="Comma-separated size presets for --mode suite",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--dynamic", action="store_true", help="Emit dynamic benchmark instances")

    args = parser.parse_args(argv)

    if args.mode == "example":
        generate_example(args.output, seed=args.base_seed, is_dynamic=args.dynamic)
    else:
        generate_suite(
            output_dir=args.output,
            n_per_size=args.n,
            sizes=parse_size_csv(args.sizes),
            base_seed=args.base_seed,
            is_dynamic=args.dynamic,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
