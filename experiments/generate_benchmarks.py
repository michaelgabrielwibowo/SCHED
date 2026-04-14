#!/usr/bin/env python
"""
Benchmark generation CLI and compatibility surface.

The canonical generator implementation lives in `utils.benchmark_generator`.
This module keeps the existing experiment-facing import path and CLI stable.
"""

from pathlib import Path

try:
    from ..utils.benchmark_generator import (
        BenchmarkGenerator,
        GeneratorConfig,
        InstanceSize,
    )
except ImportError:  # pragma: no cover - supports repo-root imports
    from utils.benchmark_generator import (
        BenchmarkGenerator,
        GeneratorConfig,
        InstanceSize,
    )


def generate_suite(output_dir: str = "benchmarks", n_per_size: int = 5):
    """Generate the published benchmark suite layout."""
    print("=" * 60)
    print("SFJSSP Benchmark Suite Generator")
    print("=" * 60)

    root = Path(output_dir)
    sizes = [InstanceSize.SMALL, InstanceSize.MEDIUM]

    for size in sizes:
        print(f"\n--- {size.value.upper()} Instances ---")
        size_dir = root / size.value
        size_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(n_per_size):
            config = GeneratorConfig(
                size=size,
                seed=42 + idx,
                instance_id=f"SFJSSP_{size.value}_{idx:03d}",
            )
            generator = BenchmarkGenerator(config)
            instance = generator.generate()
            generator.save_instance(instance, str(size_dir / f"{instance.instance_id}.json"))
            print(
                f"  {instance.instance_id}: "
                f"J={instance.n_jobs}, M={instance.n_machines}, W={instance.n_workers}"
            )

    print(f"\nGenerated {n_per_size * len(sizes)} instances to {root}")


def generate_example(output_dir: str = "benchmarks/examples"):
    """Generate a single example benchmark instance."""
    print("=" * 60)
    print("Generating Example Instance")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = GeneratorConfig(
        size=InstanceSize.SMALL,
        seed=42,
        instance_id="SFJSSP_EXAMPLE_001",
    )
    generator = BenchmarkGenerator(config)
    instance = generator.generate()

    filepath = output_path / "example_001.json"
    generator.save_instance(instance, str(filepath))

    print(f"\nGenerated: {instance}")
    print(f"  Jobs: {instance.n_jobs}")
    print(f"  Machines: {instance.n_machines}")
    print(f"  Workers: {instance.n_workers}")
    print(f"  Operations: {instance.n_operations}")

    print("\nSample machines:")
    for machine in instance.machines[:2]:
        print(
            f"  M{machine.machine_id}: "
            f"P={machine.power_processing:.1f}kW, idle={machine.power_idle:.1f}kW"
        )

    print("\nSample workers:")
    for worker in instance.workers[:2]:
        print(
            f"  W{worker.worker_id}: "
            f"${worker.labor_cost_per_hour:.1f}/hr, fatigue={worker.fatigue_rate:.3f}"
        )

    print(f"\nSaved to: {filepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFJSSP benchmarks")
    parser.add_argument("--mode", choices=["example", "suite"], default="example")
    parser.add_argument("--output", type=str, default="benchmarks")
    parser.add_argument("--n", type=int, default=5, help="Instances per size (for suite)")

    args = parser.parse_args()

    if args.mode == "example":
        generate_example(args.output)
    else:
        generate_suite(args.output, args.n)

    print("\nDone!")
