#!/usr/bin/env python
"""
Generate SFJSSP Benchmark Suite

Generates synthetic benchmark instances with explicit labeling.
Calibrated against literature values from:
- DyDFJSP 2023 (fatigue parameters)
- E-DFJSP 2025 (energy parameters)
- NSGA-III 2021 (ergonomic parameters)

Evidence Status: Generator design PROPOSED, parameters CONFIRMED from sources.
"""

import os
import sys
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum
import numpy as np

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from sfjssp_model.instance import SFJSSPInstance, InstanceLabel, InstanceType, DynamicEventParams
from sfjssp_model.machine import Machine, MachineMode
from sfjssp_model.worker import Worker
from sfjssp_model.job import Job, Operation


class InstanceSize(Enum):
    """Predefined instance sizes"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class GeneratorConfig:
    """Configuration for benchmark generation"""
    size: InstanceSize = InstanceSize.SMALL
    seed: int = 42
    is_dynamic: bool = False

    # Instance counts (set by size)
    n_jobs: int = 10
    n_machines: int = 5
    n_workers: int = 5

    # Processing time range (minutes)
    processing_time_range: Tuple[float, float] = (10.0, 100.0)

    # Energy parameters (kW) - from E-DFJSP 2025
    power_processing_range: Tuple[float, float] = (5.0, 50.0)
    power_idle_range: Tuple[float, float] = (1.0, 10.0)

    # Worker parameters - from DyDFJSP 2023
    fatigue_rate_range: Tuple[float, float] = (0.01, 0.05)
    recovery_rate_range: Tuple[float, float] = (0.02, 0.10)
    labor_cost_range: Tuple[float, float] = (10.0, 50.0)

    # Ergonomic parameters - from NSGA-III 2021
    ergonomic_risk_range: Tuple[float, float] = (0.1, 0.5)
    ocra_max_range: Tuple[float, float] = (2.0, 4.0)

    def __post_init__(self):
        """Set counts based on size"""
        if self.size == InstanceSize.SMALL:
            self.n_jobs = 10
            self.n_machines = 5
            self.n_workers = 5
        elif self.size == InstanceSize.MEDIUM:
            self.n_jobs = 50
            self.n_machines = 10
            self.n_workers = 10
        elif self.size == InstanceSize.LARGE:
            self.n_jobs = 200
            self.n_machines = 20
            self.n_workers = 20


class BenchmarkGenerator:
    """Synthetic benchmark generator for SFJSSP"""

    def __init__(self, config: GeneratorConfig = None):
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate(self) -> SFJSSPInstance:
        """Generate a synthetic SFJSSP instance"""
        self.rng = np.random.default_rng(self.config.seed)

        instance = SFJSSPInstance(
            instance_id=f"SFJSSP_{self.config.size.value}_{self.config.seed}",
            label=InstanceLabel.FULLY_SYNTHETIC,
            label_justification="All parameters computer-generated",
            instance_type=InstanceType.DYNAMIC if self.config.is_dynamic else InstanceType.STATIC,
            creation_date=datetime.now().isoformat(),
            source="Synthetic Generator v1.0",
            calibration_sources=[
                "DyDFJSP 2023 (fatigue)",
                "E-DFJSP 2025 (energy)",
                "NSGA-III 2021 (ergonomic)",
            ],
            known_limitations=[
                "Worker parameters not validated against real data",
                "Ergonomic indices are approximate",
                "No real breakdown patterns",
            ],
        )

        # Generate machines
        for m_id in range(self.config.n_machines):
            machine = Machine(
                machine_id=m_id,
                machine_name=f"Machine_{m_id}",
                power_processing=self.rng.uniform(*self.config.power_processing_range),
                power_idle=self.rng.uniform(*self.config.power_idle_range),
                power_setup=self.rng.uniform(3.0, 30.0),
                setup_time=self.rng.uniform(0.0, 10.0),
            )
            # Add modes
            n_modes = self.rng.integers(2, 4)
            for mode_idx in range(n_modes):
                mode = MachineMode(
                    mode_id=mode_idx,
                    speed_factor=0.8 + mode_idx * 0.2,
                    power_multiplier=0.7 + mode_idx * 0.3,
                )
                machine.modes.append(mode)

            instance.add_machine(machine)

        # Generate workers
        for w_id in range(self.config.n_workers):
            worker = Worker(
                worker_id=w_id,
                worker_name=f"Worker_{w_id}",
                labor_cost_per_hour=self.rng.uniform(*self.config.labor_cost_range),
                base_efficiency=self.rng.uniform(0.8, 1.2),
                fatigue_rate=self.rng.uniform(*self.config.fatigue_rate_range),
                recovery_rate=self.rng.uniform(*self.config.recovery_rate_range),
                ocra_max_per_shift=self.rng.uniform(*self.config.ocra_max_range),
            )
            instance.add_worker(worker)

        # Generate jobs
        for job_id in range(self.config.n_jobs):
            n_ops = self.rng.integers(2, 5)
            operations = []

            for op_idx in range(n_ops):
                op = Operation(job_id=job_id, op_id=op_idx)

                # Eligible machines (random subset)
                n_eligible_m = self.rng.integers(1, min(4, self.config.n_machines) + 1)
                op.eligible_machines = set(
                    self.rng.choice(self.config.n_machines, size=n_eligible_m, replace=False).tolist()
                )

                # Eligible workers (random subset)
                n_eligible_w = self.rng.integers(1, min(3, self.config.n_workers) + 1)
                op.eligible_workers = set(
                    self.rng.choice(self.config.n_workers, size=n_eligible_w, replace=False).tolist()
                )

                # Processing times
                for m_id in op.eligible_machines:
                    op.processing_times[m_id] = {}
                    machine = instance.get_machine(m_id)
                    if machine and machine.modes:
                        for mode in machine.modes:
                            base_pt = self.rng.uniform(*self.config.processing_time_range)
                            op.processing_times[m_id][mode.mode_id] = base_pt / mode.speed_factor
                    else:
                        op.processing_times[m_id] = {0: self.rng.uniform(*self.config.processing_time_range)}

                operations.append(op)

            # Calculate due date
            total_min_pt = sum(
                min(min(modes.values()) for modes in op.processing_times.values())
                for op in operations
            )
            due_margin = self.rng.uniform(1.5, 3.0)

            job = Job(
                job_id=job_id,
                operations=operations,
                arrival_time=self.rng.uniform(0, 50),
                due_date=total_min_pt * due_margin,
                weight=self.rng.uniform(1.0, 2.0),
            )
            instance.add_job(job)

        # Generate ergonomic risks
        for job in instance.jobs:
            for op in job.operations:
                instance.ergonomic_risk_map[(job.job_id, op.op_id)] = self.rng.uniform(
                    *self.config.ergonomic_risk_range
                )

        # Dynamic params
        if self.config.is_dynamic:
            instance.dynamic_params = DynamicEventParams(
                arrival_rate=0.1,
                breakdown_rate=0.001,
                repair_rate=0.1,
            )

        return instance

    def save_instance(self, instance: SFJSSPInstance, filepath: str):
        """Save instance metadata to JSON"""
        data = instance.to_dict()

        # Add summaries
        data['machines'] = [
            {
                'machine_id': m.machine_id,
                'power_processing': m.power_processing,
                'power_idle': m.power_idle,
                'n_modes': len(m.modes),
            }
            for m in instance.machines
        ]

        data['workers'] = [
            {
                'worker_id': w.worker_id,
                'labor_cost_per_hour': w.labor_cost_per_hour,
                'fatigue_rate': w.fatigue_rate,
                'ocra_max': w.ocra_max_per_shift,
            }
            for w in instance.workers
        ]

        data['jobs_summary'] = [
            {
                'job_id': j.job_id,
                'n_operations': len(j.operations),
                'due_date': j.due_date,
            }
            for j in instance.jobs
        ]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  Saved: {filepath}")


def generate_suite(output_dir: str = "benchmarks", n_per_size: int = 5):
    """Generate benchmark suite"""
    print("=" * 60)
    print("SFJSSP Benchmark Suite Generator")
    print("=" * 60)

    sizes = [InstanceSize.SMALL, InstanceSize.MEDIUM]

    for size in sizes:
        print(f"\n--- {size.value.upper()} Instances ---")
        size_dir = os.path.join(output_dir, size.value)
        os.makedirs(size_dir, exist_ok=True)

        config = GeneratorConfig(size=size, seed=42)
        generator = BenchmarkGenerator(config)

        for i in range(n_per_size):
            config.seed = 42 + i
            generator.config.seed = config.seed
            generator.rng = np.random.default_rng(config.seed)

            instance = generator.generate()
            instance.instance_id = f"SFJSSP_{size.value}_{i:03d}"

            filepath = os.path.join(size_dir, f"{instance.instance_id}.json")
            generator.save_instance(instance, filepath)

            print(f"  {instance.instance_id}: J={instance.n_jobs}, M={instance.n_machines}, W={instance.n_workers}")

    print(f"\nGenerated {n_per_size * len(sizes)} instances to {output_dir}")


def generate_example(output_dir: str = "benchmarks/examples"):
    """Generate single example instance"""
    print("=" * 60)
    print("Generating Example Instance")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    config = GeneratorConfig(size=InstanceSize.SMALL, seed=42)
    generator = BenchmarkGenerator(config)
    instance = generator.generate()
    instance.instance_id = "SFJSSP_EXAMPLE_001"

    filepath = os.path.join(output_dir, "example_001.json")
    generator.save_instance(instance, filepath)

    print(f"\nGenerated: {instance}")
    print(f"  Jobs: {instance.n_jobs}")
    print(f"  Machines: {instance.n_machines}")
    print(f"  Workers: {instance.n_workers}")
    print(f"  Operations: {instance.n_operations}")

    print("\nSample machines:")
    for m in instance.machines[:2]:
        print(f"  M{m.machine_id}: P={m.power_processing:.1f}kW, idle={m.power_idle:.1f}kW")

    print("\nSample workers:")
    for w in instance.workers[:2]:
        print(f"  W{w.worker_id}: ${w.labor_cost_per_hour:.1f}/hr, fatigue={w.fatigue_rate:.3f}")

    print(f"\nSaved to: {filepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFJSSP Benchmarks")
    parser.add_argument("--mode", choices=["example", "suite"], default="example")
    parser.add_argument("--output", type=str, default="benchmarks")
    parser.add_argument("--n", type=int, default=5, help="Instances per size (for suite)")

    args = parser.parse_args()

    if args.mode == "example":
        generate_example(args.output)
    else:
        generate_suite(args.output, args.n)

    print("\nDone!")
