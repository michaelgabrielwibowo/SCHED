"""
Canonical synthetic benchmark generator for SFJSSP.

This module is the single implementation used by both utility scripts and the
`experiments.generate_benchmarks` public surface.
"""

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ..sfjssp_model.instance import (
        DynamicEventParams,
        InstanceLabel,
        InstanceType,
        SFJSSPInstance,
    )
    from ..sfjssp_model.job import Job, Operation
    from ..sfjssp_model.machine import Machine, MachineMode
    from ..sfjssp_model.worker import Worker
except ImportError:  # pragma: no cover - supports repo-root imports
    from sfjssp_model.instance import (
        DynamicEventParams,
        InstanceLabel,
        InstanceType,
        SFJSSPInstance,
    )
    from sfjssp_model.job import Job, Operation
    from sfjssp_model.machine import Machine, MachineMode
    from sfjssp_model.worker import Worker


class InstanceSize(Enum):
    """Predefined instance sizes."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


SIZE_DEFAULTS = {
    InstanceSize.SMALL: (10, 5, 5),
    InstanceSize.MEDIUM: (50, 10, 10),
    InstanceSize.LARGE: (200, 20, 20),
}

SUPPORTED_INSTANCE_SIZES: Tuple[InstanceSize, ...] = (
    InstanceSize.SMALL,
    InstanceSize.MEDIUM,
    InstanceSize.LARGE,
)
DEFAULT_SUITE_SIZES: Tuple[InstanceSize, ...] = (
    InstanceSize.SMALL,
    InstanceSize.MEDIUM,
)
BENCHMARK_DOCUMENT_SCHEMA = "sfjssp_benchmark_document_v1"
BENCHMARK_DOCUMENT_VERSION = 1
BENCHMARK_GENERATOR_VERSION = "3.0"

DEFAULT_CALIBRATION_SOURCES = [
    "DyDFJSP 2023 (fatigue parameters)",
    "E-DFJSP 2025 (energy parameters)",
    "NSGA-III 2021 (ergonomic parameters)",
]


def coerce_instance_size(value: Any) -> InstanceSize:
    """Normalize a string or enum value to an `InstanceSize`."""
    if isinstance(value, InstanceSize):
        return value
    normalized = str(value).strip().lower()
    for size in SUPPORTED_INSTANCE_SIZES:
        if size.value == normalized:
            return size
    raise ValueError(
        f"Unsupported instance size {value!r}; expected one of "
        f"{[size.value for size in SUPPORTED_INSTANCE_SIZES]}"
    )


def get_size_preset_table() -> Dict[str, Dict[str, int]]:
    """Return the authoritative public size preset table."""
    return {
        size.value: {
            "n_jobs": SIZE_DEFAULTS[size][0],
            "n_machines": SIZE_DEFAULTS[size][1],
            "n_workers": SIZE_DEFAULTS[size][2],
        }
        for size in SUPPORTED_INSTANCE_SIZES
    }


def _json_ready(value: Any) -> Any:
    """Convert config/provenance payloads into JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


@dataclass
class GeneratorConfig:
    """Configuration for synthetic benchmark generation."""

    instance_id: str = ""
    size: InstanceSize = InstanceSize.SMALL
    seed: int = 42
    is_dynamic: bool = False

    # Instance counts. Explicit values override size presets.
    n_jobs: Optional[int] = None
    n_machines: Optional[int] = None
    n_workers: Optional[int] = None

    # Job structure.
    n_operations_per_job: Tuple[int, int] = (2, 5)
    job_arrival_time: Tuple[float, float] = (0.0, 50.0)
    due_date_margin: Tuple[float, float] = (1.5, 3.0)
    job_weight_range: Tuple[float, float] = (1.0, 2.0)

    # Processing / machine characteristics.
    processing_time_range: Tuple[float, float] = (10.0, 100.0)
    n_modes_per_machine: Tuple[int, int] = (2, 4)
    power_processing_range: Tuple[float, float] = (5.0, 50.0)
    power_idle_range: Tuple[float, float] = (1.0, 10.0)
    power_setup_range: Tuple[float, float] = (3.0, 30.0)
    startup_energy_range: Tuple[float, float] = (10.0, 100.0)
    setup_time_range: Tuple[float, float] = (0.0, 10.0)
    machine_flexibility: Optional[float] = None
    mode_speed_factors: Optional[List[float]] = None
    mode_power_multipliers: Optional[List[float]] = None

    # Worker characteristics.
    worker_efficiency_range: Tuple[float, float] = (0.8, 1.2)
    labor_cost_range: Tuple[float, float] = (10.0, 50.0)
    fatigue_rate_range: Tuple[float, float] = (0.01, 0.05)
    recovery_rate_range: Tuple[float, float] = (0.02, 0.10)
    learning_coefficient_range: Tuple[float, float] = (0.01, 0.10)
    ocra_max_range: Tuple[float, float] = (2.2, 2.2)
    ergonomic_tolerance_range: Tuple[float, float] = (1.0, 1.0)
    worker_flexibility: Optional[float] = None

    # Ergonomics.
    ergonomic_risk_range: Tuple[float, float] = (0.001, 0.005)

    # Dynamic event parameters.
    arrival_rate: float = 0.1
    breakdown_rate: float = 0.001
    repair_rate: float = 0.1
    absence_probability: float = 0.05
    rush_order_probability: float = 0.1

    # Facility-level parameters.
    carbon_emission_factor: float = 0.5
    electricity_price: float = 0.10
    auxiliary_power_total: float = 50.0

    # Metadata.
    label: InstanceLabel = InstanceLabel.FULLY_SYNTHETIC
    calibration_sources: Optional[List[str]] = None

    def __post_init__(self):
        defaults = SIZE_DEFAULTS[self.size]
        if self.n_jobs is None:
            self.n_jobs = defaults[0]
        if self.n_machines is None:
            self.n_machines = defaults[1]
        if self.n_workers is None:
            self.n_workers = defaults[2]
        if not self.instance_id:
            self.instance_id = f"SFJSSP_{self.size.value}_{self.seed}"
        if self.calibration_sources is None:
            self.calibration_sources = list(DEFAULT_CALIBRATION_SOURCES)
        if self.mode_speed_factors is None:
            self.mode_speed_factors = [0.8, 1.0, 1.2, 1.4]
        if self.mode_power_multipliers is None:
            self.mode_power_multipliers = [0.7, 1.0, 1.3, 1.6]


class BenchmarkGenerator:
    """Synthetic benchmark generator for SFJSSP."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate(self) -> SFJSSPInstance:
        """Generate a synthetic SFJSSP instance from the current configuration."""
        self.rng = np.random.default_rng(self.config.seed)

        instance = SFJSSPInstance(
            instance_id=self.config.instance_id,
            instance_name=f"SFJSSP_{self.config.size.value}_{self.config.seed}",
            label=self.config.label,
            label_justification=self._get_label_justification(),
            instance_type=InstanceType.DYNAMIC if self.config.is_dynamic else InstanceType.STATIC,
            creation_date=datetime.now().isoformat(),
            source="Synthetic Generator v1.0",
            calibration_sources=list(self.config.calibration_sources),
            known_limitations=self._get_known_limitations(),
            carbon_emission_factor=self.config.carbon_emission_factor,
            default_electricity_price=self.config.electricity_price,
            auxiliary_power_total=self.config.auxiliary_power_total,
        )

        self._generate_machines(instance)
        self._generate_workers(instance)
        self._generate_jobs(instance)
        self._generate_ergonomic_risks(instance)

        if self.config.is_dynamic:
            instance.dynamic_params = DynamicEventParams(
                arrival_rate=self.config.arrival_rate,
                breakdown_rate=self.config.breakdown_rate,
                repair_rate=self.config.repair_rate,
                absence_probability=self.config.absence_probability,
                rush_order_probability=self.config.rush_order_probability,
            )

        return instance

    def generate_suite(
        self,
        sizes: Optional[List[InstanceSize]] = None,
        instances_per_size: int = 10,
        base_seed: int = 42,
    ) -> List[SFJSSPInstance]:
        """Generate a suite of benchmark instances."""
        sizes = [coerce_instance_size(size) for size in (sizes or DEFAULT_SUITE_SIZES)]
        instances: List[SFJSSPInstance] = []
        original_config = self.config

        for size in sizes:
            for idx in range(instances_per_size):
                config = replace(
                    original_config,
                    instance_id=f"SFJSSP_{size.value}_{idx:03d}",
                    size=size,
                    seed=base_seed + idx,
                    n_jobs=None,
                    n_machines=None,
                    n_workers=None,
                )
                self.config = config
                self.rng = np.random.default_rng(config.seed)
                instances.append(self.generate())

        self.config = original_config
        self.rng = np.random.default_rng(self.config.seed)
        return instances

    def save_instance(self, instance: SFJSSPInstance, filepath: str):
        """Save a fully reconstructible benchmark instance to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.build_instance_document(instance)
        data["machines_summary"] = [
            {
                "machine_id": machine.machine_id,
                "machine_name": machine.machine_name,
                "power_processing": machine.power_processing,
                "power_idle": machine.power_idle,
                "power_setup": machine.power_setup,
                "startup_energy": machine.startup_energy,
                "setup_time": machine.setup_time,
                "n_modes": len(machine.modes),
            }
            for machine in instance.machines
        ]
        data["workers_summary"] = [
            {
                "worker_id": worker.worker_id,
                "worker_name": worker.worker_name,
                "labor_cost_per_hour": worker.labor_cost_per_hour,
                "base_efficiency": worker.base_efficiency,
                "fatigue_rate": worker.fatigue_rate,
                "recovery_rate": worker.recovery_rate,
                "learning_coefficient": worker.learning_coefficient,
                "ocra_max_per_shift": worker.ocra_max_per_shift,
            }
            for worker in instance.workers
        ]
        data["jobs_summary"] = [
            {
                "job_id": job.job_id,
                "n_operations": len(job.operations),
                "arrival_time": job.arrival_time,
                "due_date": job.due_date,
                "weight": job.weight,
            }
            for job in instance.jobs
        ]

        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

        print(f"Saved instance to {path}")

    def save_suite(self, instances: List[SFJSSPInstance], output_dir: str):
        """Save a suite of instances and canonical metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for instance in instances:
            self.save_instance(instance, str(output_path / f"{instance.instance_id}.json"))

        sizes = sorted(
            {
                size_label
                for instance in instances
                if (size_label := self._infer_size_label(instance)) is not None
            }
        )
        metadata = {
            "document_schema": BENCHMARK_DOCUMENT_SCHEMA,
            "document_version": BENCHMARK_DOCUMENT_VERSION,
            "document_type": "suite_manifest",
            "generator_provenance": self._build_generator_provenance(),
            "n_instances": len(instances),
            "sizes": sizes,
            "instance_types": sorted({instance.instance_type.value for instance in instances}),
            "generation_date": datetime.now().isoformat(),
            "generator_version": BENCHMARK_GENERATOR_VERSION,
            "size_preset_table": get_size_preset_table(),
            "instances": [
                {
                    "instance_id": instance.instance_id,
                    "filename": f"{instance.instance_id}.json",
                    "size": self._infer_size_label(instance),
                    "instance_type": instance.instance_type.value,
                }
                for instance in instances
            ],
            "config": {
                "is_dynamic": self.config.is_dynamic,
                "calibration_sources": list(self.config.calibration_sources),
            },
        }

        with (output_path / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        print(f"Saved {len(instances)} instances to {output_path}")

    def _get_label_justification(self) -> str:
        if self.config.label == InstanceLabel.FULLY_SYNTHETIC:
            return "All parameters computer-generated with documented distributions"
        if self.config.label == InstanceLabel.EXTENDED_SYNTHETIC:
            return "Classical benchmark extended with synthetic SFJSSP parameters"
        if self.config.label == InstanceLabel.CALIBRATED_SYNTHETIC:
            return "Synthetic parameters calibrated against real industrial data"
        return "Computer-generated instance"

    @staticmethod
    def _get_known_limitations() -> List[str]:
        return [
            "Worker parameters not validated against real worker performance data",
            "Ergonomic risk indices are approximate (OCRA-inspired, not certified)",
            "No real breakdown/availability patterns used",
            "Processing times are synthetic, not from time studies",
            "Energy parameters based on literature values, not machine specifications",
        ]

    def _generate_machines(self, instance: SFJSSPInstance):
        auxiliary_share = (
            self.config.auxiliary_power_total / self.config.n_machines
            if self.config.n_machines
            else 0.0
        )

        for machine_id in range(self.config.n_machines):
            n_modes = max(1, self._sample_range(self.config.n_modes_per_machine))
            modes = []
            for mode_idx in range(n_modes):
                modes.append(
                    MachineMode(
                        mode_id=mode_idx,
                        mode_name=f"M{machine_id}_Mode{mode_idx}",
                        speed_factor=self._mode_factor(
                            self.config.mode_speed_factors,
                            mode_idx,
                            0.8 + (0.2 * mode_idx),
                        ),
                        power_multiplier=self._mode_factor(
                            self.config.mode_power_multipliers,
                            mode_idx,
                            0.7 + (0.3 * mode_idx),
                        ),
                    )
                )

            instance.add_machine(
                Machine(
                    machine_id=machine_id,
                    machine_name=f"Machine_{machine_id}",
                    modes=modes,
                    default_mode_id=0,
                    power_processing=self.rng.uniform(*self.config.power_processing_range),
                    power_idle=self.rng.uniform(*self.config.power_idle_range),
                    power_setup=self.rng.uniform(*self.config.power_setup_range),
                    startup_energy=self.rng.uniform(*self.config.startup_energy_range),
                    setup_time=self.rng.uniform(*self.config.setup_time_range),
                    auxiliary_power_share=auxiliary_share,
                )
            )

    def _generate_workers(self, instance: SFJSSPInstance):
        for worker_id in range(self.config.n_workers):
            instance.add_worker(
                Worker(
                    worker_id=worker_id,
                    worker_name=f"Worker_{worker_id}",
                    labor_cost_per_hour=self.rng.uniform(*self.config.labor_cost_range),
                    base_efficiency=self.rng.uniform(*self.config.worker_efficiency_range),
                    fatigue_rate=self.rng.uniform(*self.config.fatigue_rate_range),
                    recovery_rate=self.rng.uniform(*self.config.recovery_rate_range),
                    learning_coefficient=self.rng.uniform(*self.config.learning_coefficient_range),
                    ocra_max_per_shift=self.rng.uniform(*self.config.ocra_max_range),
                    ergonomic_tolerance=self.rng.uniform(*self.config.ergonomic_tolerance_range),
                )
            )

    def _generate_jobs(self, instance: SFJSSPInstance):
        machine_ids = [machine.machine_id for machine in instance.machines]
        worker_ids = [worker.worker_id for worker in instance.workers]

        for job_id in range(self.config.n_jobs):
            operations = []
            n_operations = max(1, self._sample_range(self.config.n_operations_per_job))

            for op_id in range(n_operations):
                operation = Operation(job_id=job_id, op_id=op_id)
                operation.eligible_machines = set(
                    self._sample_eligible_ids(machine_ids, self.config.machine_flexibility, fallback_cap=4)
                )
                operation.eligible_workers = set(
                    self._sample_eligible_ids(worker_ids, self.config.worker_flexibility, fallback_cap=3)
                )

                for machine_id in operation.eligible_machines:
                    operation.processing_times[machine_id] = {}
                    machine = instance.get_machine(machine_id)
                    if machine and machine.modes:
                        for mode in machine.modes:
                            base_time = self.rng.uniform(*self.config.processing_time_range)
                            operation.processing_times[machine_id][mode.mode_id] = (
                                base_time / mode.speed_factor
                            )
                    else:
                        operation.processing_times[machine_id] = {
                            0: self.rng.uniform(*self.config.processing_time_range)
                        }

                operations.append(operation)

            total_min_processing = sum(
                min(min(modes.values()) for modes in operation.processing_times.values())
                for operation in operations
            )
            arrival_time = self.rng.uniform(*self.config.job_arrival_time)
            due_margin = self.rng.uniform(*self.config.due_date_margin)
            instance.add_job(
                Job(
                    job_id=job_id,
                    operations=operations,
                    arrival_time=arrival_time,
                    due_date=arrival_time + (total_min_processing * due_margin),
                    weight=self.rng.uniform(*self.config.job_weight_range),
                )
            )

    def _generate_ergonomic_risks(self, instance: SFJSSPInstance):
        for job in instance.jobs:
            for operation in job.operations:
                instance.ergonomic_risk_map[(job.job_id, operation.op_id)] = self.rng.uniform(
                    *self.config.ergonomic_risk_range
                )

    def _sample_range(self, bounds: Tuple[int, int]) -> int:
        low, high = bounds
        if high <= low:
            return int(low)
        return int(self.rng.integers(low, high))

    def _sample_eligible_ids(
        self,
        ids: List[int],
        flexibility: Optional[float],
        fallback_cap: int,
    ) -> List[int]:
        if not ids:
            return []

        if flexibility is None:
            max_count = min(len(ids), fallback_cap)
        else:
            max_count = int(np.ceil(len(ids) * flexibility))
            if len(ids) > 1:
                max_count = max(2, max_count)
            max_count = min(len(ids), max_count)

        max_count = max(1, max_count)
        count = 1 if max_count == 1 else int(self.rng.integers(1, max_count + 1))
        return sorted(self.rng.choice(ids, size=count, replace=False).tolist())

    @staticmethod
    def _mode_factor(values: List[float], index: int, fallback: float) -> float:
        if index < len(values):
            return float(values[index])
        return float(fallback)

    def _serialize_config(self) -> Dict[str, Any]:
        """Return the full generator config as a JSON-ready dictionary."""
        return {
            key: _json_ready(value)
            for key, value in asdict(self.config).items()
        }

    def _build_generator_provenance(self) -> Dict[str, Any]:
        """Return generator provenance embedded in saved benchmark artifacts."""
        return {
            "generator_module": "utils.benchmark_generator",
            "generator_version": BENCHMARK_GENERATOR_VERSION,
            "generated_at": datetime.now().isoformat(),
            "runtime_supported_sizes": [size.value for size in SUPPORTED_INSTANCE_SIZES],
            "default_suite_sizes": [size.value for size in DEFAULT_SUITE_SIZES],
        }

    def build_instance_document(self, instance: SFJSSPInstance) -> Dict[str, Any]:
        """Build the canonical persisted benchmark document for one instance."""
        data = instance.to_dict()
        data["document_schema"] = BENCHMARK_DOCUMENT_SCHEMA
        data["document_version"] = BENCHMARK_DOCUMENT_VERSION
        data["document_type"] = "instance"
        data["generator_provenance"] = self._build_generator_provenance()
        data["generator_config"] = self._serialize_config()
        data["size_preset_table"] = get_size_preset_table()
        return data

    @staticmethod
    def _infer_size_label(instance: SFJSSPInstance) -> Optional[str]:
        search_space = " ".join(
            part.lower()
            for part in [instance.instance_id, instance.instance_name]
            if part
        )
        for size in InstanceSize:
            if size.value in search_space:
                return size.value
        return None
