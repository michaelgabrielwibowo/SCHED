"""
Synthetic Benchmark Generator for SFJSSP

Evidence Status:
- Generator design: Based on DATASET_INVENTORY_SFJSSP.md [PROPOSED]
- Calibration sources: DyDFJSP 2023, E-DFJSP 2025, NSGA-III 2021 [CONFIRMED parameters]
- Labeling protocol: PROPOSED for transparency

This generator creates synthetic SFJSSP instances with:
- Human factors (fatigue, ergonomic risk, skills, learning)
- Environmental factors (energy, carbon, machine modes)
- Dynamic events (job arrivals, breakdowns)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
import json
from datetime import datetime

from sfjssp_model.instance import (
    SFJSSPInstance,
    InstanceLabel,
    InstanceType,
    DynamicEventParams,
)
from sfjssp_model.job import Job, Operation
from sfjssp_model.machine import Machine, MachineMode, MachineState
from sfjssp_model.worker import Worker, WorkerSkill, WorkerState


class InstanceSize(Enum):
    """Predefined instance sizes"""
    SMALL = "small"       # 10 jobs, 5 machines, 5 workers
    MEDIUM = "medium"     # 50 jobs, 10 machines, 10 workers
    LARGE = "large"       # 200 jobs, 20 machines, 20 workers
    INDUSTRIAL = "industrial"  # 500 jobs, 50 machines, 50 workers


@dataclass
class GeneratorConfig:
    """
    Configuration for benchmark generation

    Evidence: Parameters calibrated from literature [CONFIRMED sources noted]
    """
    # Instance identification
    instance_id: str = "SFJSSP_GEN_001"
    size: InstanceSize = InstanceSize.SMALL

    # Random seed
    seed: int = 42

    # Job parameters
    n_jobs: int = 10
    n_operations_per_job: Tuple[int, int] = (2, 5)  # (min, max)
    job_arrival_time: Tuple[float, float] = (0.0, 100.0)
    due_date_margin: Tuple[float, float] = (10.0, 20.0)  # Multiplier for processing time
    job_weight_range: Tuple[float, float] = (1.0, 2.0)

    # Machine parameters (calibrated from E-DFJSP 2025)
    n_machines: int = 5
    n_modes_per_machine: Tuple[int, int] = (2, 4)
    power_processing_range: Tuple[float, float] = (5.0, 50.0)  # kW
    power_idle_range: Tuple[float, float] = (1.0, 10.0)  # kW
    power_setup_range: Tuple[float, float] = (3.0, 30.0)  # kW
    startup_energy_range: Tuple[float, float] = (10.0, 100.0)  # kWh
    setup_time_range: Tuple[float, float] = (0.0, 10.0)  # minutes
    machine_flexibility: float = 0.7  # Probability machine is eligible for operation

    # Worker parameters (calibrated from DyDFJSP 2023, NSGA-III 2021)
    n_workers: int = 5
    worker_efficiency_range: Tuple[float, float] = (0.7, 1.2)
    labor_cost_range: Tuple[float, float] = (10.0, 50.0)  # $/hour
    fatigue_rate_range: Tuple[float, float] = (0.01, 0.05)  # alpha
    recovery_rate_range: Tuple[float, float] = (0.02, 0.10)  # beta
    learning_coefficient_range: Tuple[float, float] = (0.01, 0.10)  # gamma
    ocra_max_range: Tuple[float, float] = (2.0, 4.0)  # OCRA threshold
    ergonomic_tolerance_range: Tuple[float, float] = (0.9, 1.1)
    worker_flexibility: float = 0.7  # Probability worker is eligible for operation

    # Ergonomic parameters (calibrated from NSGA-III 2021)
    # [CHANGED] Lowered from (0.1, 0.5) to match strict OCRA < 2.2 limit over minutes
    ergonomic_risk_range: Tuple[float, float] = (0.001, 0.01)  # Risk rate per minute

    # Processing time parameters
    processing_time_range: Tuple[float, float] = (10.0, 100.0)  # minutes
    mode_speed_factors: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.5])
    mode_power_multipliers: List[float] = field(default_factory=lambda: [0.7, 1.0, 1.4, 2.0])

    # Dynamic event parameters (for DYNAMIC instances)
    is_dynamic: bool = False
    arrival_rate: float = 0.1  # Jobs per time unit
    breakdown_rate: float = 0.001  # Per machine per time unit
    repair_rate: float = 0.1  # Per time unit
    absence_probability: float = 0.05

    # Environmental parameters
    carbon_emission_factor: float = 0.5  # kg CO2/kWh
    electricity_price: float = 0.10  # $/kWh
    auxiliary_power_total: float = 50.0  # kW

    # Labeling
    label: InstanceLabel = InstanceLabel.FULLY_SYNTHETIC
    calibration_sources: List[str] = field(default_factory=lambda: [
        "DyDFJSP 2023 (fatigue parameters)",
        "E-DFJSP 2025 (energy parameters)",
        "NSGA-III 2021 (ergonomic parameters)",
    ])

    def __post_init__(self):
        """Set default counts based on size if not specified"""
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
        elif self.size == InstanceSize.INDUSTRIAL:
            self.n_jobs = 500
            self.n_machines = 50
            self.n_workers = 50


class BenchmarkGenerator:
    """
    Synthetic benchmark generator for SFJSSP

    Generates instances with explicit labeling and documented assumptions.

    Evidence: Generator design from DATASET_INVENTORY_SFJSSP.md [PROPOSED]
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize generator

        Args:
            config: Generator configuration
        """
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate(self) -> SFJSSPInstance:
        """
        Generate a synthetic SFJSSP instance

        Returns:
            SFJSSPInstance with all parameters populated
        """
        # Reset random state
        self.rng = np.random.default_rng(self.config.seed)

        # Create instance
        instance = SFJSSPInstance(
            instance_id=self.config.instance_id,
            instance_name=f"SFJSSP_{self.config.size.value}_{self.config.seed}",
            label=self.config.label,
            label_justification=self._get_label_justification(),
            instance_type=InstanceType.DYNAMIC if self.config.is_dynamic else InstanceType.STATIC,
            creation_date=datetime.now().isoformat(),
            source="Synthetic Generator v1.0",
            calibration_sources=self.config.calibration_sources,
            known_limitations=self._get_known_limitations(),
            carbon_emission_factor=self.config.carbon_emission_factor,
            default_electricity_price=self.config.electricity_price,
            auxiliary_power_total=self.config.auxiliary_power_total,
        )

        # Generate machines
        self._generate_machines(instance)

        # Generate workers
        self._generate_workers(instance)

        # Generate jobs
        self._generate_jobs(instance)

        # Generate ergonomic risk map
        self._generate_ergonomic_risks(instance)

        # Set dynamic parameters if needed
        if self.config.is_dynamic:
            instance.dynamic_params = DynamicEventParams(
                arrival_rate=self.config.arrival_rate,
                breakdown_rate=self.config.breakdown_rate,
                repair_rate=self.config.repair_rate,
                absence_probability=self.config.absence_probability,
            )

        return instance

    def generate_suite(
        self,
        sizes: Optional[List[InstanceSize]] = None,
        instances_per_size: int = 10,
        base_seed: int = 42
    ) -> List[SFJSSPInstance]:
        """
        Generate a suite of benchmark instances

        Args:
            sizes: List of sizes to generate
            instances_per_size: Number of instances per size
            base_seed: Base random seed

        Returns:
            List of SFJSSPInstance objects
        """
        sizes = sizes or [InstanceSize.SMALL, InstanceSize.MEDIUM, InstanceSize.LARGE]
        instances = []

        for size in sizes:
            for i in range(instances_per_size):
                config = GeneratorConfig(
                    size=size,
                    seed=base_seed + i,
                    instance_id=f"SFJSSP_{size.value}_{i:03d}",
                    is_dynamic=self.config.is_dynamic,
                )
                self.config = config
                self.rng = np.random.default_rng(config.seed)

                instance = self.generate()
                instances.append(instance)

        return instances

    def _get_label_justification(self) -> str:
        """Get justification for instance label"""
        if self.config.label == InstanceLabel.FULLY_SYNTHETIC:
            return "All parameters computer-generated with documented distributions"
        elif self.config.label == InstanceLabel.EXTENDED_SYNTHETIC:
            return "Classical benchmark extended with synthetic SFJSSP parameters"
        elif self.config.label == InstanceLabel.CALIBRATED_SYNTHETIC:
            return "Synthetic parameters calibrated against real industrial data"
        else:
            return "Computer-generated instance"

    def _get_known_limitations(self) -> List[str]:
        """Get known limitations of generated instance"""
        return [
            "Worker parameters not validated against real worker performance data",
            "Ergonomic risk indices are approximate (OCRA-inspired, not certified)",
            "No real breakdown/availability patterns used",
            "Processing times are synthetic, not from time studies",
            "Energy parameters based on literature values, not machine specifications",
        ]

    def _generate_machines(self, instance: SFJSSPInstance):
        """Generate machines with energy parameters"""
        for m_id in range(self.config.n_machines):
            # Generate modes
            n_modes = self.rng.integers(
                *self.config.n_modes_per_machine
            )
            modes = []
            for mode_idx in range(n_modes):
                mode = MachineMode(
                    mode_id=mode_idx,
                    mode_name=f"M{m_id}_Mode{mode_idx}",
                    speed_factor=self.rng.choice(self.config.mode_speed_factors),
                    power_multiplier=self.rng.choice(self.config.mode_power_multipliers),
                )
                modes.append(mode)

            # Generate machine
            machine = Machine(
                machine_id=m_id,
                machine_name=f"Machine_{m_id}",
                modes=modes,
                default_mode_id=0,
                power_processing=self.rng.uniform(*self.config.power_processing_range),
                power_idle=self.rng.uniform(*self.config.power_idle_range),
                power_setup=self.rng.uniform(*self.config.power_setup_range),
                startup_energy=self.rng.uniform(*self.config.startup_energy_range),
                setup_time=self.rng.uniform(*self.config.setup_time_range),
                auxiliary_power_share=self.config.auxiliary_power_total / self.config.n_machines,
            )

            instance.add_machine(machine)

    def _generate_workers(self, instance: SFJSSPInstance):
        """Generate workers with human factors"""
        for w_id in range(self.config.n_workers):
            worker = Worker(
                worker_id=w_id,
                worker_name=f"Worker_{w_id}",
                labor_cost_per_hour=self.rng.uniform(*self.config.labor_cost_range),
                base_efficiency=self.rng.uniform(*self.config.worker_efficiency_range),
                fatigue_rate=self.rng.uniform(*self.config.fatigue_rate_range),
                recovery_rate=self.rng.uniform(*self.config.recovery_rate_range),
                learning_coefficient=self.rng.uniform(*self.config.learning_coefficient_range),
                ocra_max_per_shift=self.rng.uniform(*self.config.ocra_max_range),
                ergonomic_tolerance=self.rng.uniform(*self.config.ergonomic_tolerance_range),
            )

            instance.add_worker(worker)

    def _generate_jobs(self, instance: SFJSSPInstance):
        """Generate jobs with operations"""
        for job_id in range(self.config.n_jobs):
            n_ops = self.rng.integers(*self.config.n_operations_per_job)
            operations = []

            for op_idx in range(n_ops):
                op = Operation(
                    job_id=job_id,
                    op_id=op_idx,
                )

                # Assign eligible machines
                n_eligible_machines = self.rng.integers(
                    1, max(2, int(self.config.n_machines * self.config.machine_flexibility)) + 1
                )
                eligible_machines = self.rng.choice(
                    self.config.n_machines, size=n_eligible_machines, replace=False
                )
                op.eligible_machines = set(eligible_machines.tolist())

                # Assign eligible workers
                n_eligible_workers = self.rng.integers(
                    1, max(2, int(self.config.n_workers * self.config.worker_flexibility)) + 1
                )
                eligible_workers = self.rng.choice(
                    self.config.n_workers, size=n_eligible_workers, replace=False
                )
                op.eligible_workers = set(eligible_workers.tolist())

                # Generate processing times for each machine/mode combination
                for m_id in op.eligible_machines:
                    op.processing_times[m_id] = {}
                    machine = instance.get_machine(m_id)

                    if machine and machine.modes:
                        for mode in machine.modes:
                            base_time = self.rng.uniform(*self.config.processing_time_range)
                            adjusted_time = base_time / mode.speed_factor
                            op.processing_times[m_id][mode.mode_id] = adjusted_time
                    else:
                        op.processing_times[m_id] = {
                            0: self.rng.uniform(*self.config.processing_time_range)
                        }

                operations.append(op)

            # Create job
            total_min_processing = sum(
                min(
                    min(modes.values()) for modes in op.processing_times.values()
                )
                for op in operations
            )

            due_date_margin = self.rng.uniform(*self.config.due_date_margin)
            job = Job(
                job_id=job_id,
                operations=operations,
                arrival_time=self.rng.uniform(*self.config.job_arrival_time),
                due_date=total_min_processing * due_date_margin,
                weight=self.rng.uniform(*self.config.job_weight_range),
            )

            instance.add_job(job)

    def _generate_ergonomic_risks(self, instance: SFJSSPInstance):
        """Generate ergonomic risk map for all operations"""
        for job in instance.jobs:
            for op in job.operations:
                risk_rate = self.rng.uniform(*self.config.ergonomic_risk_range)
                instance.ergonomic_risk_map[(job.job_id, op.op_id)] = risk_rate

    def save_instance(self, instance: SFJSSPInstance, filepath: str):
        """
        Save instance to JSON file

        Note: This saves metadata and parameters. Full serialization
        requires custom handlers for complex objects.
        """
        data = instance.to_dict()

        # Add detailed parameter information
        data['generator_config'] = {
            'seed': self.config.seed,
            'size': self.config.size.value,
            'n_jobs': self.config.n_jobs,
            'n_machines': self.config.n_machines,
            'n_workers': self.config.n_workers,
            'is_dynamic': self.config.is_dynamic,
        }

        # Add machine details
        data['machines'] = [
            {
                'machine_id': m.machine_id,
                'machine_name': m.machine_name,
                'power_processing': m.power_processing,
                'power_idle': m.power_idle,
                'power_setup': m.power_setup,
                'startup_energy': m.startup_energy,
                'setup_time': m.setup_time,
                'n_modes': len(m.modes),
                'modes': [
                    {
                        'mode_id': mode.mode_id,
                        'speed_factor': mode.speed_factor,
                        'power_multiplier': mode.power_multiplier,
                    }
                    for mode in m.modes
                ],
            }
            for m in instance.machines
        ]

        # Add worker details
        data['workers'] = [
            {
                'worker_id': w.worker_id,
                'worker_name': w.worker_name,
                'labor_cost_per_hour': w.labor_cost_per_hour,
                'base_efficiency': w.base_efficiency,
                'fatigue_rate': w.fatigue_rate,
                'recovery_rate': w.recovery_rate,
                'learning_coefficient': w.learning_coefficient,
                'ocra_max_per_shift': w.ocra_max_per_shift,
                'ergonomic_tolerance': w.ergonomic_tolerance,
            }
            for w in instance.workers
        ]

        # Add job summary (not full details for brevity)
        data['jobs_summary'] = [
            {
                'job_id': j.job_id,
                'n_operations': len(j.operations),
                'arrival_time': j.arrival_time,
                'due_date': j.due_date,
                'weight': j.weight,
            }
            for j in instance.jobs
        ]

        # Add ergonomic risk summary
        data['ergonomic_risks'] = {
            f"{k[0]}_{k[1]}": v for k, v in instance.ergonomic_risk_map.items()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved instance to {filepath}")

    def save_suite(
        self,
        instances: List[SFJSSPInstance],
        output_dir: str
    ):
        """Save a suite of instances to a directory"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        for instance in instances:
            filepath = os.path.join(output_dir, f"{instance.instance_id}.json")
            self.save_instance(instance, filepath)

        # Save suite metadata
        metadata = {
            'n_instances': len(instances),
            'sizes': list(set(i.instance_type.value for i in instances)),
            'generation_date': datetime.now().isoformat(),
            'generator_version': "1.0",
            'config': {
                'is_dynamic': self.config.is_dynamic,
                'calibration_sources': self.config.calibration_sources,
            },
        }

        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(instances)} instances to {output_dir}")
