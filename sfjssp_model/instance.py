"""
SFJSSP Instance data structure

Evidence Status:
- Instance structure: PROPOSED synthesis of FJSSP + DRCFJSSP + SFJSSP
- Dynamic event parameters: CONFIRMED from dynamic FJSSP literature
- Labeling system: PROPOSED for transparency
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
import numpy as np

from .job import Job, Operation
from .machine import Machine, MachineMode, MachineState
from .worker import Worker, WorkerSkill, WorkerState
from .period_clock import PeriodClock


class InstanceLabel(Enum):
    """
    Dataset labeling system for transparency

    Evidence: Proposed labeling protocol for synthetic and calibrated instances
    """
    REAL_INDUSTRIAL = "real_industrial"
    CALIBRATED_SYNTHETIC = "calibrated_synthetic"
    EXTENDED_SYNTHETIC = "extended_synthetic"
    FULLY_SYNTHETIC = "fully_synthetic"
    ACQUISITION_UNCERTAIN = "acquisition_uncertain"


class InstanceType(Enum):
    """Instance classification"""
    STATIC = "static"  # All jobs known at start
    DYNAMIC = "dynamic"  # Jobs arrive over time


@dataclass
class DynamicEventParams:
    """
    Parameters for dynamic event generation

    Evidence:
    - Job arrivals: Poisson process [CONFIRMED dynamic FJSSP]
    - Breakdowns: Exponential distribution [CONFIRMED dynamic FJSSP]
    """
    # Job arrival process
    arrival_rate: float = 0.1  # lambda: jobs per time unit (Poisson)

    # Machine breakdown process
    breakdown_rate: float = 0.001  # failures per machine per time unit
    repair_rate: float = 0.1  # repairs per time unit (exponential mean)

    # Worker absence process
    absence_probability: float = 0.05  # probability per worker per day

    # Rush order probability
    rush_order_probability: float = 0.1  # probability an arrival is high priority


@dataclass(frozen=True)
class AvailabilityWindow:
    """Explicit unavailability interval for one machine or worker."""

    start_time: float
    end_time: float
    reason: str = "unavailable"
    source: str = "calendar"
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.start_time < 0.0:
            raise ValueError("AvailabilityWindow.start_time must be non-negative")
        if self.end_time < self.start_time:
            raise ValueError("AvailabilityWindow.end_time must be >= start_time")

    def overlaps(self, start_time: float, end_time: float) -> bool:
        """True when the half-open intervals [start, end) overlap."""
        return start_time < self.end_time and end_time > self.start_time

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "reason": self.reason,
            "source": self.source,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AvailabilityWindow":
        return cls(
            start_time=data["start_time"],
            end_time=data["end_time"],
            reason=data.get("reason", "unavailable"),
            source=data.get("source", "calendar"),
            details=data.get("details", {}),
        )


@dataclass(frozen=True)
class MachineBreakdownEvent:
    """Typed machine breakdown event for canonical replay and serialization."""

    machine_id: int
    start_time: float
    repair_duration: float
    source: str = "generated"
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.start_time + self.repair_duration

    def to_availability_window(self) -> AvailabilityWindow:
        details = {"machine_id": self.machine_id, "repair_duration": self.repair_duration}
        details.update(self.details)
        return AvailabilityWindow(
            start_time=self.start_time,
            end_time=self.end_time,
            reason="breakdown",
            source=self.source,
            details=details,
        )

    def to_dict(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "start_time": self.start_time,
            "repair_duration": self.repair_duration,
            "end_time": self.end_time,
            "source": self.source,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MachineBreakdownEvent":
        if "repair_duration" in data:
            repair_duration = data["repair_duration"]
        else:
            repair_duration = data["end_time"] - data["start_time"]
        return cls(
            machine_id=data["machine_id"],
            start_time=data["start_time"],
            repair_duration=repair_duration,
            source=data.get("source", "generated"),
            details=data.get("details", {}),
        )


@dataclass(frozen=True)
class WorkerAbsenceEvent:
    """Typed worker absence event for canonical replay and serialization."""

    worker_id: int
    start_time: float
    end_time: float
    source: str = "generated"
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.start_time < 0.0:
            raise ValueError("WorkerAbsenceEvent.start_time must be non-negative")
        if self.end_time < self.start_time:
            raise ValueError("WorkerAbsenceEvent.end_time must be >= start_time")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_availability_window(self) -> AvailabilityWindow:
        details = {"worker_id": self.worker_id, "duration": self.duration}
        details.update(self.details)
        return AvailabilityWindow(
            start_time=self.start_time,
            end_time=self.end_time,
            reason="absence",
            source=self.source,
            details=details,
        )

    def to_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "source": self.source,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerAbsenceEvent":
        end_time = data.get("end_time")
        if end_time is None:
            end_time = data["start_time"] + data["duration"]
        return cls(
            worker_id=data["worker_id"],
            start_time=data["start_time"],
            end_time=end_time,
            source=data.get("source", "generated"),
            details=data.get("details", {}),
        )


@dataclass
class SFJSSPInstance:
    """
    Complete SFJSSP Problem Instance

    Contains all data needed to define and solve an SFJSSP problem:
    - Jobs with operations and processing requirements
    - Machines with modes and energy parameters
    - Workers with skills and human factors
    - Dynamic event parameters (for dynamic scenarios)
    - Instance metadata and labeling

    Evidence: Instance structure synthesizes:
    - Standard FJSSP [CONFIRMED]
    - DRCFJSSP dual resources [CONFIRMED]
    - Energy parameters from E-DFJSP 2025 [CONFIRMED]
    - Human factors from DyDFJSP 2023 + NSGA-III 2021 [CONFIRMED]
    - Dynamic events from DRL literature [CONFIRMED]
    """
    # Instance identification
    instance_id: str = "SFJSSP_001"
    instance_name: str = ""

    # Labeling for transparency (PROPOSED protocol)
    label: InstanceLabel = InstanceLabel.FULLY_SYNTHETIC
    label_justification: str = "Computer-generated instance"

    # Instance type
    instance_type: InstanceType = InstanceType.STATIC

    # Core problem data
    jobs: List[Job] = field(default_factory=list)
    machines: List[Machine] = field(default_factory=list)
    workers: List[Worker] = field(default_factory=list)

    # Global shared period clock
    period_clock: PeriodClock = field(default_factory=PeriodClock)

    # Ergonomic risk parameters (CONFIRMED from NSGA-III 2021)
    # Map: (job_id, op_id) -> ergonomic risk rate per time unit
    ergonomic_risk_map: Dict[Tuple[int, int], float] = field(default_factory=dict)
    default_ergonomic_risk: float = 0.0

    # Carbon emission factor (CONFIRMED from Low-carbon DRL 2024)
    # kg CO2 per kWh (can be time-varying)
    carbon_emission_factor: float = 0.5  # Default grid average

    # Time-of-use electricity prices (CONFIRMED from energy-aware FJSSP)
    # Map: time_period -> price per kWh
    electricity_prices: Dict[int, float] = field(default_factory=dict)
    default_electricity_price: float = 0.10  # $/kWh

    # Dynamic event parameters (for DYNAMIC instances)
    dynamic_params: Optional[DynamicEventParams] = None

    # Canonical explicit availability windows. These are the internal source of
    # truth for event/calendar-driven resource unavailability.
    machine_unavailability: Dict[int, List[AvailabilityWindow]] = field(default_factory=dict)
    worker_unavailability: Dict[int, List[AvailabilityWindow]] = field(default_factory=dict)
    machine_breakdown_events: List[MachineBreakdownEvent] = field(default_factory=list)
    worker_absence_events: List[WorkerAbsenceEvent] = field(default_factory=list)

    # Instance statistics (computed)
    n_jobs: int = 0
    n_machines: int = 0
    n_workers: int = 0
    n_operations: int = 0

    # Planning horizon (for static instances)
    planning_horizon: float = 1000.0

    # Metadata
    creation_date: str = ""
    source: str = ""  # e.g., "MK01 extended", "Gong et al. 2018"
    calibration_sources: List[str] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)

    # Auxiliary parameters (PROPOSED)
    auxiliary_power_total: float = 50.0  # Total facility auxiliary power (kW)

    def __post_init__(self):
        """Update statistics after initialization and sync clocks"""
        self._update_statistics()
        self.validate_risk_map()

        # Sync all workers to the shared period clock
        for worker in self.workers:
            worker.period_clock = self.period_clock

    def validate_risk_map(self):
        """
        Warn if any risk_rate implies a worker hits OCRA limit
        in less than 30 minutes (likely miscalibrated).
        """
        OCRA_UNIT_BUDGET = 2.2   # max allowed per shift (JMSY-9 §5.2.3)
        SHIFT_MINUTES   = 480.0  # 8 hours

        for (job_id, op_id), rate in self.ergonomic_risk_map.items():
            if rate > 0:
                time_to_limit = OCRA_UNIT_BUDGET / rate
                if time_to_limit < 30.0:
                    import warnings
                    warnings.warn(
                        f"Op ({job_id},{op_id}): risk_rate={rate:.4f} hits OCRA "
                        f"limit in {time_to_limit:.1f} min. "
                        f"Consider calibrating to ~{OCRA_UNIT_BUDGET / SHIFT_MINUTES:.5f} "
                        f"for a full-shift limit."
                    )

    def _update_statistics(self):
        """Compute instance statistics"""
        self.n_jobs = len(self.jobs)
        self.n_machines = len(self.machines)
        self.n_workers = len(self.workers)
        self.n_operations = sum(len(job.operations) for job in self.jobs)

    def add_job(self, job: Job):
        """Add a job to the instance"""
        self.jobs.append(job)
        self._update_statistics()

    def add_machine(self, machine: Machine):
        """Add a machine to the instance"""
        self.machines.append(machine)
        self._update_statistics()

    def add_worker(self, worker: Worker):
        """Add a worker to the instance"""
        worker.period_clock = self.period_clock
        self.workers.append(worker)
        self._update_statistics()

    def get_job(self, job_id: int) -> Optional[Job]:
        """Get job by ID"""
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        return None

    def get_machine(self, machine_id: int) -> Optional[Machine]:
        """Get machine by ID"""
        for machine in self.machines:
            if machine.machine_id == machine_id:
                return machine
        return None

    def get_worker(self, worker_id: int) -> Optional[Worker]:
        """Get worker by ID"""
        for worker in self.workers:
            if worker.worker_id == worker_id:
                return worker
        return None

    @staticmethod
    def _sort_windows(windows: List[AvailabilityWindow]) -> List[AvailabilityWindow]:
        return sorted(
            windows,
            key=lambda window: (
                window.start_time,
                window.end_time,
                window.reason,
                window.source,
            ),
        )

    def add_machine_unavailability(
        self,
        machine_id: int,
        start_time: float,
        end_time: float,
        *,
        reason: str = "calendar_unavailable",
        source: str = "calendar",
        details: Optional[Dict[str, Any]] = None,
    ) -> AvailabilityWindow:
        """Register one explicit machine-unavailability interval."""
        if self.get_machine(machine_id) is None:
            raise ValueError(f"Unknown machine_id {machine_id!r}")

        window = AvailabilityWindow(
            start_time=start_time,
            end_time=end_time,
            reason=reason,
            source=source,
            details=details or {},
        )
        windows = self.machine_unavailability.setdefault(machine_id, [])
        windows.append(window)
        self.machine_unavailability[machine_id] = self._sort_windows(windows)
        return window

    def add_worker_unavailability(
        self,
        worker_id: int,
        start_time: float,
        end_time: float,
        *,
        reason: str = "calendar_unavailable",
        source: str = "calendar",
        details: Optional[Dict[str, Any]] = None,
    ) -> AvailabilityWindow:
        """Register one explicit worker-unavailability interval."""
        if self.get_worker(worker_id) is None:
            raise ValueError(f"Unknown worker_id {worker_id!r}")

        window = AvailabilityWindow(
            start_time=start_time,
            end_time=end_time,
            reason=reason,
            source=source,
            details=details or {},
        )
        windows = self.worker_unavailability.setdefault(worker_id, [])
        windows.append(window)
        self.worker_unavailability[worker_id] = self._sort_windows(windows)
        return window

    def add_machine_breakdown_event(
        self,
        machine_id: int,
        start_time: float,
        repair_duration: float,
        *,
        source: str = "event",
        details: Optional[Dict[str, Any]] = None,
    ) -> MachineBreakdownEvent:
        """Register a typed machine-breakdown event and its derived window."""
        if self.get_machine(machine_id) is None:
            raise ValueError(f"Unknown machine_id {machine_id!r}")

        event = MachineBreakdownEvent(
            machine_id=machine_id,
            start_time=start_time,
            repair_duration=repair_duration,
            source=source,
            details=details or {},
        )
        self.machine_breakdown_events.append(event)
        self.machine_breakdown_events.sort(
            key=lambda item: (item.start_time, item.end_time, item.machine_id)
        )
        self.add_machine_unavailability(
            machine_id,
            event.start_time,
            event.end_time,
            reason="breakdown",
            source=source,
            details=event.to_dict(),
        )
        return event

    def add_worker_absence_event(
        self,
        worker_id: int,
        start_time: float,
        end_time: float,
        *,
        source: str = "event",
        details: Optional[Dict[str, Any]] = None,
    ) -> WorkerAbsenceEvent:
        """Register a typed worker-absence event and its derived window."""
        if self.get_worker(worker_id) is None:
            raise ValueError(f"Unknown worker_id {worker_id!r}")

        event = WorkerAbsenceEvent(
            worker_id=worker_id,
            start_time=start_time,
            end_time=end_time,
            source=source,
            details=details or {},
        )
        self.worker_absence_events.append(event)
        self.worker_absence_events.sort(
            key=lambda item: (item.start_time, item.end_time, item.worker_id)
        )
        self.add_worker_unavailability(
            worker_id,
            event.start_time,
            event.end_time,
            reason="absence",
            source=source,
            details=event.to_dict(),
        )
        return event

    def get_machine_unavailability(self, machine_id: int) -> List[AvailabilityWindow]:
        """Return canonical machine-unavailability windows."""
        return list(self.machine_unavailability.get(machine_id, []))

    def get_worker_unavailability(self, worker_id: int) -> List[AvailabilityWindow]:
        """Return canonical worker-unavailability windows."""
        return list(self.worker_unavailability.get(worker_id, []))

    def get_machine_conflicting_windows(
        self,
        machine_id: int,
        start_time: float,
        end_time: float,
    ) -> List[AvailabilityWindow]:
        return [
            window
            for window in self.machine_unavailability.get(machine_id, [])
            if window.overlaps(start_time, end_time)
        ]

    def get_worker_conflicting_windows(
        self,
        worker_id: int,
        start_time: float,
        end_time: float,
    ) -> List[AvailabilityWindow]:
        return [
            window
            for window in self.worker_unavailability.get(worker_id, [])
            if window.overlaps(start_time, end_time)
        ]

    def get_operation(self, job_id: int, op_id: int) -> Optional[Operation]:
        """Get operation by job and operation ID"""
        job = self.get_job(job_id)
        if job is None:
            return None
        if op_id < 0 or op_id >= len(job.operations):
            return None
        return job.operations[op_id]

    def get_eligible_workers(self, job_id: int, op_id: int) -> List[int]:
        """Get list of worker IDs eligible for an operation"""
        op = self.get_operation(job_id, op_id)
        if op is None:
            return []

        eligible = [
            worker.worker_id
            for worker in self.workers
            if (job_id, op_id) in worker.eligible_operations
        ]
        if eligible:
            return eligible
        return list(op.eligible_workers)

    def get_eligible_machines(self, job_id: int, op_id: int) -> List[int]:
        """Get list of machine IDs eligible for an operation"""
        op = self.get_operation(job_id, op_id)
        if op is None:
            return []
        return list(op.eligible_machines)

    def get_ergonomic_risk(self, job_id: int, op_id: int) -> float:
        """Get ergonomic risk rate for an operation"""
        return self.ergonomic_risk_map.get((job_id, op_id), self.default_ergonomic_risk)

    def get_electricity_price(self, time: float) -> float:
        """Get electricity price at a given time"""
        if not self.electricity_prices:
            return self.default_electricity_price

        # Find the appropriate time period
        time_int = int(time)
        if time_int in self.electricity_prices:
            return self.electricity_prices[time_int]

        # Default to average if not found
        return sum(self.electricity_prices.values()) / len(self.electricity_prices)

    def get_carbon_factor(self, time: float = 0.0) -> float:
        """Get carbon emission factor (can be time-varying)"""
        return self.carbon_emission_factor

    def get_auxiliary_power_per_machine(self) -> float:
        """Get auxiliary power allocation per machine"""
        if self.n_machines == 0:
            return 0.0
        return self.auxiliary_power_total / self.n_machines

    def generate_dynamic_job(self, current_time: float, rng: np.random.Generator) -> Optional[Job]:
        """
        Generate a new job arrival for dynamic scenarios

        Evidence: Dynamic job arrivals modeled as Poisson process [CONFIRMED]

        Args:
            current_time: Current simulation time
            rng: NumPy random generator

        Returns:
            Job or None if no arrival
        """
        if self.dynamic_params is None:
            return None
        if not self.machines or not self.workers:
            return None

        # Poisson arrival: P(arrival) = 1 - exp(-lambda * dt)
        # Using dt=1 for simplicity
        arrival_prob = 1 - np.exp(-self.dynamic_params.arrival_rate)

        if rng.random() > arrival_prob:
            return None

        # Generate new job ID
        new_job_id = max(job.job_id for job in self.jobs) + 1 if self.jobs else 0

        # Create new job with random operations
        # Note: This is a simplified generator; real instances should have
        # more sophisticated job generation logic
        n_ops = rng.integers(2, 6)  # 2-5 operations
        operations = []

        for op_idx in range(n_ops):
            op = Operation(
                job_id=new_job_id,
                op_id=op_idx,
            )

            # Assign random eligible machines and workers
            machine_ids = [machine.machine_id for machine in self.machines]
            worker_ids = [worker.worker_id for worker in self.workers]
            n_machines = rng.integers(1, min(4, len(machine_ids)) + 1)
            n_workers = rng.integers(1, min(4, len(worker_ids)) + 1)

            eligible_machines = list(rng.choice(
                machine_ids, size=n_machines, replace=False
            ))
            eligible_workers = list(rng.choice(
                worker_ids, size=n_workers, replace=False
            ))

            op.eligible_machines = set(eligible_machines)
            op.eligible_workers = set(eligible_workers)

            # Generate processing times
            for m_id in eligible_machines:
                op.processing_times[m_id] = {}
                machine = self.get_machine(m_id)
                if machine and machine.modes:
                    for mode in machine.modes:
                        # Processing time varies by mode speed
                        base_time = rng.uniform(10, 100)
                        op.processing_times[m_id][mode.mode_id] = (
                            base_time / mode.speed_factor
                        )
                else:
                    op.processing_times[m_id] = {0: rng.uniform(10, 100)}

            operations.append(op)

        # Job characteristics
        due_date_margin = rng.uniform(1.5, 3.0)  # Due date = arrival + margin * total_processing
        is_rush = rng.random() < self.dynamic_params.rush_order_probability

        job = Job(
            job_id=new_job_id,
            operations=operations,
            arrival_time=current_time,
            due_date=current_time + due_date_margin * sum(
                min(min(modes.values()) for modes in op.processing_times.values())
                for op in operations
            ),
            weight=2.0 if is_rush else 1.0
        )

        return job

    def generate_breakdown_event(
        self,
        current_time: float,
        rng: np.random.Generator
    ) -> Optional[Tuple[int, float, float]]:
        """
        Generate a machine breakdown event

        Evidence: Breakdowns modeled as exponential process [CONFIRMED]

        Args:
            current_time: Current simulation time
            rng: NumPy random generator

        Returns:
            Tuple of (machine_id, breakdown_time, repair_duration) or None
        """
        if self.dynamic_params is None:
            return None

        event = self.generate_breakdown_record(current_time, rng)
        if event is None:
            return None
        return (event.machine_id, event.start_time, event.repair_duration)

    def generate_breakdown_record(
        self,
        current_time: float,
        rng: np.random.Generator
    ) -> Optional[MachineBreakdownEvent]:
        """
        Generate a typed machine-breakdown event without mutating instance state.
        """
        if self.dynamic_params is None:
            return None

        # Check each available machine for breakdown
        for machine in self.machines:
            if machine.is_broken:
                continue

            # Exponential failure process
            failure_prob = 1 - np.exp(-self.dynamic_params.breakdown_rate)

            if rng.random() < failure_prob:
                # Generate repair time (exponential distribution)
                repair_duration = rng.exponential(1.0 / self.dynamic_params.repair_rate)
                return MachineBreakdownEvent(
                    machine_id=machine.machine_id,
                    start_time=current_time,
                    repair_duration=repair_duration,
                    source="generated",
                )

        return None

    def generate_absence_event(
        self,
        current_time: float,
        rng: np.random.Generator,
    ) -> Optional[Tuple[int, float, float]]:
        """
        Generate a worker absence event at shift boundaries for dynamic instances.

        `absence_probability` is interpreted as the probability that one worker
        becomes unavailable during a shift boundary review.
        """
        event = self.generate_absence_record(current_time, rng)
        if event is None:
            return None
        return (event.worker_id, event.start_time, event.duration)

    def generate_absence_record(
        self,
        current_time: float,
        rng: np.random.Generator,
    ) -> Optional[WorkerAbsenceEvent]:
        """Generate a typed worker-absence event without mutating instance state."""
        if self.dynamic_params is None or not self.workers:
            return None

        current_period = self.period_clock.get_period(current_time)
        if abs(current_time - self.period_clock.period_start(current_period)) > 1e-9:
            return None

        available_workers = [
            worker
            for worker in self.workers
            if not worker.is_absent and worker.available_time <= current_time
        ]
        if not available_workers:
            return None

        if rng.random() >= self.dynamic_params.absence_probability:
            return None

        worker = available_workers[int(rng.integers(0, len(available_workers)))]
        duration = float(rng.uniform(0.25, 1.0) * worker.SHIFT_DURATION)
        return WorkerAbsenceEvent(
            worker_id=worker.worker_id,
            start_time=current_time,
            end_time=current_time + duration,
            source="generated",
        )

    def reset(self):
        """Reset all entities to initial state"""
        for job in self.jobs:
            job.reset()
        for machine in self.machines:
            machine.reset()
        for worker in self.workers:
            worker.reset()

    def to_dict(self) -> dict:
        """Convert instance to dictionary for serialization"""
        return {
            'instance_id': self.instance_id,
            'instance_name': self.instance_name,
            'label': self.label.value,
            'label_justification': self.label_justification,
            'instance_type': self.instance_type.value,
            'jobs': [j.to_dict() for j in self.jobs],
            'machines': [m.to_dict() for m in self.machines],
            'workers': [w.to_dict() for w in self.workers],
            'planning_horizon': self.planning_horizon,
            'creation_date': self.creation_date,
            'source': self.source,
            'calibration_sources': self.calibration_sources,
            'known_limitations': self.known_limitations,
            'carbon_emission_factor': self.carbon_emission_factor,
            'electricity_prices': self.electricity_prices,
            'default_electricity_price': self.default_electricity_price,
            'auxiliary_power_total': self.auxiliary_power_total,
            'default_ergonomic_risk': self.default_ergonomic_risk,
            'ergonomic_risks': {f"{k[0]}_{k[1]}": v for k, v in self.ergonomic_risk_map.items()},
            'machine_unavailability': {
                str(machine_id): [window.to_dict() for window in windows]
                for machine_id, windows in self.machine_unavailability.items()
            },
            'worker_unavailability': {
                str(worker_id): [window.to_dict() for window in windows]
                for worker_id, windows in self.worker_unavailability.items()
            },
            'machine_breakdown_events': [
                event.to_dict() for event in self.machine_breakdown_events
            ],
            'worker_absence_events': [
                event.to_dict() for event in self.worker_absence_events
            ],
            'dynamic_params': {
                'arrival_rate': self.dynamic_params.arrival_rate,
                'breakdown_rate': self.dynamic_params.breakdown_rate,
                'repair_rate': self.dynamic_params.repair_rate,
                'absence_probability': self.dynamic_params.absence_probability,
                'rush_order_probability': self.dynamic_params.rush_order_probability,
            } if self.dynamic_params else None,
        }

    def to_json(self, filepath: str):
        """Save instance to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'SFJSSPInstance':
        """
        Create instance from dictionary with full reconstruction.
        """
        dynamic_params = None
        if data.get('dynamic_params'):
            dynamic_params = DynamicEventParams(
                arrival_rate=data['dynamic_params'].get('arrival_rate', 0.1),
                breakdown_rate=data['dynamic_params'].get('breakdown_rate', 0.001),
                repair_rate=data['dynamic_params'].get('repair_rate', 0.1),
                absence_probability=data['dynamic_params'].get('absence_probability', 0.05),
                rush_order_probability=data['dynamic_params'].get('rush_order_probability', 0.1),
            )

        instance = cls(
            instance_id=data.get('instance_id', 'SFJSSP_001'),
            instance_name=data.get('instance_name', ''),
            label=InstanceLabel(data.get('label', 'fully_synthetic')),
            label_justification=data.get('label_justification', ''),
            instance_type=InstanceType(data.get('instance_type', 'static')),
            planning_horizon=data.get('planning_horizon', 1000.0),
            creation_date=data.get('creation_date', ''),
            source=data.get('source', ''),
            calibration_sources=data.get('calibration_sources', []),
            known_limitations=data.get('known_limitations', []),
            carbon_emission_factor=data.get('carbon_emission_factor', 0.5),
            electricity_prices={int(k): v for k, v in data.get('electricity_prices', {}).items()},
            default_electricity_price=data.get('default_electricity_price', 0.10),
            auxiliary_power_total=data.get('auxiliary_power_total', 50.0),
            default_ergonomic_risk=data.get('default_ergonomic_risk', 0.0),
            dynamic_params=dynamic_params,
        )
        
        # Recursive reconstruction
        instance.machines = [Machine.from_dict(m_data) for m_data in data.get('machines', [])]
        instance.workers = [Worker.from_dict(w_data) for w_data in data.get('workers', [])]
        instance.jobs = [Job.from_dict(j_data) for j_data in data.get('jobs', [])]
        for worker in instance.workers:
            worker.period_clock = instance.period_clock

        raw_machine_windows = data.get('machine_unavailability', {})
        raw_worker_windows = data.get('worker_unavailability', {})
        instance.machine_unavailability = {
            int(machine_id): cls._sort_windows(
                [AvailabilityWindow.from_dict(window) for window in windows]
            )
            for machine_id, windows in raw_machine_windows.items()
        }
        instance.worker_unavailability = {
            int(worker_id): cls._sort_windows(
                [AvailabilityWindow.from_dict(window) for window in windows]
            )
            for worker_id, windows in raw_worker_windows.items()
        }

        instance.machine_breakdown_events = [
            MachineBreakdownEvent.from_dict(event_data)
            for event_data in data.get('machine_breakdown_events', [])
        ]
        instance.worker_absence_events = [
            WorkerAbsenceEvent.from_dict(event_data)
            for event_data in data.get('worker_absence_events', [])
        ]

        if not raw_machine_windows:
            for event in instance.machine_breakdown_events:
                instance.add_machine_unavailability(
                    event.machine_id,
                    event.start_time,
                    event.end_time,
                    reason="breakdown",
                    source=event.source,
                    details=event.to_dict(),
                )
        if not raw_worker_windows:
            for event in instance.worker_absence_events:
                instance.add_worker_unavailability(
                    event.worker_id,
                    event.start_time,
                    event.end_time,
                    reason="absence",
                    source=event.source,
                    details=event.to_dict(),
                )
        
        # Risk map reconstruction
        raw_risks = data.get('ergonomic_risks')
        if raw_risks is None:
            raw_risks = data.get('ergonomic_risk_map', {})
        for k_str, val in raw_risks.items():
            key_text = k_str.strip()
            if key_text.startswith("(") and key_text.endswith(")"):
                jid_text, oid_text = [part.strip() for part in key_text[1:-1].split(",")]
            else:
                jid_text, oid_text = key_text.split("_")
            jid, oid = int(jid_text), int(oid_text)
            instance.ergonomic_risk_map[(jid, oid)] = val
            
        instance._update_statistics()
        return instance

    def __repr__(self):
        return (
            f"SFJSSPInstance(id='{self.instance_id}', "
            f"jobs={self.n_jobs}, machines={self.n_machines}, "
            f"workers={self.n_workers}, ops={self.n_operations})"
        )
