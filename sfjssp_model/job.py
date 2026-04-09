"""
Job and Operation data structures for SFJSSP

Evidence Status:
- Basic FJSSP structure: CONFIRMED from standard literature
- Due dates and weights: CONFIRMED from standard literature
- SFJSSP-specific extensions: PROPOSED
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
import numpy as np


@dataclass
class Operation:
    """
    An operation within a job

    In SFJSSP, each operation requires:
    - One machine (from eligible set)
    - One worker (from eligible set)
    - Specific machine mode
    """
    job_id: int
    op_id: int  # Position within job (0-indexed)

    # Processing requirements
    # Map: machine_id -> {mode_id: processing_time}
    processing_times: Dict[int, Dict[int, float]] = field(default_factory=dict)

    # --- ADDED: Precedence-related delays (PDF Section 5.2.3) ---
    transport_time: float = 0.0
    waiting_time: float = 0.0

    # Eligible resources (CONFIRMED from DRCFJSSP literature)
    eligible_machines: Set[int] = field(default_factory=set)
    eligible_workers: Set[int] = field(default_factory=set)

    # Operation state (PROPOSED for dynamic scheduling)
    start_time: float = 0.0
    completion_time: float = 0.0
    assigned_machine: int = -1
    assigned_worker: int = -1
    assigned_mode: int = -1

    # --- ADDED: Task-related period constraints (PDF Section 5.2.3) ---
    period_start: float = 0.0
    period_end: float = 0.0

    # Status tracking
    is_completed: bool = False
    is_scheduled: bool = False

    def is_within_period(self) -> bool:
        """
        Check if the scheduled operation stays strictly within its assigned period bounds.
        """
        if self.start_time is None or self.completion_time is None:
            return False
        if self.period_start is not None and self.start_time < self.period_start:
            return False
        if self.period_end is not None and self.completion_time > self.period_end:
            return False
        return True

    def assign_period_bounds(self, clock) -> None:
        """
        Auto-assign period_start and period_end from a PeriodClock
        based on the operation's scheduled start time.

        Call this immediately after start_time and completion_time are set.
        Raises ValueError if the operation crosses a period boundary,
        since a single operation should not span two shifts.
        """
        if self.start_time is None:
            raise ValueError("Cannot assign period bounds before start_time is set.")

        period_idx = clock.get_period(self.start_time)
        self.period_start = clock.period_start(period_idx)
        self.period_end   = clock.period_end(period_idx)

        # Enforce: operation must not cross into the next period
        if self.completion_time is not None and self.completion_time > self.period_end:
            raise ValueError(
                f"Operation ({self.job_id},{self.op_id}) crosses period boundary: "
                f"completion={self.completion_time:.1f} > period_end={self.period_end:.1f}. "
                f"Reduce processing time or split across periods."
            )

    def get_processing_time(self, machine_id: int, mode_id: int,
                           worker_efficiency: float = 1.0) -> float:
        """
        Get processing time considering machine, mode, and worker efficiency

        Evidence: Processing time adjustment by worker efficiency
        CONFIRMED in DRCFJSSP literature
        """
        if machine_id not in self.processing_times:
            raise ValueError(f"Machine {machine_id} not eligible for operation")

        if mode_id not in self.processing_times[machine_id]:
            raise ValueError(f"Mode {mode_id} not available on machine {machine_id}")

        base_time = self.processing_times[machine_id][mode_id]

        # Adjust for worker efficiency
        # Evidence: Worker efficiency affects processing time [CONFIRMED]
        adjusted_time = base_time / worker_efficiency

        return adjusted_time

    def to_dict(self) -> dict:
        """Convert operation to dictionary for serialization"""
        return {
            'job_id': self.job_id,
            'op_id': self.op_id,
            'processing_times': {str(m): {str(k): v for k, v in modes.items()} for m, modes in self.processing_times.items()},
            'transport_time': self.transport_time,
            'waiting_time': self.waiting_time,
            'eligible_machines': list(self.eligible_machines),
            'eligible_workers': list(self.eligible_workers),
            'start_time': self.start_time,
            'completion_time': self.completion_time,
            'assigned_machine': self.assigned_machine,
            'assigned_worker': self.assigned_worker,
            'assigned_mode': self.assigned_mode,
            'is_completed': self.is_completed,
            'is_scheduled': self.is_scheduled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Operation':
        """Create operation from dictionary"""
        op = cls(
            job_id=data['job_id'],
            op_id=data['op_id'],
            transport_time=data.get('transport_time', 0.0),
            waiting_time=data.get('waiting_time', 0.0),
        )
        # Convert machine IDs back to int
        raw_pt = data.get('processing_times', {})
        op.processing_times = {int(m): {int(k): v for k, v in modes.items()} for m, modes in raw_pt.items()}
        op.eligible_machines = set(data.get('eligible_machines', []))
        op.eligible_workers = set(data.get('eligible_workers', []))
        op.start_time = data.get('start_time')
        op.completion_time = data.get('completion_time')
        op.assigned_machine = data.get('assigned_machine')
        op.assigned_worker = data.get('assigned_worker')
        op.assigned_mode = data.get('assigned_mode')
        op.is_completed = data.get('is_completed', False)
        op.is_scheduled = data.get('is_scheduled', False)
        return op

    def is_ready(self, completed_ops: Set[int]) -> bool:
        """
        Check if operation is ready to start (predecessors completed)

        Evidence: Precedence constraints CONFIRMED in standard FJSSP
        """
        return self.op_id == 0 or (self.job_id, self.op_id - 1) in completed_ops

    def reset(self):
        """Reset operation to unscheduled state"""
        self.start_time = 0.0
        self.completion_time = 0.0
        self.assigned_machine = -1
        self.assigned_worker = -1
        self.assigned_mode = -1
        self.is_completed = False
        self.is_scheduled = False

    def __hash__(self):
        return hash((self.job_id, self.op_id))

    def __eq__(self, other):
        if not isinstance(other, Operation):
            return False
        return self.job_id == other.job_id and self.op_id == other.op_id


@dataclass
class Job:
    """
    A job consisting of ordered operations

    In SFJSSP, jobs arrive dynamically and have due dates
    """
    job_id: int
    operations: List[Operation] = field(default_factory=list)

    # Job characteristics (CONFIRMED from standard FJSSP)
    arrival_time: float = 0.0
    due_date: Optional[float] = None
    weight: float = 1.0  # Importance weight for tardiness calculation

    # Job state
    is_completed: bool = False
    completion_time: float = 0.0

    def get_tardiness(self) -> float:
        """
        Calculate tardiness (max(0, completion - due_date))

        Evidence: Tardiness calculation CONFIRMED in standard FJSSP
        """
        if self.due_date is None:
            return 0.0
        # completion_time is now guaranteed 0.0 or float
        return max(0.0, self.completion_time - self.due_date)

    def get_total_processing_time(self) -> float:
        """Sum of minimum processing times across all operations"""
        total = 0.0
        for op in self.operations:
            min_time = float('inf')
            for machine_times in op.processing_times.values():
                min_time = min(min_time, min(machine_times.values()))
            if min_time < float('inf'):
                total += min_time
        return total

    def to_dict(self) -> dict:
        """Convert job to dictionary for serialization"""
        return {
            'job_id': self.job_id,
            'operations': [op.to_dict() for op in self.operations],
            'arrival_time': self.arrival_time,
            'due_date': self.due_date,
            'weight': self.weight,
            'is_completed': self.is_completed,
            'completion_time': self.completion_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Job':
        """Create job from dictionary"""
        job = cls(
            job_id=data['job_id'],
            arrival_time=data.get('arrival_time', 0.0),
            due_date=data.get('due_date'),
            weight=data.get('weight', 1.0),
        )
        job.operations = [Operation.from_dict(op_data) for op_data in data.get('operations', [])]
        job.is_completed = data.get('is_completed', False)
        job.completion_time = data.get('completion_time')
        return job

    def check_is_completed(self) -> bool:
        """Check if all operations are completed"""
        return all(op.is_completed for op in self.operations)

    def reset(self):
        """Reset job to initial state"""
        self.is_completed = False
        self.completion_time = 0.0
        for op in self.operations:
            op.reset()

    def __hash__(self):
        return hash(self.job_id)

    def __eq__(self, other):
        if not isinstance(other, Job):
            return False
        return self.job_id == other.job_id