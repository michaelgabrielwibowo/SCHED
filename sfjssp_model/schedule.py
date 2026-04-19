"""
Schedule data structure for SFJSSP

Evidence Status:
- Schedule representation: PROPOSED synthesis
- Objective evaluation: Proposed multi-objective synthesis
- Constraint checking: Based on literature constraints [CONFIRMED components]
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from .instance import SFJSSPInstance
from .job import Job, Operation
from .machine import Machine, MachineState
from .worker import Worker, WorkerState


@dataclass
class ScheduledOperation:
    """
    An operation with assigned resources and timing

    Evidence: Assignment variables from mathematical model [CONFIRMED components]
    """
    job_id: int
    op_id: int

    # Assignment decisions
    machine_id: int
    worker_id: int
    mode_id: int

    # Timing decisions
    start_time: float
    completion_time: float
    processing_time: float

    # Setup time (if applicable)
    setup_time: float = 0.0

    # Transport time (if applicable)
    transport_time: float = 0.0

    def __repr__(self):
        return (
            f"Op({self.job_id},{self.op_id}): M{self.machine_id}:"
            f"W{self.worker_id} [{self.start_time:.1f}-{self.completion_time:.1f}]"
        )


@dataclass
class MachineSchedule:
    """Schedule for a single machine"""
    machine_id: int
    operations: List[ScheduledOperation] = field(default_factory=list)

    def add_operation(self, op: ScheduledOperation):
        """Add operation and keep sorted by start time"""
        self.operations.append(op)
        self.operations.sort(key=lambda x: x.start_time)

    def get_idle_periods(self, horizon: float) -> List[Tuple[float, float]]:
        """Get idle time periods"""
        if not self.operations:
            return [(0.0, horizon)]

        idle_periods = []
        last_end = 0.0

        for op in self.operations:
            if op.start_time > last_end:
                idle_periods.append((last_end, op.start_time))
            last_end = op.completion_time

        if last_end < horizon:
            idle_periods.append((last_end, horizon))

        return idle_periods


@dataclass
class WorkerSchedule:
    """Schedule for a single worker"""
    worker_id: int
    operations: List[ScheduledOperation] = field(default_factory=list)

    def add_operation(self, op: ScheduledOperation):
        """Add operation and keep sorted by start time"""
        self.operations.append(op)
        self.operations.sort(key=lambda x: x.start_time)


@dataclass(frozen=True)
class ConstraintViolation:
    """Canonical structured hard-feasibility violation."""

    code: str
    message: str
    job_id: Optional[int] = None
    op_id: Optional[int] = None
    machine_id: Optional[int] = None
    worker_id: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize one violation record."""
        return {
            "code": self.code,
            "message": self.message,
            "job_id": self.job_id,
            "op_id": self.op_id,
            "machine_id": self.machine_id,
            "worker_id": self.worker_id,
            "details": dict(self.details),
        }


@dataclass
class Schedule:
    """
    Complete schedule for SFJSSP instance

    Contains:
    - Assignment of operations to machines, workers, and modes
    - Start and completion times for all operations
    - Computed objective values

    Evidence: Schedule structure synthesizes:
    - Standard FJSSP schedule representation [CONFIRMED]
    - Dual-resource assignment from DRCFJSSP [CONFIRMED]
    - Energy tracking from E-DFJSP 2025 [CONFIRMED]
    - Human factor tracking from DyDFJSP 2023 + NSGA-III 2021 [CONFIRMED]
    """
    instance_id: str

    # Operation schedules
    # Map: (job_id, op_id) -> ScheduledOperation
    scheduled_ops: Dict[Tuple[int, int], ScheduledOperation] = field(
        default_factory=dict
    )

    # Machine-wise schedules
    machine_schedules: Dict[int, MachineSchedule] = field(default_factory=dict)

    # Worker-wise schedules
    worker_schedules: Dict[int, WorkerSchedule] = field(default_factory=dict)

    # Schedule metadata
    makespan: float = 0.0
    is_feasible: bool = True
    constraint_violations: List[str] = field(default_factory=list)
    constraint_violation_details: List[ConstraintViolation] = field(default_factory=list)

    # Objective values (computed)
    objectives: Dict[str, float] = field(default_factory=dict)

    # Solver/export metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Energy consumption breakdown
    energy_breakdown: Dict[str, float] = field(default_factory=dict)

    # Human factor metrics
    ergonomic_metrics: Dict[str, float] = field(default_factory=dict)
    fatigue_metrics: Dict[str, float] = field(default_factory=dict)

    # Robustness metrics (workload balance and schedule slack)
    robustness_metrics: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _minutes_to_hours(duration: float) -> float:
        """Convert schedule durations stored in minutes to hours."""
        return duration / 60.0

    def add_operation(
        self,
        job_id: int,
        op_id: int,
        machine_id: int,
        worker_id: int,
        mode_id: int,
        start_time: float,
        completion_time: float,
        processing_time: float,
        setup_time: float = 0.0,
        transport_time: float = 0.0
    ):
        """Add a scheduled operation"""
        key = (job_id, op_id)

        scheduled_op = ScheduledOperation(
            job_id=job_id,
            op_id=op_id,
            machine_id=machine_id,
            worker_id=worker_id,
            mode_id=mode_id,
            start_time=start_time,
            completion_time=completion_time,
            processing_time=processing_time,
            setup_time=setup_time,
            transport_time=transport_time
        )

        self.scheduled_ops[key] = scheduled_op

        # Add to machine schedule
        if machine_id not in self.machine_schedules:
            self.machine_schedules[machine_id] = MachineSchedule(machine_id=machine_id)
        self.machine_schedules[machine_id].add_operation(scheduled_op)

        # Add to worker schedule
        if worker_id not in self.worker_schedules:
            self.worker_schedules[worker_id] = WorkerSchedule(worker_id=worker_id)
        self.worker_schedules[worker_id].add_operation(scheduled_op)

    def get_operation(self, job_id: int, op_id: int) -> Optional[ScheduledOperation]:
        """Get scheduled operation by job and op ID"""
        return self.scheduled_ops.get((job_id, op_id))

    def is_operation_scheduled(self, job_id: int, op_id: int) -> bool:
        """Check if operation is scheduled"""
        return (job_id, op_id) in self.scheduled_ops

    def get_job_completion_time(self, job_id: int, instance: SFJSSPInstance) -> float:
        """Get completion time of a job (last operation)"""
        job = instance.get_job(job_id)
        if job is None:
            return 0.0

        last_op_id = len(job.operations) - 1
        scheduled_op = self.get_operation(job_id, last_op_id)

        if scheduled_op is None:
            return 0.0

        return scheduled_op.completion_time

    def get_job_tardiness(self, job_id: int, instance: SFJSSPInstance) -> float:
        """Calculate tardiness for a job"""
        job = instance.get_job(job_id)
        if job is None or job.due_date is None:
            return 0.0

        completion = self.get_job_completion_time(job_id, instance)
        return max(0.0, completion - job.due_date)

    def compute_makespan(self) -> float:
        """Compute makespan (maximum completion time)"""
        if not self.scheduled_ops:
            self.makespan = 0.0
            return 0.0

        self.makespan = max(
            (op.completion_time or 0.0) for op in self.scheduled_ops.values()
        )
        return self.makespan

    def get_ready_time_for_assignment(
        self,
        instance: SFJSSPInstance,
        job_id: int,
        op_id: int,
        next_machine_id: Optional[int] = None,
    ) -> float:
        """Return the precedence-ready time for one operation assignment."""
        job = instance.get_job(job_id)
        if job is None:
            return float("inf")
        if op_id == 0:
            return job.arrival_time

        prev_sched = self.get_operation(job_id, op_id - 1)
        if prev_sched is None:
            return float("inf")

        prev_model_op = job.operations[op_id - 1]
        waiting_time = getattr(prev_model_op, "waiting_time", 0.0)
        transport_time = 0.0
        if next_machine_id is None:
            transport_time = prev_sched.transport_time
        elif prev_sched.machine_id != next_machine_id:
            transport_time = prev_sched.transport_time or getattr(
                prev_model_op,
                "transport_time",
                0.0,
            )

        return prev_sched.completion_time + waiting_time + transport_time

    def update_predecessor_transport(
        self,
        instance: SFJSSPInstance,
        job_id: int,
        op_id: int,
        next_machine_id: int,
    ) -> float:
        """
        Store the predecessor-to-successor transport delay on the predecessor op.

        Transport time is modeled as a post-operation transfer delay that blocks
        the successor. When the successor stays on the same machine, that delay
        is zero.
        """
        if op_id <= 0:
            return 0.0

        prev_sched = self.get_operation(job_id, op_id - 1)
        if prev_sched is None:
            return 0.0

        prev_model_op = instance.get_job(job_id).operations[op_id - 1]
        transport_time = 0.0
        if prev_sched.machine_id != next_machine_id:
            transport_time = getattr(prev_model_op, "transport_time", 0.0)
        prev_sched.transport_time = transport_time
        return transport_time

    def _compute_worker_timeline_metrics(
        self,
        instance: SFJSSPInstance,
    ) -> Dict[int, Dict[str, Any]]:
        """Rebuild worker-level rest, OCRA, and fatigue metrics from the schedule."""
        metrics: Dict[int, Dict[str, Any]] = {}

        for worker in instance.workers:
            ops = sorted(
                self.worker_schedules.get(worker.worker_id, WorkerSchedule(worker.worker_id)).operations,
                key=lambda op: op.start_time,
            )
            shift_exposures: Dict[int, float] = {}
            total_work = 0.0
            total_rest = 0.0
            last_end = 0.0
            fatigue = 0.0
            max_fatigue = 0.0

            for sched_op in ops:
                rest_gap = max(0.0, sched_op.start_time - last_end)
                total_rest += rest_gap
                fatigue = float(
                    np.clip(
                        fatigue - worker.recovery_rate * rest_gap,
                        0.0,
                        worker.fatigue_max,
                    )
                )

                total_work += sched_op.processing_time
                fatigue = float(
                    np.clip(
                        fatigue + worker.fatigue_rate * sched_op.processing_time,
                        0.0,
                        worker.fatigue_max,
                    )
                )
                max_fatigue = max(max_fatigue, fatigue)

                shift_idx = int(sched_op.start_time // worker.SHIFT_DURATION)
                shift_exposures.setdefault(shift_idx, 0.0)
                shift_exposures[shift_idx] += (
                    instance.get_ergonomic_risk(sched_op.job_id, sched_op.op_id)
                    * sched_op.processing_time
                )
                last_end = sched_op.completion_time

            elapsed = last_end
            rest_fraction = (total_rest / elapsed) if elapsed > 0.0 else 0.0
            max_shift_exposure = max(shift_exposures.values()) if shift_exposures else 0.0

            metrics[worker.worker_id] = {
                "worker": worker,
                "total_work": total_work,
                "total_rest": total_rest,
                "elapsed": elapsed,
                "rest_fraction": rest_fraction,
                "shift_exposures": shift_exposures,
                "max_shift_exposure": max_shift_exposure,
                "fatigue_end": fatigue,
                "max_fatigue": max_fatigue,
            }

        return metrics

    def compute_total_energy(
        self,
        instance: SFJSSPInstance
    ) -> Dict[str, float]:
        """
        Compute total energy consumption from the canonical schedule timeline.

        Setup is modeled as occupying the tail of the pre-operation gap, so idle
        energy is charged only on gap time not already consumed by setup.
        """
        energy = {
            'processing': 0.0,
            'idle': 0.0,
            'setup': 0.0,
            'startup': 0.0,
            'auxiliary': 0.0,
            'transport': 0.0,
            'total': 0.0
        }

        for sched_op in self.scheduled_ops.values():
            machine = instance.get_machine(sched_op.machine_id)
            if machine is None:
                continue

            energy['processing'] += machine.get_processing_energy(
                sched_op.processing_time,
                sched_op.mode_id,
            )

            if sched_op.setup_time > 0:
                energy['setup'] += machine.get_setup_energy(sched_op.setup_time)

            if sched_op.transport_time > 0:
                energy['transport'] += (
                    machine.power_transport * self._minutes_to_hours(sched_op.transport_time)
                )

        for machine_id, machine_sched in self.machine_schedules.items():
            machine = instance.get_machine(machine_id)
            if machine is None:
                continue

            ops = sorted(machine_sched.operations, key=lambda op: op.start_time)
            if not ops:
                continue

            prev_completion = 0.0
            for sched_op in ops:
                raw_gap = max(0.0, sched_op.start_time - prev_completion)
                idle_gap = max(0.0, raw_gap - sched_op.setup_time)
                if idle_gap > 0.0:
                    energy['idle'] += machine.get_idle_energy(idle_gap)
                prev_completion = sched_op.completion_time

            local_horizon = ops[-1].completion_time
            aux_power = machine.auxiliary_power_share or instance.get_auxiliary_power_per_machine()
            energy['auxiliary'] += aux_power * self._minutes_to_hours(local_horizon)
            energy['startup'] += machine.startup_energy

        energy['total'] = sum(value for key, value in energy.items() if key != 'total')

        self.energy_breakdown = energy
        return energy

    def compute_carbon_emissions(self, instance: SFJSSPInstance) -> float:
        """
        Compute total carbon emissions

        Evidence: Carbon = energy * emission factor [CONFIRMED Low-carbon DRL 2024]
        """
        if not self.energy_breakdown:
            self.compute_total_energy(instance)

        total_energy = self.energy_breakdown.get('total', 0.0)
        carbon_factor = instance.get_carbon_factor()

        return total_energy * carbon_factor

    def compute_ergonomic_metrics(
        self,
        instance: SFJSSPInstance
    ) -> Dict[str, float]:
        """
        Compute ergonomic exposure metrics from the worker timelines.
        """
        worker_metrics = self._compute_worker_timeline_metrics(instance)
        all_shift_exposures: List[float] = []
        worker_max_exposure: Dict[int, float] = {}
        for worker_id, metrics in worker_metrics.items():
            exposures = list(metrics["shift_exposures"].values())
            worker_max_exposure[worker_id] = metrics["max_shift_exposure"]
            all_shift_exposures.extend(exposures)

        max_ocra = max(all_shift_exposures) if all_shift_exposures else 0.0
        mean_ocra = float(np.mean(all_shift_exposures)) if all_shift_exposures else 0.0

        self.ergonomic_metrics = {
            'max_exposure': max_ocra,
            'mean_exposure': mean_ocra,
            'total_exposure': sum(all_shift_exposures),
            'worker_max_exposure': worker_max_exposure,
        }

        return self.ergonomic_metrics

    def compute_labor_cost(self, instance: SFJSSPInstance) -> float:
        """
        Compute total labor cost

        Evidence: Labor cost = sum(worker_time * cost_rate) [CONFIRMED DRCFJSSP]
        """
        total_cost = 0.0

        for worker_id, worker_sched in self.worker_schedules.items():
            worker = instance.get_worker(worker_id)
            if worker is None:
                continue

            total_work_time = sum(
                op.processing_time for op in worker_sched.operations
            )
            total_cost += worker.get_labor_cost(total_work_time)

        return total_cost

    def compute_fatigue_metrics(
        self,
        instance: SFJSSPInstance
    ) -> Dict[str, float]:
        """
        Compute fatigue metrics from the reconstructed worker timelines.
        """
        worker_metrics = self._compute_worker_timeline_metrics(instance)
        fatigue_values = [metrics["fatigue_end"] for metrics in worker_metrics.values()]
        peak_values = [metrics["max_fatigue"] for metrics in worker_metrics.values()]

        self.fatigue_metrics = {
            'max_fatigue': max(peak_values) if peak_values else 0.0,
            'mean_fatigue': float(np.mean(fatigue_values)) if fatigue_values else 0.0,
            'fatigue_variance': float(np.var(fatigue_values)) if fatigue_values else 0.0,
        }

        return self.fatigue_metrics

    def compute_tardiness_metrics(
        self,
        instance: SFJSSPInstance
    ) -> Dict[str, float]:
        """Compute tardiness-related metrics"""
        tardiness_values = []
        weighted_tardiness = 0.0

        for job in instance.jobs:
            tardy = self.get_job_tardiness(job.job_id, instance)
            tardiness_values.append(tardy)
            weighted_tardiness += job.weight * tardy

        return {
            'total_tardiness': sum(tardiness_values),
            'mean_tardiness': np.mean(tardiness_values) if tardiness_values else 0.0,
            'max_tardiness': max(tardiness_values) if tardiness_values else 0.0,
            'weighted_tardiness': weighted_tardiness,
            'n_tardy_jobs': sum(1 for t in tardiness_values if t > 0)
        }

    def compute_robustness_metrics(
        self,
        instance: SFJSSPInstance
    ) -> Dict[str, float]:
        """
        Compute schedule robustness proxy metrics.

        Returns:
            Dict with robustness metrics
        """
        # Machine workload variance
        machine_workloads = []
        for m_id, m_sched in self.machine_schedules.items():
            workload = sum(op.processing_time for op in m_sched.operations)
            machine_workloads.append(workload)
            
        machine_variance = np.var(machine_workloads) if machine_workloads else 0.0
        
        # Worker workload variance
        worker_workloads = []
        for w_id, w_sched in self.worker_schedules.items():
            workload = sum(op.processing_time for op in w_sched.operations)
            worker_workloads.append(workload)
            
        worker_variance = np.var(worker_workloads) if worker_workloads else 0.0

        # Average slack time (buffer between operations of the same job)
        total_slack = 0.0
        slack_instances = 0
        for job in instance.jobs:
            for i in range(1, len(job.operations)):
                prev_op = self.get_operation(job.job_id, i - 1)
                curr_op = self.get_operation(job.job_id, i)
                if prev_op and curr_op:
                    slack = curr_op.start_time - prev_op.completion_time
                    total_slack += max(0.0, slack)
                    slack_instances += 1
                    
        avg_slack = total_slack / slack_instances if slack_instances > 0 else 0.0

        self.robustness_metrics = {
            'machine_workload_variance': machine_variance,
            'worker_workload_variance': worker_variance,
            'average_slack_time': avg_slack
        }

        return self.robustness_metrics

    def compute_resilience_metrics(
        self,
        instance: SFJSSPInstance
    ) -> Dict[str, float]:
        """Backward-compatible alias for schedule robustness metrics."""
        return self.compute_robustness_metrics(instance)

    def evaluate(
        self,
        instance: SFJSSPInstance,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Evaluate schedule on all objectives

        Args:
            instance: SFJSSP instance
            weights: Optional weights for composite score

        Returns:
            Dict with all objective values and optional composite score
        """
        # Compute all metrics
        self.compute_makespan()
        self.compute_total_energy(instance)
        carbon = self.compute_carbon_emissions(instance)
        self.compute_ergonomic_metrics(instance)
        self.compute_fatigue_metrics(instance)
        tardiness = self.compute_tardiness_metrics(instance)
        labor_cost = self.compute_labor_cost(instance)
        robustness = self.compute_robustness_metrics(instance)

        # Tool wear calculation (INDUSTRY 5.0 Resilience)
        # Based on mode-specific wear rates defined in machine.py
        total_tool_wear = 0.0
        for sched_op in self.scheduled_ops.values():
            machine = instance.get_machine(sched_op.machine_id)
            wear_rate = 0.0001 # Base rate
            if machine and machine.modes:
                mode = next((m for m in machine.modes if m.mode_id == sched_op.mode_id), None)
                if mode:
                    wear_rate *= getattr(mode, 'tool_wear_rate', 1.0)
            
            total_tool_wear += sched_op.processing_time * wear_rate

        tool_replacement_cost = total_tool_wear * 0.5

        self.objectives = {
            'makespan': self.makespan,
            'total_energy': self.energy_breakdown.get('total', 0.0),
            'processing_energy': self.energy_breakdown.get('processing', 0.0),
            'idle_energy': self.energy_breakdown.get('idle', 0.0),
            'carbon_emissions': carbon,
            'max_ergonomic_exposure': self.ergonomic_metrics.get('max_exposure', 0.0),
            'mean_ergonomic_exposure': self.ergonomic_metrics.get('mean_exposure', 0.0),
            'max_fatigue': self.fatigue_metrics.get('max_fatigue', 0.0),
            'fatigue_variance': self.fatigue_metrics.get('fatigue_variance', 0.0),
            'total_labor_cost': labor_cost,
            'tool_replacement_cost': tool_replacement_cost,
            'total_cost_including_tool_replacement': labor_cost + tool_replacement_cost,
            'total_tool_wear': total_tool_wear,
            'total_tardiness': tardiness['total_tardiness'],
            'weighted_tardiness': tardiness['weighted_tardiness'],
            'n_tardy_jobs': tardiness['n_tardy_jobs'],
            'machine_workload_variance': robustness['machine_workload_variance'],
            'worker_workload_variance': robustness['worker_workload_variance'],
            'average_slack_time': robustness['average_slack_time'],
        }

        # Compute composite score if weights provided
        if weights:
            # Normalize objectives (simplified - should use proper normalization)
            self.objectives['composite_score'] = sum(
                self.objectives.get(k, 0.0) * w
                for k, w in weights.items()
            )

        return self.objectives

    def _build_violation(
        self,
        code: str,
        message: str,
        *,
        job_id: Optional[int] = None,
        op_id: Optional[int] = None,
        machine_id: Optional[int] = None,
        worker_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> ConstraintViolation:
        """Create one canonical hard-violation record."""
        return ConstraintViolation(
            code=code,
            message=message,
            job_id=job_id,
            op_id=op_id,
            machine_id=machine_id,
            worker_id=worker_id,
            details=details or {},
        )

    def collect_constraint_violations(
        self,
        instance: SFJSSPInstance
    ) -> List[ConstraintViolation]:
        """
        Return canonical hard-feasibility violations with stable codes.
        """
        violations: List[ConstraintViolation] = []

        # Check that the schedule covers the full instance.
        for job in instance.jobs:
            for op in job.operations:
                if not self.is_operation_scheduled(job.job_id, op.op_id):
                    violations.append(
                        self._build_violation(
                            "unscheduled_operation",
                            f"Unscheduled operation: Job {job.job_id} Op {op.op_id}",
                            job_id=job.job_id,
                            op_id=op.op_id,
                        )
                    )

        # Check precedence constraints.
        for job in instance.jobs:
            first_op = self.get_operation(job.job_id, 0)
            if first_op and first_op.start_time < job.arrival_time:
                violations.append(
                    self._build_violation(
                        "arrival_violation",
                        f"Arrival violation: Job {job.job_id} starts before {job.arrival_time}",
                        job_id=job.job_id,
                        op_id=0,
                        machine_id=first_op.machine_id,
                        worker_id=first_op.worker_id,
                        details={
                            "job_arrival_time": job.arrival_time,
                            "scheduled_start_time": first_op.start_time,
                        },
                    )
                )

            for i in range(1, len(job.operations)):
                prev_model_op = job.operations[i - 1]
                prev_sched = self.get_operation(job.job_id, i - 1)
                curr_sched = self.get_operation(job.job_id, i)

                if prev_sched is None or curr_sched is None:
                    continue

                base_gap = self.get_ready_time_for_assignment(
                    instance,
                    job.job_id,
                    i,
                    next_machine_id=curr_sched.machine_id,
                )
                waiting_time = prev_model_op.waiting_time
                transport_time = prev_sched.transport_time
                same_machine = curr_sched.machine_id == prev_sched.machine_id

                if curr_sched.start_time < base_gap:
                    code = "precedence_violation"
                    gap_source = "completion"
                    if not same_machine and transport_time > 0.0:
                        code = "transport_gap"
                        gap_source = "transport"
                    elif waiting_time > 0.0:
                        gap_source = "waiting"

                    violations.append(
                        self._build_violation(
                            code,
                            f"Precedence violation: Job {job.job_id} Op {i}",
                            job_id=job.job_id,
                            op_id=i,
                            machine_id=curr_sched.machine_id,
                            worker_id=curr_sched.worker_id,
                            details={
                                "required_ready_time": base_gap,
                                "scheduled_start_time": curr_sched.start_time,
                                "predecessor_completion_time": prev_sched.completion_time,
                                "waiting_time": waiting_time,
                                "transport_time": transport_time,
                                "gap_source": gap_source,
                                "predecessor_machine_id": prev_sched.machine_id,
                            },
                        )
                    )

        # Check machine capacity (no overlap) and setup gap coverage.
        for machine_sched in self.machine_schedules.values():
            ops = sorted(machine_sched.operations, key=lambda x: x.start_time)
            if ops and ops[0].setup_time > ops[0].start_time:
                violations.append(
                    self._build_violation(
                        "setup_gap",
                        f"Machine setup violation: M{machine_sched.machine_id} "
                        f"({ops[0].job_id},{ops[0].op_id})",
                        job_id=ops[0].job_id,
                        op_id=ops[0].op_id,
                        machine_id=machine_sched.machine_id,
                        worker_id=ops[0].worker_id,
                        details={
                            "required_setup_time": ops[0].setup_time,
                            "scheduled_start_time": ops[0].start_time,
                        },
                    )
                )

            for i in range(len(ops) - 1):
                current_op = ops[i]
                next_op = ops[i + 1]
                required_start = current_op.completion_time + next_op.setup_time
                if required_start > next_op.start_time:
                    code = (
                        "machine_overlap"
                        if current_op.completion_time > next_op.start_time
                        else "setup_gap"
                    )
                    violations.append(
                        self._build_violation(
                            code,
                            f"Machine overlap: M{machine_sched.machine_id} "
                            f"({current_op.job_id},{current_op.op_id}) vs ({next_op.job_id},{next_op.op_id})",
                            job_id=next_op.job_id,
                            op_id=next_op.op_id,
                            machine_id=machine_sched.machine_id,
                            worker_id=next_op.worker_id,
                            details={
                                "current_operation": (current_op.job_id, current_op.op_id),
                                "next_operation": (next_op.job_id, next_op.op_id),
                                "current_completion_time": current_op.completion_time,
                                "required_start_time": required_start,
                                "scheduled_start_time": next_op.start_time,
                                "required_setup_time": next_op.setup_time,
                            },
                        )
                    )

        # Check worker capacity (no overlap).
        for worker_sched in self.worker_schedules.values():
            ops = sorted(worker_sched.operations, key=lambda x: x.start_time)
            for i in range(len(ops) - 1):
                if ops[i].completion_time > ops[i + 1].start_time:
                    violations.append(
                        self._build_violation(
                            "worker_overlap",
                            f"Worker overlap: W{worker_sched.worker_id}",
                            job_id=ops[i + 1].job_id,
                            op_id=ops[i + 1].op_id,
                            machine_id=ops[i + 1].machine_id,
                            worker_id=worker_sched.worker_id,
                            details={
                                "current_operation": (ops[i].job_id, ops[i].op_id),
                                "next_operation": (ops[i + 1].job_id, ops[i + 1].op_id),
                                "current_completion_time": ops[i].completion_time,
                                "next_start_time": ops[i + 1].start_time,
                            },
                        )
                    )

        # Check eligibility constraints.
        for (job_id, op_id), sched_op in self.scheduled_ops.items():
            job = instance.get_job(job_id)
            if job is None or op_id >= len(job.operations):
                continue
            op = job.operations[op_id]

            if sched_op.machine_id not in op.eligible_machines:
                violations.append(
                    self._build_violation(
                        "ineligible_machine_assignment",
                        f"Ineligible machine: Op({job_id},{op_id}) on M{sched_op.machine_id}",
                        job_id=job_id,
                        op_id=op_id,
                        machine_id=sched_op.machine_id,
                        worker_id=sched_op.worker_id,
                    )
                )

            if sched_op.worker_id not in op.eligible_workers:
                violations.append(
                    self._build_violation(
                        "ineligible_worker_assignment",
                        f"Ineligible worker: Op({job_id},{op_id}) by W{sched_op.worker_id}",
                        job_id=job_id,
                        op_id=op_id,
                        machine_id=sched_op.machine_id,
                        worker_id=sched_op.worker_id,
                    )
                )

        # Check period bounds.
        for job in instance.jobs:
            for op in job.operations:
                sched_op = self.get_operation(job.job_id, op.op_id)
                if sched_op is None:
                    continue

                if op.period_start is not None and op.period_end is not None:
                    if (
                        sched_op.start_time < op.period_start
                        or sched_op.completion_time > op.period_end
                    ):
                        violations.append(
                            self._build_violation(
                                "period_violation",
                                f"Period violation: Job {job.job_id} Op {op.op_id}",
                                job_id=job.job_id,
                                op_id=op.op_id,
                                machine_id=sched_op.machine_id,
                                worker_id=sched_op.worker_id,
                                details={
                                    "period_start": op.period_start,
                                    "period_end": op.period_end,
                                    "scheduled_start_time": sched_op.start_time,
                                    "scheduled_completion_time": sched_op.completion_time,
                                },
                            )
                        )

        # Human-factor feasibility checks reconstructed from the worker timelines.
        worker_metrics = self._compute_worker_timeline_metrics(instance)
        for worker_id, metrics in worker_metrics.items():
            worker = metrics["worker"]
            if metrics["elapsed"] > 0.0 and metrics["rest_fraction"] < worker.min_rest_fraction:
                violations.append(
                    self._build_violation(
                        "rest_violation",
                        f"Rest fraction violation: W{worker_id} "
                        f"(rest/elapsed = {metrics['rest_fraction']:.3f} < {worker.min_rest_fraction:.3f})",
                        worker_id=worker_id,
                        details={
                            "rest_fraction": metrics["rest_fraction"],
                            "required_rest_fraction": worker.min_rest_fraction,
                            "elapsed": metrics["elapsed"],
                            "total_rest": metrics["total_rest"],
                            "total_work": metrics["total_work"],
                        },
                    )
                )

            if metrics["max_shift_exposure"] > worker.ocra_max_per_shift:
                violations.append(
                    self._build_violation(
                        "ocra_violation",
                        f"Ergonomic exposure violation: W{worker_id} "
                        f"({metrics['max_shift_exposure']:.3f} > {worker.ocra_max_per_shift:.3f})",
                        worker_id=worker_id,
                        details={
                            "max_shift_exposure": metrics["max_shift_exposure"],
                            "allowed_max_exposure": worker.ocra_max_per_shift,
                        },
                    )
                )

        return violations

    def check_feasibility(self, instance: SFJSSPInstance) -> bool:
        """
        Check schedule feasibility
        """
        self.constraint_violation_details = self.collect_constraint_violations(instance)
        self.constraint_violations = [
            violation.message for violation in self.constraint_violation_details
        ]
        self.is_feasible = not self.constraint_violation_details
        return self.is_feasible

    def to_gantt_dict(self) -> dict:
        """Convert to Gantt chart data structure"""
        return {
            'machine_schedules': {
                mid: [
                    {
                        'job_id': op.job_id,
                        'op_id': op.op_id,
                        'start': op.start_time,
                        'end': op.completion_time,
                        'worker_id': op.worker_id
                    }
                    for op in ms.operations
                ]
                for mid, ms in self.machine_schedules.items()
            },
            'worker_schedules': {
                wid: [
                    {
                        'job_id': op.job_id,
                        'op_id': op.op_id,
                        'start': op.start_time,
                        'end': op.completion_time,
                        'machine_id': op.machine_id
                    }
                    for op in ws.operations
                ]
                for wid, ws in self.worker_schedules.items()
            }
        }

    def __repr__(self):
        return (
            f"Schedule(makespan={self.makespan:.2f}, "
            f"feasible={self.is_feasible}, "
            f"ops={len(self.scheduled_ops)})"
        )
