"""
Schedule data structure for SFJSSP

Evidence Status:
- Schedule representation: PROPOSED synthesis
- Objective evaluation: Based on MATHEMATICAL_MODEL_SFJSSP.md [PROPOSED]
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

    # Objective values (computed)
    objectives: Dict[str, float] = field(default_factory=dict)

    # Energy consumption breakdown
    energy_breakdown: Dict[str, float] = field(default_factory=dict)

    # Human factor metrics
    ergonomic_metrics: Dict[str, float] = field(default_factory=dict)
    fatigue_metrics: Dict[str, float] = field(default_factory=dict)

    # Resilience metrics (for dynamic scenarios)
    resilience_metrics: Dict[str, float] = field(default_factory=dict)

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
            return 0.0

        self.makespan = max(
            op.completion_time for op in self.scheduled_ops.values()
        )
        return self.makespan

    def compute_total_energy(
        self,
        instance: SFJSSPInstance
    ) -> Dict[str, float]:
        """
        Compute total energy consumption

        Evidence: Energy components from E-DFJSP 2025 [CONFIRMED]

        Returns:
            Dict with keys: processing, idle, setup, startup, auxiliary, total
        """
        energy = {
            'processing': 0.0,
            'idle': 0.0,
            'setup': 0.0,
            'startup': 0.0,
            'auxiliary': 0.0,
            'transport': 0.0,  # [CHANGED] Added transport energy
            'total': 0.0
        }

        # Processing and setup energy
        for sched_op in self.scheduled_ops.values():
            machine = instance.get_machine(sched_op.machine_id)
            if machine is None:
                continue

            # Get mode
            mode = None
            if machine.modes:
                mode = next(
                    (m for m in machine.modes if m.mode_id == sched_op.mode_id),
                    None
                )

            # Processing energy
            power = machine.power_processing
            if mode:
                power *= mode.power_multiplier
            energy['processing'] += power * sched_op.processing_time

            # Setup energy
            if sched_op.setup_time > 0:
                energy['setup'] += machine.power_setup * sched_op.setup_time

            # [CHANGED] Transport energy calculation
            if sched_op.transport_time > 0:
                energy['transport'] += 5.0 * sched_op.transport_time

        # Idle energy
        for machine_id, machine_sched in self.machine_schedules.items():
            machine = instance.get_machine(machine_id)
            if machine is None:
                continue

            idle_periods = machine_sched.get_idle_periods(self.makespan)
            for start, end in idle_periods:
                energy['idle'] += machine.power_idle * (end - start)

        # Auxiliary energy
        aux_power = instance.get_auxiliary_power_per_machine()
        for machine_id in self.machine_schedules:
            machine = instance.get_machine(machine_id)
            if machine:
                energy['auxiliary'] += aux_power * self.makespan

        # Startup energy: sum of startup energy for all machines used
        for machine_id in self.machine_schedules:
            machine = instance.get_machine(machine_id)
            if machine:
                energy['startup'] += machine.startup_energy

        energy['total'] = sum(energy.values())

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
        Compute ergonomic risk metrics per shift.
        """
        # Track exposure per worker per 8-hour shift
        # Map: worker_id -> List of exposures (one per shift)
        worker_shift_exposures = {w.worker_id: [0.0] for w in instance.workers}
        SHIFT_LEN = 480.0

        for sched_op in sorted(self.scheduled_ops.values(), key=lambda x: x.start_time):
            risk_rate = instance.get_ergonomic_risk(
                sched_op.job_id, sched_op.op_id
            )
            exposure = risk_rate * sched_op.processing_time
            
            w_id = sched_op.worker_id
            # Determine which shift this operation falls into (roughly)
            # A more precise way would be to track shift resets in Worker,
            # but here we can approximate by time.
            shift_idx = int(sched_op.start_time // SHIFT_LEN)
            
            # Ensure we have enough shifts in the list
            while len(worker_shift_exposures[w_id]) <= shift_idx:
                worker_shift_exposures[w_id].append(0.0)
                
            worker_shift_exposures[w_id][shift_idx] += exposure

        # Find max OCRA encountered in any single shift across all workers
        all_shift_exposures = []
        for exposures in worker_shift_exposures.values():
            all_shift_exposures.extend(exposures)
            
        max_ocra = max(all_shift_exposures) if all_shift_exposures else 0.0
        mean_ocra = np.mean(all_shift_exposures) if all_shift_exposures else 0.0

        self.ergonomic_metrics = {
            'max_exposure': max_ocra,
            'mean_exposure': mean_ocra,
            'total_exposure': sum(all_shift_exposures)
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
            total_cost += worker.labor_cost_per_hour * total_work_time

        return total_cost

    def compute_fatigue_metrics(
        self,
        instance: SFJSSPInstance
    ) -> Dict[str, float]:
        """
        Compute worker fatigue metrics

        Evidence: Fatigue dynamics from DyDFJSP 2023 [CONFIRMED]

        Returns:
            Dict with: max_fatigue, mean_fatigue, fatigue_variance
        """
        worker_fatigue = {}

        for worker in instance.workers:
            # Get worker's total work time
            if worker.worker_id in self.worker_schedules:
                total_work = sum(
                    op.processing_time
                    for op in self.worker_schedules[worker.worker_id].operations
                )
            else:
                total_work = 0.0

            # Simplified fatigue model: F = alpha * work_time
            fatigue = min(1.0, worker.fatigue_rate * total_work)
            worker_fatigue[worker.worker_id] = fatigue

        fatigue_values = list(worker_fatigue.values())

        self.fatigue_metrics = {
            'max_fatigue': max(fatigue_values) if fatigue_values else 0.0,
            'mean_fatigue': np.mean(fatigue_values) if fatigue_values else 0.0,
            'fatigue_variance': np.var(fatigue_values) if fatigue_values else 0.0
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

    def compute_resilience_metrics(
        self,
        instance: SFJSSPInstance
    ) -> Dict[str, float]:
        """
        Compute resilience metrics

        Evidence: L1.2 resilience metrics for dynamic scenarios [PROPOSED]

        Returns:
            Dict with resilience metrics
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

        self.resilience_metrics = {
            'machine_workload_variance': machine_variance,
            'worker_workload_variance': worker_variance,
            'average_slack_time': avg_slack
        }
        
        return self.resilience_metrics

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
        resilience = self.compute_resilience_metrics(instance)

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
            'total_tardiness': tardiness['total_tardiness'],
            'weighted_tardiness': tardiness['weighted_tardiness'],
            'n_tardy_jobs': tardiness['n_tardy_jobs'],
            'machine_workload_variance': resilience['machine_workload_variance'],
            'worker_workload_variance': resilience['worker_workload_variance'],
            'average_slack_time': resilience['average_slack_time'],
        }

        # Compute composite score if weights provided
        if weights:
            # Normalize objectives (simplified - should use proper normalization)
            self.objectives['composite_score'] = sum(
                self.objectives.get(k, 0.0) * w
                for k, w in weights.items()
            )

        return self.objectives

    def check_feasibility(self, instance: SFJSSPInstance) -> bool:
        """
        Check schedule feasibility
        """
        self.is_feasible = True
        self.constraint_violations = []

        # Check precedence constraints
        for job in instance.jobs:
            for i, op in enumerate(job.operations):
                if i == 0:
                    continue

                prev_sched = self.get_operation(job.job_id, i - 1)
                curr_sched = self.get_operation(job.job_id, i)

                if prev_sched and curr_sched:
                    # Include transport_time and modeled waiting_time of previous operation
                    base_gap = prev_sched.completion_time + prev_sched.transport_time
                    prev_model_op = job.operations[i - 1]
                    waiting_time = getattr(prev_model_op, "waiting_time", 0.0)
                    base_gap += waiting_time

                    if curr_sched.start_time < base_gap:
                        self.is_feasible = False
                        self.constraint_violations.append(
                            f"Precedence violation: Job {job.job_id} Op {i}"
                        )

        # Check machine capacity (no overlap)
        for machine_sched in self.machine_schedules.values():
            ops = sorted(machine_sched.operations, key=lambda x: x.start_time)
            for i in range(len(ops) - 1):
                # [CHANGED] Added setup_time
                if (ops[i].completion_time + ops[i].setup_time) > ops[i + 1].start_time:
                    self.is_feasible = False
                    self.constraint_violations.append(
                        f"Machine overlap: M{machine_sched.machine_id} "
                        f"({ops[i].job_id},{ops[i].op_id}) vs ({ops[i+1].job_id},{ops[i+1].op_id})"
                    )

        # Check worker capacity (no overlap)
        for worker_sched in self.worker_schedules.values():
            ops = sorted(worker_sched.operations, key=lambda x: x.start_time)
            for i in range(len(ops) - 1):
                if ops[i].completion_time > ops[i + 1].start_time:
                    self.is_feasible = False
                    self.constraint_violations.append(
                        f"Worker overlap: W{worker_sched.worker_id}"
                    )

        # Check eligibility constraints
        for (job_id, op_id), sched_op in self.scheduled_ops.items():
            job = instance.get_job(job_id)
            if job is None:
                continue

            op = job.operations[op_id] if op_id < len(job.operations) else None
            if op is None:
                continue

            if sched_op.machine_id not in op.eligible_machines:
                self.is_feasible = False
                self.constraint_violations.append(
                    f"Ineligible machine: Op({job_id},{op_id}) on M{sched_op.machine_id}"
                )

            if sched_op.worker_id not in op.eligible_workers:
                self.is_feasible = False
                self.constraint_violations.append(
                    f"Ineligible worker: Op({job_id},{op_id}) by W{sched_op.worker_id}"
                )

        # Check period bounds (operation must lie within its assigned period)
        for job in instance.jobs:
            for op in job.operations:
                sched_op = self.get_operation(job.job_id, op.op_id)
                if sched_op is None:
                    continue

                if op.period_start is not None and op.period_end is not None:
                    if not op.is_within_period():
                        self.is_feasible = False
                        self.constraint_violations.append(
                            f"Period violation: Job {job.job_id} Op {op.op_id}"
                        )

        # Check due-date constraints (job completion must not exceed due date)
        for job in instance.jobs:
            if job.due_date is None:
                continue

            completion = self.get_job_completion_time(job.job_id, instance)
            if completion > job.due_date:
                self.is_feasible = False
                self.constraint_violations.append(
                    f"Due date violation: Job {job.job_id} "
                    f"(C={completion:.2f} > D={job.due_date:.2f})"
                )

        # --- Human-related feasibility checks ---

        # 1) Rest fraction >= 12.5% of work time for each worker
        MIN_REST_FRACTION = 0.125

        for worker_id, worker_sched in self.worker_schedules.items():
            ops = sorted(worker_sched.operations, key=lambda x: x.start_time)
            if not ops:
                continue

            # [FIX] Count rest from t=0 to include initial idle time
            last_end = ops[-1].completion_time
            total_work = sum(op.processing_time for op in ops)
            total_rest = last_end - total_work

            if total_work > 0.0:
                rest_fraction = total_rest / total_work
                if rest_fraction < MIN_REST_FRACTION:
                    self.is_feasible = False
                    self.constraint_violations.append(
                        f"Rest fraction violation: W{worker_id} "
                        f"(rest/work = {rest_fraction:.3f} < {MIN_REST_FRACTION:.3f})"
                    )

        # 2) OCRA / ergonomic exposure <= 2.2 per shift (approximate)

        # Reuse existing metric computation
        erg_metrics = self.compute_ergonomic_metrics(instance)
        max_exposure = erg_metrics.get("max_exposure", 0.0)

        # Threshold from instance if available, else default 2.2
        ocra_threshold = getattr(instance, "ocra_max_per_shift", 2.2)

        if max_exposure > ocra_threshold:
            self.is_feasible = False
            self.constraint_violations.append(
                f"Ergonomic exposure violation: "
                f"max_exposure = {max_exposure:.3f} > {ocra_threshold:.3f}"
            )

        # --- Period-based feasibility: no two consecutive periods per worker ---

        # Get canonical SHIFT_DURATION from instance workers (fallback 480.0)
        if instance.workers:
            shift_duration = instance.workers[0].SHIFT_DURATION
        else:
            shift_duration = 480.0  # default 8h

        for worker_id, w_sched in self.worker_schedules.items():
            ops = sorted(w_sched.operations, key=lambda x: x.start_time)
            if not ops:
                continue

            # Map operations to period indices
            periods = [int(op.start_time // shift_duration) for op in ops]

            # Check for consecutive periods
            for p_prev, p_curr in zip(periods[:-1], periods[1:]):
                if p_curr - p_prev == 1:
                    self.is_feasible = False
                    self.constraint_violations.append(
                        f"Period constraint violation: W{worker_id} "
                        f"worked in consecutive periods {p_prev} and {p_curr}"
                    )
                    break

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
