"""
Greedy Heuristic Solvers for SFJSSP

Evidence Status:
- Dispatching rules: CONFIRMED from scheduling literature
- FIFO, SPT, EDD: Standard rules [CONFIRMED]
- Composite rules: PROPOSED for SFJSSP multi-objective
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

try:
    from ..sfjssp_model.instance import SFJSSPInstance
    from ..sfjssp_model.schedule import Schedule
    from ..sfjssp_model.job import Job, Operation
    from ..sfjssp_model.machine import Machine
    from ..sfjssp_model.worker import Worker
except ImportError:  # pragma: no cover - supports repo-root imports
    from sfjssp_model.instance import SFJSSPInstance
    from sfjssp_model.schedule import Schedule
    from sfjssp_model.job import Job, Operation
    from sfjssp_model.machine import Machine
    from sfjssp_model.worker import Worker


# Type alias for dispatching rules
DispatchingRule = Callable[[SFJSSPInstance, Schedule, List[Tuple[int, int]]], int]


def fifo_rule(
    instance: SFJSSPInstance,
    schedule: Schedule,
    ready_ops: List[Tuple[int, int]]
) -> int:
    """
    First-In-First-Out: Select operation from job that arrived earliest

    Evidence: FIFO is a standard dispatching rule [CONFIRMED]

    Returns: index into ready_ops list
    """
    if not ready_ops:
        return -1

    # Find operation from job with earliest arrival time
    best_idx = 0
    best_arrival = float('inf')

    for i, (job_id, op_id) in enumerate(ready_ops):
        job = instance.get_job(job_id)
        if job and job.arrival_time < best_arrival:
            best_arrival = job.arrival_time
            best_idx = i

    return best_idx


def spt_rule(
    instance: SFJSSPInstance,
    schedule: Schedule,
    ready_ops: List[Tuple[int, int]]
) -> int:
    """
    Shortest Processing Time: Select operation with minimum processing time

    Evidence: SPT minimizes mean flow time [CONFIRMED]

    Returns: index into ready_ops list
    """
    if not ready_ops:
        return -1

    best_idx = 0
    best_time = float('inf')

    for i, (job_id, op_id) in enumerate(ready_ops):
        job = instance.get_job(job_id)
        if job is None or op_id >= len(job.operations):
            continue

        op = job.operations[op_id]

        # Get minimum processing time across eligible machines
        min_pt = float('inf')
        for machine_times in op.processing_times.values():
            min_pt = min(min_pt, min(machine_times.values()))

        if min_pt < best_time:
            best_time = min_pt
            best_idx = i

    return best_idx


def edt_rule(
    instance: SFJSSPInstance,
    schedule: Schedule,
    ready_ops: List[Tuple[int, int]]
) -> int:
    """
    Earliest Due Date: Select operation from job with earliest due date

    Evidence: EDD minimizes maximum tardiness [CONFIRMED]

    Returns: index into ready_ops list
    """
    if not ready_ops:
        return -1

    best_idx = 0
    best_due = float('inf')

    for i, (job_id, op_id) in enumerate(ready_ops):
        job = instance.get_job(job_id)
        if job and job.due_date and job.due_date < best_due:
            best_due = job.due_date
            best_idx = i

    return best_idx


def earliest_ready_rule(
    instance: SFJSSPInstance,
    schedule: Schedule,
    ready_ops: List[Tuple[int, int]]
) -> int:
    """
    Select operation that became ready earliest

    Evidence: Ready-time based selection [CONFIRMED]

    Returns: index into ready_ops list
    """
    if not ready_ops:
        return -1

    best_idx = 0
    best_ready_time = float("inf")

    for i, (job_id, op_id) in enumerate(ready_ops):
        job = instance.get_job(job_id)
        if job is None:
            continue

        if op_id == 0:
            ready_time = job.arrival_time
        else:
            prev_sched = schedule.get_operation(job_id, op_id - 1)
            prev_model_op = job.operations[op_id - 1]
            if prev_sched is None:
                ready_time = float("inf")
            else:
                ready_time = (
                    prev_sched.completion_time
                    + prev_sched.transport_time
                    + getattr(prev_model_op, "waiting_time", 0.0)
                )

        if ready_time < best_ready_time:
            best_ready_time = ready_time
            best_idx = i

    return best_idx


def min_energy_rule(
    instance: SFJSSPInstance,
    schedule: Schedule,
    ready_ops: List[Tuple[int, int]]
) -> int:
    """
    Select operation that can be processed with minimum energy

    Evidence: Energy-aware scheduling [CONFIRMED E-DFJSP 2025]

    Returns: index into ready_ops list
    """
    if not ready_ops:
        return -1

    best_idx = 0
    best_energy = float('inf')

    for i, (job_id, op_id) in enumerate(ready_ops):
        job = instance.get_job(job_id)
        if job is None or op_id >= len(job.operations):
            continue

        op = job.operations[op_id]

        # Find minimum energy across eligible machine/mode combinations
        min_energy = float('inf')
        for m_id in op.eligible_machines:
            machine = instance.get_machine(m_id)
            if machine is None:
                continue

            if m_id in op.processing_times:
                for mode_id, pt in op.processing_times[m_id].items():
                    mode = None
                    if machine.modes:
                        mode = next(
                            (m for m in machine.modes if m.mode_id == mode_id),
                            None
                        )

                    power = machine.power_processing
                    if mode:
                        power *= mode.power_multiplier

                    energy = power * pt
                    min_energy = min(min_energy, energy)

        if min_energy < best_energy:
            best_energy = min_energy
            best_idx = i

    return best_idx


def min_ergonomic_rule(
    instance: SFJSSPInstance,
    schedule: Schedule,
    ready_ops: List[Tuple[int, int]]
) -> int:
    """
    Select operation with minimum ergonomic risk

    Evidence: Ergonomic risk minimization [CONFIRMED NSGA-III 2021]

    Returns: index into ready_ops list
    """
    if not ready_ops:
        return -1

    best_idx = 0
    best_risk = float('inf')

    for i, (job_id, op_id) in enumerate(ready_ops):
        risk = instance.get_ergonomic_risk(job_id, op_id)

        if risk < best_risk:
            best_risk = risk
            best_idx = i

    return best_idx


def composite_rule(
    instance: SFJSSPInstance,
    schedule: Schedule,
    ready_ops: List[Tuple[int, int]],
    weights: Optional[Dict[str, float]] = None
) -> int:
    """
    Composite dispatching rule combining multiple criteria

    Evidence: Composite rules for multi-objective [PROPOSED for SFJSSP]

    Args:
        weights: Dict with keys 'spt', 'edd', 'energy', 'ergonomic'
    """
    if not ready_ops:
        return -1

    weights = weights or {
        'spt': 0.4,
        'edd': 0.3,
        'energy': 0.2,
        'ergonomic': 0.1,
    }

    # Normalize scores for each criterion
    scores = np.zeros(len(ready_ops))

    # SPT scores
    spt_scores = np.array([
        _get_min_processing_time(instance, job_id, op_id)
        for job_id, op_id in ready_ops
    ], dtype=float)
    if spt_scores.max() > spt_scores.min():
        spt_scores = 1.0 - (spt_scores - spt_scores.min()) / (spt_scores.max() - spt_scores.min())
    else:
        spt_scores = np.ones(len(ready_ops))

    # EDD scores
    edd_scores = np.array([
        _get_due_date_score(instance, job_id, op_id)
        for job_id, op_id in ready_ops
    ], dtype=float)
    if edd_scores.max() > edd_scores.min():
        edd_scores = 1.0 - (edd_scores - edd_scores.min()) / (edd_scores.max() - edd_scores.min())
    else:
        edd_scores = np.ones(len(ready_ops))

    # Energy scores
    energy_scores = np.array([
        _get_energy_score(instance, job_id, op_id)
        for job_id, op_id in ready_ops
    ], dtype=float)
    if energy_scores.max() > energy_scores.min():
        energy_scores = 1.0 - (energy_scores - energy_scores.min()) / (energy_scores.max() - energy_scores.min())
    else:
        energy_scores = np.ones(len(ready_ops))

    # Ergonomic scores
    ergonomic_scores = np.array([
        instance.get_ergonomic_risk(job_id, op_id)
        for job_id, op_id in ready_ops
    ], dtype=float)
    if ergonomic_scores.max() > ergonomic_scores.min():
        ergonomic_scores = 1.0 - (ergonomic_scores - ergonomic_scores.min()) / (ergonomic_scores.max() - ergonomic_scores.min())
    else:
        ergonomic_scores = np.ones(len(ready_ops))

    # Combine scores
    scores = (
        weights.get('spt', 0) * spt_scores +
        weights.get('edd', 0) * edd_scores +
        weights.get('energy', 0) * energy_scores +
        weights.get('ergonomic', 0) * ergonomic_scores
    )

    return int(np.argmax(scores))


def _get_min_processing_time(
    instance: SFJSSPInstance,
    job_id: int,
    op_id: int
) -> float:
    """Get minimum processing time for operation"""
    job = instance.get_job(job_id)
    if job is None or op_id >= len(job.operations):
        return float('inf')

    op = job.operations[op_id]
    min_pt = float('inf')

    for machine_times in op.processing_times.values():
        min_pt = min(min_pt, min(machine_times.values()))

    return min_pt if min_pt < float('inf') else 100.0


def _get_due_date_score(
    instance: SFJSSPInstance,
    job_id: int,
    op_id: int
) -> float:
    """Get due date score where lower values indicate more urgent jobs."""
    job = instance.get_job(job_id)
    if job is None:
        return 0.0

    if job.due_date is None:
        return 500.0

    return job.due_date


def _get_energy_score(
    instance: SFJSSPInstance,
    job_id: int,
    op_id: int
) -> float:
    """Get minimum energy score for operation"""
    job = instance.get_job(job_id)
    if job is None or op_id >= len(job.operations):
        return float('inf')

    op = job.operations[op_id]
    min_energy = float('inf')

    for m_id in op.eligible_machines:
        machine = instance.get_machine(m_id)
        if machine is None:
            continue

        if m_id in op.processing_times:
            for mode_id, pt in op.processing_times[m_id].items():
                mode = None
                if machine.modes:
                    mode = next(
                        (m for m in machine.modes if m.mode_id == mode_id),
                        None
                    )

                power = machine.power_processing
                if mode:
                    power *= mode.power_multiplier

                energy = power * pt
                min_energy = min(min_energy, energy)

    return min_energy if min_energy < float('inf') else 1000.0


class GreedyScheduler:
    """
    Greedy scheduler for SFJSSP
    """

    def __init__(
        self,
        job_rule: DispatchingRule = spt_rule,
        assignment_rule: str = 'min_time'
    ):
        self.job_rule = job_rule
        self.assignment_rule = assignment_rule

    def schedule(
        self,
        instance: SFJSSPInstance,
        verbose: bool = False
    ) -> Schedule:
        """
        Generate schedule using greedy construction
        """
        schedule = Schedule(instance_id=instance.instance_id)

        # Track resource availability
        machine_available = {m.machine_id: 0.0 for m in instance.machines}
        worker_available = {w.worker_id: 0.0 for w in instance.workers}

        # Get all operations
        all_ops = []
        for job in instance.jobs:
            for op_idx, op in enumerate(job.operations):
                all_ops.append((job.job_id, op_idx))

        scheduled = set()
        remaining = set(all_ops)

        while remaining:
            # Find ready operations
            ready_ops = []
            for job_id, op_id in remaining:
                if self._is_ready(job_id, op_id, schedule, instance):
                    ready_ops.append((job_id, op_id))

            if not ready_ops:
                schedule.is_feasible = False
                schedule.constraint_violations.append(
                    f"Deadlock: {len(remaining)} operations remain but none are ready"
                )
                if verbose:
                    print(f"Warning: No ready operations but {len(remaining)} remaining")
                break

            # Select operation using dispatching rule
            selected_idx = self.job_rule(instance, schedule, ready_ops)
            job_id, op_id = ready_ops[selected_idx]

            # Select machine and worker
            machine_id, worker_id, mode_id, start_time = self._select_resources(
                instance, schedule, job_id, op_id,
                machine_available, worker_available,
                verbose=verbose
            )

            if machine_id is None or worker_id is None:
                schedule.is_feasible = False
                schedule.constraint_violations.append(
                    f"No feasible assignment for Job {job_id} Op {op_id}"
                )
                if verbose:
                    print(f"Warning: No feasible resources for J{job_id}.O{op_id}")
                break

            # Timing already computed in _select_resources
            job = instance.get_job(job_id)
            op = job.operations[op_id]
            machine = instance.get_machine(machine_id)
            worker = instance.get_worker(worker_id)

            # Record genuine rest
            worker_free_at = worker_available[worker_id]
            if start_time > worker_free_at:
                genuine_rest = start_time - worker_free_at
                worker.record_rest(genuine_rest)

            # Check for mandatory 12.5% rest rule
            est_pt = op.get_processing_time(machine_id, mode_id, worker.get_efficiency())
            mandatory_rest = worker.requires_mandatory_rest(
                proposed_task_duration=est_pt, 
                current_time=start_time
            )
            
            if mandatory_rest > 0:
                start_time += mandatory_rest
                worker.record_rest(mandatory_rest)

            # Final processing time calculation
            processing_time = op.get_processing_time(
                machine_id, mode_id, worker.get_efficiency()
            )

            completion_time = start_time + processing_time

            # Enforce "A task cannot span two periods"
            clock = instance.period_clock
            max_period_jumps = 5
            jumps = 0
            while clock.crosses_boundary(start_time, completion_time) and jumps < max_period_jumps:
                start_time = clock.period_start(clock.get_period(start_time) + 1)
                processing_time = op.get_processing_time(
                    machine_id, mode_id, worker.get_efficiency()
                )
                completion_time = start_time + processing_time
                jumps += 1

            # Re-check mandatory rest after period jump
            mandatory_rest = worker.requires_mandatory_rest(
                proposed_task_duration=processing_time, 
                current_time=start_time
            )
            if mandatory_rest > 0:
                start_time += mandatory_rest
                worker.record_rest(mandatory_rest)
                completion_time = start_time + processing_time
            
            op.start_time = start_time
            op.completion_time = completion_time
            op.assign_period_bounds(instance.period_clock)

            prev_op_sched = schedule.get_operation(job_id, op_id - 1) if op_id > 0 else None
            transport_time = 0.0
            if prev_op_sched and prev_op_sched.machine_id != machine_id:
                transport_time = getattr(op, "transport_time", 0.0)

            actual_setup = 0.0
            last_machine_free = machine_available[machine_id]
            if start_time > last_machine_free:
                idle_gap = start_time - last_machine_free
                actual_setup = min(idle_gap, machine.setup_time)

            # Add to schedule
            schedule.add_operation(
                job_id=job_id,
                op_id=op_id,
                machine_id=machine_id,
                worker_id=worker_id,
                mode_id=mode_id,
                start_time=start_time,
                completion_time=completion_time,
                processing_time=processing_time,
                setup_time=actual_setup,
                transport_time=transport_time,
            )

            # Update resource availability
            machine_available[machine_id] = completion_time
            worker_available[worker_id] = completion_time

            # [FIX] Record idle and setup time for machine
            if start_time > last_machine_free:
                # If there's a gap, some of it is setup, the rest is idle
                # In this simple model, we assume setup happens first if needed
                idle_gap = start_time - last_machine_free
                setup_needed = machine.setup_time
                actual_idle = idle_gap - actual_setup
                
                machine.total_setup_time += actual_setup
                machine.total_idle_time += actual_idle

            # [FIX] Record transport time for machine (if job moved)
            if transport_time > 0:
                machine.total_transport_time += transport_time

            if machine.total_processing_time == 0.0:
                machine.startup_count += 1
            machine.total_processing_time += processing_time

            op.is_scheduled = True
            risk_rate = instance.get_ergonomic_risk(job_id, op_id)
            worker.record_work(
                processing_time,
                risk_rate=risk_rate,
                current_time=start_time,
                operation_type=op_id,
            )

            scheduled.add((job_id, op_id))
            remaining.remove((job_id, op_id))

        schedule.compute_makespan()
        schedule.check_feasibility(instance)
        return schedule

    def _is_ready(
        self,
        job_id: int,
        op_id: int,
        schedule: Schedule,
        instance: SFJSSPInstance
    ) -> bool:
        job = instance.get_job(job_id)
        if job is None:
            return False
        if op_id == 0:
            return True
        return schedule.is_operation_scheduled(job_id, op_id - 1)

    def _select_resources(
        self,
        instance: SFJSSPInstance,
        schedule: Schedule,
        job_id: int,
        op_id: int,
        machine_available: Dict[int, float],
        worker_available: Dict[int, float],
        verbose: bool = False
    ) -> Tuple[Optional[int], Optional[int], int, float]:
        """
        Select machine and worker for operation
        Returns: (machine_id, worker_id, mode_id, start_time)
        """
        job = instance.get_job(job_id)
        op = job.operations[op_id]

        best_machine = None
        best_worker = None
        best_mode = 0
        best_start = float('inf')
        best_score = float('inf')

        for m_id in op.eligible_machines:
            machine = instance.get_machine(m_id)
            if not machine: continue

            for w_id in op.eligible_workers:
                worker = instance.get_worker(w_id)
                if not worker: continue

                # Initial earliest start
                earliest_start = max(
                    job.arrival_time if op_id == 0 else 0.0,
                    machine_available.get(m_id, 0.0) + machine.setup_time,
                    worker_available.get(w_id, 0.0),
                    worker.mandatory_shift_lockout_until
                )
                
                if op_id > 0:
                    prev_op = schedule.get_operation(job_id, op_id - 1)
                    if prev_op:
                        # [FIX] Include transport and waiting from PREV op
                        prev_model_op = instance.get_job(job_id).operations[op_id - 1]
                        earliest_start = max(
                            earliest_start, 
                            prev_op.completion_time + prev_op.transport_time + getattr(prev_model_op, "waiting_time", 0.0)
                        )

                # Evaluate each mode
                if m_id in op.processing_times:
                    for mode_id, pt in op.processing_times[m_id].items():
                        est_proc = op.get_processing_time(
                            m_id,
                            mode_id,
                            worker.get_efficiency(),
                        )
                        
                        temp_start = earliest_start
                        clock = instance.period_clock
                        risk_rate = instance.get_ergonomic_risk(job_id, op_id)
                        
                        # Find earliest time that works
                        max_tries = 50
                        found = False
                        for _ in range(max_tries):
                            # 1. Machine gap check (centralized)
                            m_valid, m_next = machine.validate_gap(temp_start, machine.setup_time)
                            if not m_valid:
                                temp_start = max(temp_start, m_next)
                                continue

                            # 2. Worker check (centralized Industry 5.0 engine)
                            w_valid, w_next = worker.validate_assignment(temp_start, est_proc, risk_rate)
                            if not w_valid:
                                temp_start = max(temp_start, w_next)
                                continue

                            found = True
                            break                        
                        if not found:
                            continue

                        score = self._evaluate_assignment(
                            instance, op, m_id, w_id, mode_id, pt,
                            machine_available, worker_available
                        )

                        if temp_start < best_start or (temp_start == best_start and score < best_score):
                            best_score = score
                            best_machine = m_id
                            best_worker = w_id
                            best_mode = mode_id
                            best_start = temp_start

        return best_machine, best_worker, best_mode, best_start

    def _evaluate_assignment(
        self,
        instance: SFJSSPInstance,
        op: Operation,
        machine_id: int,
        worker_id: int,
        mode_id: int,
        processing_time: float,
        machine_available: Dict[int, float],
        worker_available: Dict[int, float]
    ) -> float:
        if self.assignment_rule == 'min_time':
            return processing_time
        elif self.assignment_rule == 'first_available':
            return max(
                machine_available.get(machine_id, 0),
                worker_available.get(worker_id, 0)
            ) + processing_time
        elif self.assignment_rule == 'min_energy':
            machine = instance.get_machine(machine_id)
            mode = next((m for m in machine.modes if m.mode_id == mode_id), None)
            power = machine.power_processing * (mode.power_multiplier if mode else 1.0)
            return power * processing_time
        else:
            return processing_time
