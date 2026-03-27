"""
Greedy Heuristic Solvers for SFJSSP

Evidence Status:
- Dispatching rules: CONFIRMED from scheduling literature
- FIFO, SPT, EDD: Standard rules [CONFIRMED]
- Composite rules: PROPOSED for SFJSSP multi-objective
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

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

    return best_idx  # Return index, not tuple


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

    return best_idx  # Return index, not tuple


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

    return best_idx  # Return index, not tuple


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

    # First operation in list is typically the earliest ready
    # (assuming ready_ops is maintained in order)
    return 0  # Return index


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

            # Get processing time and power for each mode
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

    return best_idx  # Return index


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

    return best_idx  # Return index


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

    return int(np.argmax(scores))  # Return index, not tuple


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
    """Get due date score (higher = more urgent)"""
    job = instance.get_job(job_id)
    if job is None:
        return 0.0

    if job.due_date is None:
        return 500.0  # Default for jobs without due dates

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

    Uses dispatching rules to select operations and assignment rules
    for machine/worker selection.

    Evidence:
    - Greedy construction heuristics [CONFIRMED scheduling literature]
    - Dispatching rules for dynamic scheduling [CONFIRMED]
    """

    def __init__(
        self,
        job_rule: DispatchingRule = spt_rule,
        assignment_rule: str = 'min_time'
    ):
        """
        Initialize greedy scheduler

        Args:
            job_rule: Dispatching rule for job selection
            assignment_rule: Rule for machine/worker assignment
                ('min_time', 'first_available', 'min_energy')
        """
        self.job_rule = job_rule
        self.assignment_rule = assignment_rule

    def schedule(
        self,
        instance: SFJSSPInstance,
        verbose: bool = False
    ) -> Schedule:
        """
        Generate schedule using greedy construction

        Args:
            instance: SFJSSP problem instance
            verbose: Print progress

        Returns:
            Schedule object
        """
        schedule = Schedule(instance_id=instance.instance_id)
        current_time = 0.0

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
                # No ready operations - should not happen if data is consistent
                if verbose:
                    print(f"Warning: No ready operations but {len(remaining)} remaining")
                break

            # Select operation using dispatching rule
            selected_idx = self.job_rule(instance, schedule, ready_ops)
            job_id, op_id = ready_ops[selected_idx]

            # Select machine and worker
            machine_id, worker_id, mode_id = self._select_resources(
                instance, schedule, job_id, op_id,
                machine_available, worker_available
            )

            if machine_id is None or worker_id is None:
                # No available resources - skip
                remaining.remove((job_id, op_id))
                continue

            # Calculate timing
            job = instance.get_job(job_id)
            op = job.operations[op_id]
            machine = instance.get_machine(machine_id)
            worker = instance.get_worker(worker_id)

            # Earliest start time
            earliest_start = max(
                machine_available[machine_id],
                worker_available[worker_id]
            )

            # Check predecessor
            if op_id > 0:
                prev_op = schedule.get_operation(job_id, op_id - 1)
                if prev_op:
                    earliest_start = max(earliest_start, prev_op.completion_time)

            start_time = earliest_start

            # Calculate and apply rest duration BEFORE calculating processing time
            rest_duration = max(0.0, start_time - worker_available[worker_id])
            if rest_duration > 0:
                worker.record_rest(rest_duration)

            # [CHANGED] Check for mandatory 12.5% rest rule
            est_pt = op.get_processing_time(machine_id, mode_id, worker.get_efficiency())
            mandatory_rest = worker.requires_mandatory_rest(
                proposed_task_duration=est_pt, 
                current_time=start_time
            )
            
            if mandatory_rest > 0:
                start_time += mandatory_rest
                worker.record_rest(mandatory_rest)
                worker.total_rest_time += mandatory_rest

            # Get processing time AFTER rest recovery
            processing_time = op.get_processing_time(
                machine_id, mode_id, worker.get_efficiency()
            )

            completion_time = start_time + processing_time

            # Add to schedule
            schedule.add_operation(
                job_id=job_id,
                op_id=op_id,
                machine_id=machine_id,
                worker_id=worker_id,
                mode_id=mode_id,
                start_time=start_time,
                completion_time=completion_time,
                processing_time=processing_time
            )

            # Update availability
            machine_available[machine_id] = completion_time
            worker_available[worker_id] = completion_time

            # Update operation state
            op.start_time = start_time
            op.completion_time = completion_time
            op.is_scheduled = True

            # Update worker state
            risk_rate = instance.get_ergonomic_risk(job_id, op_id)
            worker.record_work(processing_time, risk_rate=risk_rate, current_time=start_time)

            scheduled.add((job_id, op_id))
            remaining.remove((job_id, op_id))

            if verbose and len(scheduled) % 10 == 0:
                print(f"Scheduled {len(scheduled)}/{len(all_ops)} operations")

        # Finalize schedule
        schedule.compute_makespan()
        schedule.check_feasibility(instance)

        if verbose:
            print(f"Complete. Makespan: {schedule.makespan:.2f}")
            print(f"Feasible: {schedule.is_feasible}")

        return schedule

    def _is_ready(
        self,
        job_id: int,
        op_id: int,
        schedule: Schedule,
        instance: SFJSSPInstance
    ) -> bool:
        """Check if operation is ready (predecessors done)"""
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
        worker_available: Dict[int, float]
    ) -> Tuple[Optional[int], Optional[int], int]:
        """
        Select machine and worker for operation

        Returns:
            (machine_id, worker_id, mode_id) or (None, None, 0) if infeasible
        """
        job = instance.get_job(job_id)
        if job is None or op_id >= len(job.operations):
            return None, None, 0

        op = job.operations[op_id]

        best_machine = None
        best_worker = None
        best_mode = 0
        best_score = float('inf')

        for m_id in op.eligible_machines:
            machine = instance.get_machine(m_id)
            if not machine or not machine.is_available(machine_available.get(m_id, 0)):
                continue

            for w_id in op.eligible_workers:
                worker = instance.get_worker(w_id)
                if not worker or not worker.is_available(worker_available.get(w_id, 0)):
                    continue

                # Evaluate each mode
                if m_id in op.processing_times:
                    for mode_id, pt in op.processing_times[m_id].items():
                        score = self._evaluate_assignment(
                            instance, op, m_id, w_id, mode_id, pt,
                            machine_available, worker_available
                        )

                        if score < best_score:
                            best_score = score
                            best_machine = m_id
                            best_worker = w_id
                            best_mode = mode_id

        return best_machine, best_worker, best_mode


"""
OPTIONAL: SFJSSP STRICT PERIOD RULE IN GreedyScheduler._select_resources
Label: TOO OVER (only enable if you really want strict "no consecutive periods")

Prerequisites (from worker.py):
- Worker has:
    worked_periods: Set[int] = field(default_factory=set)
    def _get_period_index(self, t: float) -> int: ...
    def can_work_in_period(self, start_time: float, end_time: float) -> bool: ...
- Worker.record_work(...) adds the current period index to worked_periods.

Integration point:
- This patch shows how to plug can_work_in_period(...) into
  GreedyScheduler._select_resources BEFORE evaluating a candidate
  (machine, worker, mode) assignment.

To activate:
  1) Remove the triple quotes around this block.
  2) Replace the existing _select_resources method with the version below.
  3) Make sure worker.can_work_in_period(...) is implemented as in worker.py.

--------------------------------------------------------------------
Example _select_resources with OPTIONAL period check marked clearly.

    def _select_resources(
        self,
        instance: SFJSSPInstance,
        schedule: Schedule,
        job_id: int,
        op_id: int,
        machine_available: Dict[int, float],
        worker_available: Dict[int, float]
    ) -> Tuple[Optional[int], Optional[int], int]:
        """
        Select machine and worker for operation

        Returns:
            (machine_id, worker_id, mode_id) or (None, None, 0) if infeasible
        """
        job = instance.get_job(job_id)
        if job is None or op_id >= len(job.operations):
            return None, None, 0

        op = job.operations[op_id]

        best_machine = None
        best_worker = None
        best_mode = 0
        best_score = float('inf')

        for m_id in op.eligible_machines:
            machine = instance.get_machine(m_id)
            if not machine or not machine.is_available(machine_available.get(m_id, 0.0)):
                continue

            for w_id in op.eligible_workers:
                worker = instance.get_worker(w_id)
                if not worker or not worker.is_available(worker_available.get(w_id, 0.0)):
                    continue

                # ------------------------------------------------------
                # OPTIONAL STRICT PERIOD RULE (TOO OVER):
                #   Enforce "no two consecutive periods" at assignment time.
                #
                #   - Compute the earliest possible start time for this
                #     (machine, worker) pair:
                #       earliest_start = max(machine_available[m],
                #                            worker_available[w])
                #   - For each candidate mode, compute an approximate/true
                #     processing time and call worker.can_work_in_period(...).
                #
                #   If can_work_in_period(...) returns False, skip this
                #   (machine, worker, mode) combination.
                # ------------------------------------------------------

                # earliest_start = max(
                #     machine_available.get(m_id, 0.0),
                #     worker_available.get(w_id, 0.0),
                # )
                #
                # if m_id in op.processing_times:
                #     for mode_id, pt in op.processing_times[m_id].items():
                #         # Option A (simple): use base pt for end time
                #         # est_proc = pt
                #
                #         # Option B (closer to reality): include worker efficiency
                #         # est_proc = op.get_processing_time(
                #         #     m_id, mode_id, worker.get_efficiency()
                #         # )
                #
                #         est_proc = pt  # or use Option B above
                #         est_end = earliest_start + est_proc
                #
                #         if not worker.can_work_in_period(
                #             start_time=earliest_start,
                #             end_time=est_end,
                #         ):
                #             # Skip this (machine, worker, mode) because it
                #             # would violate the strict period rule
                #             continue
                #
                #         # If you enable this block, you also need to move the
                #         # _evaluate_assignment(...) call inside here, after
                #         # the can_work_in_period(...) check.
                #
                # NOTE:
                #   The ACTIVE code below keeps the existing behavior and does
                #   NOT call can_work_in_period(...). Uncomment/edit carefully
                #   if you want this rule.

                # Evaluate each mode (current behavior)
                if m_id in op.processing_times:
                    for mode_id, pt in op.processing_times[m_id].items():
                        score = self._evaluate_assignment(
                            instance, op, m_id, w_id, mode_id, pt,
                            machine_available, worker_available
                        )

                        if score < best_score:
                            best_score = score
                            best_machine = m_id
                            best_worker = w_id
                            best_mode = mode_id

        return best_machine, best_worker, best_mode

"""
END OPTIONAL PERIOD RULE PATCH FOR GreedyScheduler._select_resources


"""
TOO OVER: OPTIONAL NITPICKING NOTES (SAFE TO IGNORE)

These are modeling choices that are *not* explicitly fixed by JMSY-9, and
changing them would be more about calibration than correctness. They are
documented here only for completeness and future experimentation.

1) Startup energy placeholder (E_M component)
   - Current code:
       energy['startup'] = len(self.machine_schedules) * 10.0
   - JMSY-9 only says startup belongs to the machine energy term (E_M); it
     does NOT specify an exact formula or coefficient.
   - This is a harmless placeholder; tuning it is optional and depends on
     real factory data and units, not on the paper itself.

2) Auxiliary energy modeling (E_C component)
   - Current code:
       energy['auxiliary'] += aux_power * self.makespan
   - This matches the *spirit* of "auxiliary energy proportional to the
     makespan", but JMSY-9 does not force any specific structure beyond
     "auxiliary energy exists".
   - Any change here is a parametric modeling decision, not a fix.

3) 12.5% rest rule granularity
   - Current Worker logic enforces:
       total_rest_time >= 12.5% of total_work_time   (over the horizon)
     using requires_mandatory_rest(...).
   - The paper states "rest time is at least 12.5% of worked time" but
     does not strictly define whether that is per-day, per-shift, or
     global horizon.
   - Our implementation already satisfies the aggregate condition; making
     it per-shift would be *stricter*, but is not required by the text.

4) Mono-objective vs multi-objective use
   - JMSY-9 defines SFJSSP with a mono-objective: minimize total energy
     (E_T + E_M + E_C), treating human constraints (OCRA, rest, etc.)
     as hard constraints.
   - This code also computes:
       - makespan
       - tardiness metrics
       - resilience metrics
       - ergonomic / fatigue indicators
     and can combine them into a composite score.
   - As long as "total_energy" stays as a primary objective and the hard
     constraints are enforced, having extra diagnostics/objectives is an
     *extension*, not a violation.

Summary:
- All four points here are intentionally labeled "TOO OVER": they are
  refinements you may explore if you have real data or want closer
  alignment with a *specific* plant, but they are not missing or broken
  with respect to the JMSY-9 SFJSSP definition.
"""

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
        """
        Evaluate assignment quality

        Returns score (lower = better)
        """
        if self.assignment_rule == 'min_time':
            return processing_time

        elif self.assignment_rule == 'first_available':
            # Earliest completion time
            return max(
                machine_available.get(machine_id, 0),
                worker_available.get(worker_id, 0)
            ) + processing_time

        elif self.assignment_rule == 'min_energy':
            machine = instance.get_machine(machine_id)
            if machine is None:
                return float('inf')

            mode = None
            if machine.modes:
                mode = next(
                    (m for m in machine.modes if m.mode_id == mode_id),
                    None
                )

            power = machine.power_processing
            if mode:
                power *= mode.power_multiplier

            return power * processing_time

        else:
            return processing_time
