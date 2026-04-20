"""
Exact-solver entrypoints for SFJSSP.

Public support boundary:
- CP-SAT is parity-tested only on narrow canonical fixtures for
  `objective="makespan"`.
- CP-SAT now models canonical machine/worker blackout windows and typed
  breakdown/absence events as fixed no-overlap intervals for narrow makespan
  fixtures.
- CP-SAT `objective="energy"` is parity-tested only on a dedicated
  single-operation energy-tradeoff fixture slice.
- CP objective variants `energy`, `ergonomic`, and `composite` remain
  experimental outside explicitly parity-proven fixture scopes.
- The MIP path is quarantined and unavailable.

Requires: ortools >= 9.8.0 for CP-SAT execution.
"""

import math
import warnings
from fractions import Fraction
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Try to import ortools
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


ORTOOLS_IMPORT_ERROR = (
    "ortools is required for exact solver support: "
    "pip install ortools>=9.8.0"
)
VERIFIED_CP_OBJECTIVE = "makespan"
VERIFIED_CP_OBJECTIVE_SCOPES = {
    "makespan": "narrow_canonical_makespan_fixtures",
    "energy": "single_operation_energy_tradeoff_fixture",
}
EXPERIMENTAL_CP_OBJECTIVES = ("energy", "ergonomic", "composite")
SUPPORTED_CP_OBJECTIVES = (VERIFIED_CP_OBJECTIVE,) + EXPERIMENTAL_CP_OBJECTIVES
QUARANTINED_MIP_MESSAGE = (
    "MIP exact solving is currently quarantined. "
    "The legacy formulation is not validated against the canonical "
    "schedule feasibility semantics."
)


def _require_ortools() -> None:
    """Raise a clear error when exact solvers are used without OR-Tools."""
    if not ORTOOLS_AVAILABLE:
        raise ImportError(ORTOOLS_IMPORT_ERROR)


def _validate_cp_objective(objective: str) -> None:
    """Reject unsupported CP objective names early."""
    if objective not in SUPPORTED_CP_OBJECTIVES:
        supported = ", ".join(SUPPORTED_CP_OBJECTIVES)
        raise ValueError(
            f"Unsupported CP objective '{objective}'. Supported objectives: {supported}"
        )


def _instance_has_explicit_unavailability(instance: Any) -> bool:
    """Return whether the instance uses canonical blackout-window semantics."""
    return any(instance.get_machine_unavailability(machine.machine_id) for machine in instance.machines) or any(
        instance.get_worker_unavailability(worker.worker_id) for worker in instance.workers
    ) or bool(getattr(instance, "machine_breakdown_events", [])) or bool(
        getattr(instance, "worker_absence_events", [])
    )


def _iter_machine_blackout_windows(instance: Any):
    """Yield canonical machine blackout windows in deterministic order."""
    for machine in sorted(instance.machines, key=lambda item: item.machine_id):
        for window in instance.get_machine_unavailability(machine.machine_id):
            yield machine.machine_id, window


def _iter_worker_blackout_windows(instance: Any):
    """Yield canonical worker blackout windows in deterministic order."""
    for worker in sorted(instance.workers, key=lambda item: item.worker_id):
        for window in instance.get_worker_unavailability(worker.worker_id):
            yield worker.worker_id, window


def _compute_cp_horizon(instance: Any, n_ops: int, max_processing: float) -> int:
    """Compute a conservative integer time horizon for CP-SAT."""
    base_horizon = int(math.ceil(max(1.0, n_ops * max_processing * 3.0)))
    latest_arrival = max((float(job.arrival_time) for job in instance.jobs), default=0.0)
    latest_due_date = max((float(job.due_date) for job in instance.jobs), default=0.0)
    latest_blackout_end = max(
        (
            float(window.end_time)
            for _, window in _iter_machine_blackout_windows(instance)
        ),
        default=0.0,
    )
    latest_worker_blackout_end = max(
        (
            float(window.end_time)
            for _, window in _iter_worker_blackout_windows(instance)
        ),
        default=0.0,
    )
    return int(
        math.ceil(
            max(
                1.0,
                base_horizon,
                latest_arrival + max_processing,
                latest_due_date,
                latest_blackout_end,
                latest_worker_blackout_end,
            )
        )
    )


def _is_single_operation_energy_fixture_scope(instance: Any) -> bool:
    """Return whether the instance matches the narrow verified CP energy slice."""
    if len(getattr(instance, "jobs", [])) != 1:
        return False
    job = instance.jobs[0]
    if len(getattr(job, "operations", [])) != 1:
        return False
    operation = job.operations[0]
    if len(getattr(instance, "workers", [])) != 1 or len(operation.eligible_workers) != 1:
        return False
    if _instance_has_explicit_unavailability(instance):
        return False
    if any(getattr(machine, "setup_time", 0.0) > 0.0 for machine in instance.machines):
        return False
    if getattr(operation, "transport_time", 0.0) != 0.0:
        return False
    if getattr(operation, "waiting_time", 0.0) != 0.0:
        return False
    if len(operation.eligible_machines) < 2:
        return False
    worker_id = next(iter(operation.eligible_workers))
    worker = instance.get_worker(worker_id)
    if worker is None:
        return False
    if getattr(worker, "min_rest_fraction", 0.0) != 0.0:
        return False
    if getattr(instance, "get_ergonomic_risk")(job.job_id, operation.op_id) != 0.0:
        return False
    return True


def _is_narrow_canonical_makespan_fixture_scope(instance: Any) -> bool:
    """Return whether the instance matches the current verified CP makespan slice."""
    if any(getattr(machine, "setup_time", 0.0) > 0.0 for machine in instance.machines):
        return False
    for worker in instance.workers:
        if getattr(worker, "min_rest_fraction", 0.0) != 0.0:
            return False
    for job in instance.jobs:
        for operation in job.operations:
            if getattr(instance, "get_ergonomic_risk")(job.job_id, operation.op_id) != 0.0:
                return False
    return True


def _resolve_cp_verification_scope(instance: Any, objective: str) -> Optional[str]:
    """Return the exact verified solver scope for this run, if one exists."""
    if objective == VERIFIED_CP_OBJECTIVE and _is_narrow_canonical_makespan_fixture_scope(instance):
        return VERIFIED_CP_OBJECTIVE_SCOPES["makespan"]
    if objective == "energy" and _is_single_operation_energy_fixture_scope(instance):
        return VERIFIED_CP_OBJECTIVE_SCOPES["energy"]
    return None
    latest_worker_blackout_end = max(
        (
            float(window.end_time)
            for _, window in _iter_worker_blackout_windows(instance)
        ),
        default=0.0,
    )
    return int(
        math.ceil(
            max(
                1.0,
                base_horizon,
                latest_arrival + max_processing,
                latest_due_date,
                latest_blackout_end,
                latest_worker_blackout_end,
            )
        )
    )


class CPScheduler:
    """
    CP-SAT scheduler for SFJSSP.

    Publicly verified scope is intentionally narrow: parity coverage exists only
    for the makespan objective on canonical hand-built fixtures. Non-makespan
    objectives remain research paths and should be treated as surrogate modes.

    Args:
        time_limit: Time limit in seconds
        num_workers: Number of parallel workers for solver
        energy_weights: Weights for experimental composite objective studies
    """

    def __init__(
        self,
        time_limit: int = 60,
        num_workers: int = 4,
        energy_weights: Optional[Dict[str, float]] = None,
    ):
        _require_ortools()
        self.available = True
        self.time_limit = time_limit
        self.num_workers = num_workers
        self.energy_weights = energy_weights or {
            'makespan': 0.4,
            'energy': 0.3,
            'ergonomic': 0.2,
            'labor_cost': 0.1,
        }
        self.model = None
        self.solver = None

    def solve(
        self,
        instance: Any,
        objective: str = "makespan",
        verbose: bool = True,
    ) -> Optional[Any]:
        """
        Solve SFJSSP instance using CP-SAT

        Args:
            instance: SFJSSPInstance to solve
            objective: Objective to minimize
                - "makespan": the only parity-covered CP objective
                - "energy": verified only on the single-operation tradeoff
                  fixture slice; experimental otherwise
                - "ergonomic": experimental surrogate ergonomic objective
                - "composite": experimental weighted surrogate objective
            verbose: Print solver progress

        Returns:
            Schedule object or None if infeasible/timeout
        """
        _validate_cp_objective(objective)
        verification_scope = _resolve_cp_verification_scope(instance, objective)

        if verification_scope is None and objective != VERIFIED_CP_OBJECTIVE:
            warnings.warn(
                (
                    f"CPScheduler objective='{objective}' is experimental. "
                    "Only objective='makespan' has parity coverage in the "
                    "canonical test fixtures, except for explicitly "
                    "parity-proven objective slices."
                ),
                UserWarning,
                stacklevel=2,
            )

        # Create CP model
        self.model = cp_model.CpModel()

        # Build model
        self._build_model(instance, objective)

        # Create solver
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = self.time_limit
        self.solver.parameters.num_workers = self.num_workers
        self.solver.parameters.log_search_progress = verbose

        # Solve
        status = self.solver.Solve(self.model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Extract solution
            schedule = self._extract_solution(instance, objective)
            if not schedule.is_feasible:
                warnings.warn(
                    (
                        "CP-SAT returned a schedule that is infeasible under the "
                        "canonical Schedule.check_feasibility() oracle. Treat this "
                        "result as a research output, not a verified exact solution."
                    ),
                    UserWarning,
                    stacklevel=2,
                )

            if verbose:
                print(f"Solution found: makespan={schedule.makespan:.1f}")
                print(f"Status: {cp_model.Status.Name(status)}")

            return schedule
        else:
            if verbose:
                print(f"No solution found. Status: {cp_model.Status.Name(status)}")
            return None

    def _build_model(self, instance: Any, objective: str):
        """
        Build the current CP-SAT approximation of the SFJSSP.

        The formulation models precedence, dual-resource assignment, and worker
        rest/ergonomic proxy constraints. Machine setup semantics are not fully
        encoded in the CP intervals, so extracted schedules are checked again by
        the canonical schedule oracle after solving.
        """
        model = self.model

        # Data
        jobs = instance.jobs
        machines = instance.machines
        workers = instance.workers
        n_jobs = len(jobs)
        n_machines = len(machines)
        n_workers = len(workers)

        # Count operations
        all_operations = []
        for job in jobs:
            for op_idx in range(len(job.operations)):
                all_operations.append((job.job_id, op_idx))

        n_ops = len(all_operations)

        # Time horizon (upper bound)
        max_processing = 0
        for job in jobs:
            for op in job.operations:
                for m_id in op.eligible_machines:
                    if m_id in op.processing_times:
                        for pt in op.processing_times[m_id].values():
                            max_processing = max(max_processing, pt)

        horizon = _compute_cp_horizon(instance, n_ops, max_processing)

        # Scale factors for objective normalization
        # (CP-SAT requires integer objectives)
        ENERGY_SCALE = 1000  # Scale kWh terms to milli-kWh integers
        ERGO_SCALE = 1000    # Scale ergonomic indices
        COST_SCALE = 100     # Scale labor costs to cents
        OBJECTIVE_SCALE = 100000
        self.energy_scale = ENERGY_SCALE
        self.ergo_scale = ERGO_SCALE
        self.cost_scale = COST_SCALE

        # Decision variables
        # start[(j, o)] = start time of operation o of job j
        self.start = {}
        # end[(j, o)] = end time
        self.end = {}
        # duration[(j, o)] = processing time (may vary by assignment)
        self.duration = {}

        # Assignment variables
        # assign_machine[(j, o, m)] = 1 if operation (j,o) assigned to machine m
        self.assign_machine = {}
        # assign_worker[(j, o, w)] = 1 if operation (j,o) assigned to worker w
        self.assign_worker = {}
        # assign_mode[(j, o, m, mode)] = 1 if machine m uses mode for operation
        self.assign_mode = {}

        # Energy variables
        # energy[(j, o)] = energy consumption for operation
        self.energy = {}
        self.machine_assignment_vars = {m.machine_id: [] for m in machines}
        self.machine_processing_duration_terms = {m.machine_id: [] for m in machines}
        self.machine_processing_energy_terms = {m.machine_id: [] for m in machines}

        # Machine/worker intervals
        self.machine_intervals = {m.machine_id: [] for m in machines}
        self.worker_intervals = {w.worker_id: [] for w in workers}
        self.machine_blackout_counts = {m.machine_id: 0 for m in machines}
        self.worker_blackout_counts = {w.worker_id: 0 for w in workers}

        # Create variables for each operation
        for job_id, op_idx in all_operations:
            job = instance.get_job(job_id)
            op = job.operations[op_idx]

            # Get min/max processing times
            min_pt = float('inf')
            max_pt = 0
            for m_id in op.eligible_machines:
                if m_id in op.processing_times:
                    for pt in op.processing_times[m_id].values():
                        min_pt = min(min_pt, pt)
                        max_pt = max(max_pt, pt)

            if min_pt == float('inf'):
                min_pt = 10
                max_pt = 100

            # Start time variable
            self.start[(job_id, op_idx)] = model.NewIntVar(
                0, horizon, f'start_{job_id}_{op_idx}'
            )

            # Duration variable (will be linked to machine assignment)
            self.duration[(job_id, op_idx)] = model.NewIntVar(
                int(min_pt), int(max_pt * 2), f'duration_{job_id}_{op_idx}'
            )

            # End time variable
            self.end[(job_id, op_idx)] = model.NewIntVar(
                0, horizon, f'end_{job_id}_{op_idx}'
            )

            # Energy variable (scaled)
            max_energy_per_op = 0
            for machine in machines:
                machine_modes = machine.modes or [None]
                for mode in machine_modes:
                    mode_id = mode.mode_id if mode is not None else machine.default_mode_id
                    max_energy_per_op = max(
                        max_energy_per_op,
                        int(
                            round(
                                machine.get_processing_energy(max_pt, mode_id)
                                * ENERGY_SCALE
                            )
                        ),
                    )
            self.energy[(job_id, op_idx)] = model.NewIntVar(
                0, max_energy_per_op,
                f'energy_{job_id}_{op_idx}'
            )

            # Constraint: end = start + duration
            model.Add(
                self.end[(job_id, op_idx)] ==
                self.start[(job_id, op_idx)] + self.duration[(job_id, op_idx)]
            )

            # Machine assignment with mode selection
            for m_id in op.eligible_machines:
                machine = instance.get_machine(m_id)
                if m_id not in op.processing_times:
                    continue

                mode_times = op.processing_times[m_id]

                # Get machine modes (default to single mode if none)
                if machine and machine.modes:
                    modes = machine.modes
                else:
                    # Create default mode for machines without modes
                    from sfjssp_model.machine import MachineMode
                    modes = [MachineMode(mode_id=0, mode_name="default", speed_factor=1.0, power_multiplier=1.0)]

                for mode in modes:
                    mode_id = mode.mode_id

                    # Base processing time for this mode
                    if mode_id in mode_times:
                        base_pt = int(mode_times[mode_id])
                    else:
                        # Fallback: use first available time
                        base_pt = int(list(mode_times.values())[0])

                    # Adjust for mode speed
                    pt = int(base_pt / mode.speed_factor) if hasattr(mode, 'speed_factor') else base_pt

                    # Create presence variable
                    presence = model.NewBoolVar(
                        f'assign_{job_id}_{op_idx}_m{m_id}_mode{mode_id}'
                    )
                    self.assign_machine[(job_id, op_idx, m_id, mode_id)] = presence
                    self.machine_assignment_vars[m_id].append(presence)

                    # Create optional interval for this machine-mode
                    interval = model.NewOptionalIntervalVar(
                        self.start[(job_id, op_idx)],
                        pt,
                        self.end[(job_id, op_idx)],
                        presence,
                        f'interval_{job_id}_{op_idx}_m{m_id}_mode{mode_id}'
                    )
                    self.machine_intervals[m_id].append(interval)
                    self.machine_processing_duration_terms[m_id].append(pt * presence)
                    
                    if not hasattr(self, 'all_intervals_with_demands'):
                        self.all_intervals_with_demands = []
                    power_demand = machine.power_processing if machine else 10.0
                    if hasattr(mode, 'power_multiplier'):
                        power_demand *= mode.power_multiplier
                    self.all_intervals_with_demands.append((interval, int(power_demand * 100)))

                    # Link duration to assignment
                    model.Add(self.duration[(job_id, op_idx)] == pt).OnlyEnforceIf(presence)

                    # Energy calculation for this option
                    energy_val = int(
                        round(
                            (machine.get_processing_energy(pt, mode_id) if machine else 0.0)
                            * ENERGY_SCALE
                        )
                    )
                    self.machine_processing_energy_terms[m_id].append(energy_val * presence)
                    model.Add(self.energy[(job_id, op_idx)] == energy_val).OnlyEnforceIf(presence)

            # Worker assignment
            for w_id in op.eligible_workers:
                worker = instance.get_worker(w_id)
                presence = model.NewBoolVar(f'assign_w_{job_id}_{op_idx}_w{w_id}')
                self.assign_worker[(job_id, op_idx, w_id)] = presence

                rest_duration = model.NewIntVar(0, horizon, f'rest_{job_id}_{op_idx}_w{w_id}')
                min_rest_fraction = getattr(worker, "min_rest_fraction", 0.0) if worker is not None else 0.0
                rest_ratio = Fraction(str(min_rest_fraction)).limit_denominator(1000)
                if rest_ratio.numerator == 0:
                    model.Add(rest_duration == 0).OnlyEnforceIf(presence)
                else:
                    model.Add(
                        rest_ratio.denominator * rest_duration
                        >= rest_ratio.numerator * self.duration[(job_id, op_idx)]
                    ).OnlyEnforceIf(presence)
                
                total_duration_with_rest = model.NewIntVar(0, horizon, f'total_dur_{job_id}_{op_idx}_w{w_id}')
                model.Add(total_duration_with_rest == self.duration[(job_id, op_idx)] + rest_duration).OnlyEnforceIf(presence)

                # Create worker interval with rest
                # [FIX] start + duration must equal end for IntervalVar.
                # We create a specific worker_end that includes rest.
                worker_end = model.NewIntVar(0, horizon, f'worker_end_{job_id}_{op_idx}_w{w_id}')
                model.Add(worker_end == self.start[(job_id, op_idx)] + total_duration_with_rest).OnlyEnforceIf(presence)

                interval = model.NewOptionalIntervalVar(
                    self.start[(job_id, op_idx)],
                    total_duration_with_rest,
                    worker_end,
                    presence,
                    f'interval_w_{job_id}_{op_idx}_w{w_id}'
                )
                self.worker_intervals[w_id].append(interval)

        # Constraints

        # Per-period ergonomic exposure tracking.
        period_len = 480
        num_periods = (horizon // period_len) + 1
        
        for w_id in range(n_workers):
            for p in range(num_periods):
                p_start = p * period_len
                p_end = (p + 1) * period_len
                overlaps = []
                for job_id, op_idx in all_operations:
                    if (job_id, op_idx, w_id) in self.assign_worker:
                        is_assigned = self.assign_worker[(job_id, op_idx, w_id)]
                        start_in_p = model.NewBoolVar(f'j{job_id}_o{op_idx}_w{w_id}_p{p}')
                        model.AddLinearConstraint(
                            self.start[(job_id, op_idx)],
                            p_start,
                            p_end - 1,
                        ).OnlyEnforceIf([is_assigned, start_in_p])
                        model.Add(start_in_p == 0).OnlyEnforceIf(is_assigned.Not())
                        overlaps.append(start_in_p)

                period_exposure = []
                for job_id, op_idx in all_operations:
                    if (job_id, op_idx, w_id) in self.assign_worker:
                        is_assigned = self.assign_worker[(job_id, op_idx, w_id)]
                        risk_rate = int(instance.get_ergonomic_risk(job_id, op_idx) * ERGO_SCALE)

                        start_in_p = model.NewBoolVar(f'ergo_j{job_id}_o{op_idx}_w{w_id}_p{p}')
                        model.AddLinearConstraint(
                            self.start[(job_id, op_idx)],
                            p_start,
                            p_end - 1,
                        ).OnlyEnforceIf([is_assigned, start_in_p])
                        model.Add(start_in_p == 0).OnlyEnforceIf(is_assigned.Not())
                        model.Add(self.end[(job_id, op_idx)] <= p_end).OnlyEnforceIf(start_in_p)

                        exposure = model.NewIntVar(0, 5000, f'exp_j{job_id}_o{op_idx}_w{w_id}_p{p}')
                        model.Add(exposure == self.duration[(job_id, op_idx)] * risk_rate).OnlyEnforceIf(start_in_p)
                        model.Add(exposure == 0).OnlyEnforceIf(start_in_p.Not())
                        period_exposure.append(exposure)

                if period_exposure:
                    worker = instance.get_worker(w_id)
                    ocra_limit = int(round((worker.ocra_max_per_shift if worker else 2.2) * ERGO_SCALE))
                    model.Add(sum(period_exposure) <= ocra_limit)

        # 1. Each operation assigned to exactly one machine-mode combination
        for job_id, op_idx in all_operations:
            job = instance.get_job(job_id)
            op = job.operations[op_idx]

            machine_mode_vars = []
            for m_id in op.eligible_machines:
                if m_id not in op.processing_times:
                    continue
                machine = instance.get_machine(m_id)
                if machine and machine.modes:
                    modes = machine.modes
                else:
                    from sfjssp_model.machine import MachineMode
                    modes = [MachineMode(mode_id=0, mode_name="default", speed_factor=1.0, power_multiplier=1.0)]
                for mode in modes:
                    key = (job_id, op_idx, m_id, mode.mode_id)
                    if key in self.assign_machine:
                        machine_mode_vars.append(self.assign_machine[key])

            if machine_mode_vars:
                model.Add(sum(machine_mode_vars) == 1)

        # 2. Each operation assigned to exactly one worker
        for job_id, op_idx in all_operations:
            job = instance.get_job(job_id)
            op = job.operations[op_idx]

            worker_vars = [
                self.assign_worker[(job_id, op_idx, w_id)]
                for w_id in op.eligible_workers
                if (job_id, op_idx, w_id) in self.assign_worker
            ]
            if worker_vars:
                model.Add(sum(worker_vars) == 1)

        # 3. Precedence constraints (within job)
        for job in jobs:
            if job.operations:
                model.Add(
                    self.start[(job.job_id, 0)] >= int(math.ceil(job.arrival_time))
                )
            for op_idx in range(1, len(job.operations)):
                prev_op_idx = op_idx - 1
                prev_model_op = job.operations[prev_op_idx]
                model.Add(
                    self.start[(job.job_id, op_idx)] >=
                    self.end[(job.job_id, prev_op_idx)]
                    + int(math.ceil(getattr(prev_model_op, "waiting_time", 0.0)))
                    + int(math.ceil(getattr(prev_model_op, "transport_time", 0.0)))
                )

        # 4. No-overlap on machines
        self._add_machine_blackout_intervals(instance)
        for m_id, intervals in self.machine_intervals.items():
            if intervals:
                model.AddNoOverlap(intervals)

        # 5. No-overlap on workers
        self._add_worker_blackout_intervals(instance)
        for w_id, intervals in self.worker_intervals.items():
            if intervals:
                model.AddNoOverlap(intervals)

        # Objective functions

        # Makespan
        self.makespan = model.NewIntVar(0, horizon, 'makespan')
        for job in jobs:
            last_op_idx = len(job.operations) - 1
            model.Add(self.makespan >= self.end[(job.job_id, last_op_idx)])

        # Total energy aligned with schedule.compute_total_energy:
        # processing + startup + idle + auxiliary for each used machine.
        machine_total_energy_terms = []
        for machine in machines:
            m_id = machine.machine_id
            assignment_vars = self.machine_assignment_vars[m_id]

            machine_used = model.NewBoolVar(f"machine_used_{m_id}")
            if assignment_vars:
                for presence in assignment_vars:
                    model.Add(presence <= machine_used)
                model.Add(sum(assignment_vars) >= machine_used)
            else:
                model.Add(machine_used == 0)

            processing_duration = model.NewIntVar(0, horizon, f"machine_proc_dur_{m_id}")
            if self.machine_processing_duration_terms[m_id]:
                model.Add(processing_duration == sum(self.machine_processing_duration_terms[m_id]))
            else:
                model.Add(processing_duration == 0)

            processing_energy = model.NewIntVar(0, horizon * ENERGY_SCALE, f"machine_proc_energy_{m_id}")
            if self.machine_processing_energy_terms[m_id]:
                model.Add(processing_energy == sum(self.machine_processing_energy_terms[m_id]))
            else:
                model.Add(processing_energy == 0)

            active_span = model.NewIntVar(0, horizon, f"machine_active_span_{m_id}")
            model.Add(active_span <= self.makespan)
            model.Add(active_span <= horizon * machine_used)
            model.Add(active_span >= self.makespan - horizon * (1 - machine_used))

            idle_time = model.NewIntVar(0, horizon, f"machine_idle_time_{m_id}")
            model.Add(idle_time == active_span - processing_duration)

            startup_energy = int(round(machine.startup_energy * ENERGY_SCALE))
            idle_energy_rate = int(round(machine.power_idle * ENERGY_SCALE / 60.0))
            aux_power_share = machine.auxiliary_power_share or instance.get_auxiliary_power_per_machine()
            auxiliary_energy_rate = int(round(aux_power_share * ENERGY_SCALE / 60.0))

            machine_energy = model.NewIntVar(0, horizon * ENERGY_SCALE * 10, f"machine_total_energy_{m_id}")
            model.Add(
                machine_energy
                == processing_energy
                + (startup_energy * machine_used)
                + (idle_energy_rate * idle_time)
                + (auxiliary_energy_rate * active_span)
            )
            machine_total_energy_terms.append(machine_energy)

        self.total_energy = model.NewIntVar(0, horizon * ENERGY_SCALE * max(1, n_machines) * 10, 'total_energy')
        model.Add(self.total_energy == sum(machine_total_energy_terms))

        # Ergonomic risk (simplified - sum of risk-weighted durations)
        # In full model, this would track per-worker ergonomic accumulation
        ergonomic_terms = []
        for job_id, op_idx in all_operations:
            risk_rate = int(instance.get_ergonomic_risk(job_id, op_idx) * 10)
            ergonomic_terms.append(self.duration[(job_id, op_idx)] * risk_rate)
        self.total_ergonomic = model.NewIntVar(0, horizon * 100, 'total_ergonomic')
        model.Add(self.total_ergonomic == sum(ergonomic_terms))

        # Set objective based on mode
        if objective == "makespan":
            model.Minimize(self.makespan)

        elif objective == "energy":
            model.Minimize(self.total_energy)

        elif objective == "ergonomic":
            model.Minimize(self.total_ergonomic)

        elif objective == "composite":
            # Keep composite experimental, but scale terms so the linear objective
            # tracks the same raw-value ordering used by schedule.evaluate(...).
            w_m = int(round(self.energy_weights.get('makespan', 0.4) * OBJECTIVE_SCALE))
            w_e = int(round(self.energy_weights.get('energy', 0.3) * OBJECTIVE_SCALE / ENERGY_SCALE))
            w_er = int(round(self.energy_weights.get('ergonomic', 0.2) * OBJECTIVE_SCALE / ERGO_SCALE))
            model.Minimize(
                (w_m * self.makespan)
                + (w_e * self.total_energy)
                + (w_er * self.total_ergonomic)
            )

        else:
            # Default: makespan
            model.Minimize(self.makespan)

    def _annotate_required_machine_setup(
        self,
        schedule: Any,
        instance: Any,
    ) -> None:
        """Attach the canonical per-machine setup requirement to extracted ops."""
        for machine_id, machine_schedule in schedule.machine_schedules.items():
            machine = instance.get_machine(machine_id)
            required_setup = getattr(machine, "setup_time", 0.0) if machine is not None else 0.0
            for scheduled_op in machine_schedule.operations:
                scheduled_op.setup_time = required_setup

    def _add_machine_blackout_intervals(self, instance: Any) -> None:
        """Append fixed canonical machine blackout windows to the no-overlap model."""
        model = self.model
        for machine_id, window in _iter_machine_blackout_windows(instance):
            start_time = int(math.floor(window.start_time))
            end_time = int(math.ceil(window.end_time))
            if end_time <= start_time:
                continue
            interval = model.NewIntervalVar(
                model.NewConstant(start_time),
                end_time - start_time,
                model.NewConstant(end_time),
                (
                    f"machine_blackout_{machine_id}_"
                    f"{self.machine_blackout_counts[machine_id]}"
                ),
            )
            self.machine_blackout_counts[machine_id] += 1
            self.machine_intervals[machine_id].append(interval)

    def _add_worker_blackout_intervals(self, instance: Any) -> None:
        """Append fixed canonical worker blackout windows to the no-overlap model."""
        model = self.model
        for worker_id, window in _iter_worker_blackout_windows(instance):
            start_time = int(math.floor(window.start_time))
            end_time = int(math.ceil(window.end_time))
            if end_time <= start_time:
                continue
            interval = model.NewIntervalVar(
                model.NewConstant(start_time),
                end_time - start_time,
                model.NewConstant(end_time),
                (
                    f"worker_blackout_{worker_id}_"
                    f"{self.worker_blackout_counts[worker_id]}"
                ),
            )
            self.worker_blackout_counts[worker_id] += 1
            self.worker_intervals[worker_id].append(interval)

    def _extract_solution(self, instance: Any, objective: str) -> Any:
        """Extract a schedule and annotate its solver provenance."""
        try:
            from ..sfjssp_model.schedule import Schedule
        except ImportError:  # pragma: no cover - supports repo-root imports
            from sfjssp_model.schedule import Schedule

        schedule = Schedule(instance_id=instance.instance_id)

        total_energy = self.solver.Value(self.total_energy) / self.energy_scale
        total_ergonomic = 0

        for (job_id, op_idx), start_var in self.start.items():
            start_time = self.solver.Value(start_var)
            end_time = self.solver.Value(self.end[(job_id, op_idx)])
            duration = self.solver.Value(self.duration[(job_id, op_idx)])

            # Find assigned machine and mode
            machine_id = 0
            mode_id = 0
            job = instance.get_job(job_id)
            op = job.operations[op_idx]

            for m_id in op.eligible_machines:
                if m_id not in op.processing_times:
                    continue
                machine = instance.get_machine(m_id)
                if machine and machine.modes:
                    modes = machine.modes
                else:
                    from sfjssp_model.machine import MachineMode
                    modes = [MachineMode(mode_id=0, mode_name="default", speed_factor=1.0, power_multiplier=1.0)]
                for mode in modes:
                    key = (job_id, op_idx, m_id, mode.mode_id)
                    if key in self.assign_machine:
                        if self.solver.Value(self.assign_machine[key]) == 1:
                            machine_id = m_id
                            mode_id = mode.mode_id
                            break

            # Find assigned worker
            worker_id = 0
            for w_id in op.eligible_workers:
                key = (job_id, op_idx, w_id)
                if key in self.assign_worker:
                    if self.solver.Value(self.assign_worker[key]) == 1:
                        worker_id = w_id
                        break

            # Ergonomic risk contribution
            risk_rate = instance.get_ergonomic_risk(job_id, op_idx)
            total_ergonomic += duration * risk_rate

            schedule.update_predecessor_transport(
                instance,
                job_id,
                op_idx,
                machine_id,
            )

            schedule.add_operation(
                job_id=job_id,
                op_id=op_idx,
                machine_id=machine_id,
                worker_id=worker_id,
                mode_id=mode_id,
                start_time=start_time,
                completion_time=end_time,
                processing_time=duration,
                setup_time=0.0,
                transport_time=0.0,
            )

        self._annotate_required_machine_setup(schedule, instance)
        schedule.compute_makespan()
        schedule.check_feasibility(instance)
        has_unmodeled_machine_setup = any(
            getattr(machine, "setup_time", 0.0) > 0.0 for machine in instance.machines
        )
        has_explicit_blackouts = _instance_has_explicit_unavailability(instance)
        verification_scope = _resolve_cp_verification_scope(instance, objective)
        objective_verified = objective == VERIFIED_CP_OBJECTIVE or verification_scope is not None
        run_verified = objective_verified and not has_unmodeled_machine_setup
        schedule.metadata.update(
            {
                "solver": "cp_sat",
                "solver_objective": objective,
                "verified_scope": (
                    verification_scope or "experimental_surrogate_objective"
                ),
                "objective_verification_status": (
                    "verified_fixture_scope"
                    if run_verified
                    else "experimental_surrogate_objective"
                ),
                "surrogate_objective": not objective_verified,
                "surrogate_run": not run_verified,
                "surrogate_timing": has_unmodeled_machine_setup,
                "surrogate_timing_reason": (
                    "machine_setup_not_encoded_in_cp_intervals"
                    if has_unmodeled_machine_setup
                    else ""
                ),
                "calendar_event_semantics": (
                    "fixed_blackout_intervals"
                    if has_explicit_blackouts
                    else "none"
                ),
                "calendar_event_semantics_verified": (
                    run_verified
                    and has_explicit_blackouts
                ),
            }
        )

        # Store additional objectives
        schedule.objectives['total_energy_cp'] = total_energy
        schedule.objectives['total_ergonomic_cp'] = total_ergonomic

        return schedule


class MIPScheduler:
    """
    Quarantined placeholder for the legacy MIP formulation.

    This class remains importable so existing callers fail explicitly instead of
    accidentally relying on the archived formulation.
    """

    def __init__(
        self,
        time_limit: int = 300,
        solver_name: str = "SCIP",
        energy_weights: Optional[Dict[str, float]] = None,
    ):
        self.available = False
        self.time_limit = time_limit
        self.solver_name = solver_name
        self.energy_weights = energy_weights or {
            'makespan': 0.4,
            'energy': 0.3,
            'ergonomic': 0.2,
            'labor_cost': 0.1,
        }

    def solve(
        self,
        instance: Any,
        objective: str = "makespan",
        verbose: bool = True,
    ) -> Optional[Any]:
        """MIP remains unavailable until it is rederived from canonical semantics."""
        raise NotImplementedError(QUARANTINED_MIP_MESSAGE)


class EnergyAwareCPScheduler(CPScheduler):
    """
    Experimental CP-SAT helper for energy-oriented studies.

    This subclass does not widen the verified CP boundary. It only exposes
    additional surrogate objective helpers on top of the same formulation.
    """

    def __init__(
        self,
        time_limit: int = 120,
        num_workers: int = 4,
        peak_power_penalty: float = 100.0,
        tou_prices: Optional[Dict[int, float]] = None,
    ):
        super().__init__(time_limit, num_workers)
        self.peak_power_penalty = peak_power_penalty
        self.tou_prices = tou_prices or {}  # Time-of-use prices

    def solve_energy_minimizing(
        self,
        instance: Any,
        verbose: bool = True,
    ) -> Optional[Any]:
        """
        Solve with explicit energy minimization.

        Uses machine modes to trade off speed vs. energy. This helper inherits
        the same run-level verification status as `solve(..., objective="energy")`:
        parity-verified only on the dedicated single-operation tradeoff fixture
        slice and experimental otherwise.
        """
        return self.solve(instance, objective="energy", verbose=verbose)

    def solve_peak_shaving(
        self,
        instance: Any,
        max_peak_power: float,
        verbose: bool = True,
    ) -> Optional[Any]:
        """
        Solve with an added peak-power constraint.

        Experimental helper path. This remains outside the parity-validated
        exact-solver contract.
        """
        warnings.warn(
            (
                "EnergyAwareCPScheduler.solve_peak_shaving is experimental. "
                "Treat its output as a research schedule rather than a "
                "parity-validated exact result."
            ),
            UserWarning,
            stacklevel=2,
        )
        # Build model with peak power constraint
        self.model = cp_model.CpModel()
        self._build_model_with_peak_constraint(instance, max_peak_power)

        # Create solver
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = self.time_limit
        self.solver.parameters.num_workers = self.num_workers

        status = self.solver.Solve(self.model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._extract_solution(instance, "energy")
        return None

    def _build_model_with_peak_constraint(
        self,
        instance: Any,
        max_peak_power: float,
    ):
        """Build CP-SAT model with peak power constraint"""
        # First build standard model
        self._build_model(instance, "energy")

        if hasattr(self, 'all_intervals_with_demands') and self.all_intervals_with_demands:
            intervals = [item[0] for item in self.all_intervals_with_demands]
            demands = [item[1] for item in self.all_intervals_with_demands]
            self.model.AddCumulative(intervals, demands, int(max_peak_power * 100))


# Convenience function
def solve_sfjssp(
    instance: Any,
    method: str = "cp",
    objective: str = "makespan",
    time_limit: int = 60,
    verbose: bool = True,
) -> Optional[Any]:
    """
    Solve SFJSSP instance using specified method

    Args:
        instance: SFJSSPInstance to solve
        method: Solver method ("cp", "mip", "greedy")
        objective: Objective name. For `method="cp"`, only `"makespan"` has
            parity coverage, and only on narrow canonical fixtures.
            `"energy"`, `"ergonomic"`, and `"composite"` remain experimental
            surrogate objectives and emit a warning.
        time_limit: Time limit in seconds
        verbose: Print progress

    Returns:
        Schedule object or None
    """
    if method == "cp":
        solver = CPScheduler(time_limit=time_limit)
        return solver.solve(instance, objective=objective, verbose=verbose)

    elif method == "mip":
        raise NotImplementedError(QUARANTINED_MIP_MESSAGE)

    elif method == "greedy":
        from baseline_solver.greedy_solvers import GreedyScheduler, spt_rule
        scheduler = GreedyScheduler(job_rule=spt_rule)
        return scheduler.schedule(instance, verbose=verbose)

    else:
        raise ValueError(f"Unknown method: {method}")
