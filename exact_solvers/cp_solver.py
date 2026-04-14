"""
Constraint Programming and MIP Solvers for SFJSSP

Evidence Status:
- CP-SAT for scheduling: CONFIRMED from OR-Tools literature
- Energy-aware scheduling: CONFIRMED from E-DFJSP 2025
- Application to SFJSSP: PROPOSED

Requires: ortools >= 9.8.0
"""

import math
import warnings
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Try to import ortools
try:
    from ortools.sat.python import cp_model
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


ORTOOLS_IMPORT_ERROR = (
    "ortools is required for exact solver support: "
    "pip install ortools>=9.8.0"
)


def _require_ortools() -> None:
    """Raise a clear error when exact solvers are used without OR-Tools."""
    if not ORTOOLS_AVAILABLE:
        raise ImportError(ORTOOLS_IMPORT_ERROR)


class CPScheduler:
    """
    Constraint Programming scheduler for SFJSSP using OR-Tools CP-SAT

    Evidence: CP-SAT confirmed for scheduling problems [CONFIRMED]
    Application to SFJSSP with human factors: PROPOSED

    Verified support:
    - Minimize makespan

    Experimental objective modes:
    - "energy"
    - "ergonomic"
    - "composite"

    Those non-makespan modes remain available for research use, but they are not
    currently verified against the schedule-level objective calculations used by
    the rest of the repository.

    Args:
        time_limit: Time limit in seconds
        num_workers: Number of parallel workers for solver
        energy_weights: Dict of weights for multi-objective optimization
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
                - "makespan": verified objective; minimize maximum completion time
                - "energy": experimental objective; minimize total energy consumption
                - "ergonomic": experimental objective; minimize maximum ergonomic risk
                - "composite": experimental weighted multi-objective
            verbose: Print solver progress

        Returns:
            Schedule object or None if infeasible/timeout
        """
        try:
            from ..sfjssp_model.schedule import Schedule
        except ImportError:  # pragma: no cover - supports repo-root imports
            from sfjssp_model.schedule import Schedule

        if objective != "makespan":
            warnings.warn(
                (
                    f"CPScheduler objective='{objective}' is experimental. "
                    "Only objective='makespan' is currently verified against "
                    "the repository's schedule-level evaluation path."
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
        Build CP-SAT model with the current SFJSSP objective set.

        Evidence: Model structure based on:
        - Standard FJSSP formulation [CONFIRMED]
        - E-DFJSP 2025 energy modeling [CONFIRMED]
        - DyDFJSP 2023 fatigue dynamics [CONFIRMED]

        Verification status:
        - makespan objective: smoke-verified on stored small benchmarks
        - energy / ergonomic / composite objectives: experimental
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

        horizon = int(n_ops * max_processing * 3)  # Generous horizon

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
                presence = model.NewBoolVar(f'assign_w_{job_id}_{op_idx}_w{w_id}')
                self.assign_worker[(job_id, op_idx, w_id)] = presence

                # [INDUSTRY 5.0] 12.5% Mandatory Rest Rule
                # We enforce this by extending the interval duration by 12.5%
                rest_duration = model.NewIntVar(0, horizon, f'rest_{job_id}_{op_idx}_w{w_id}')
                # rest >= duration * 0.125  => 8 * rest >= duration
                model.Add(8 * rest_duration >= self.duration[(job_id, op_idx)]).OnlyEnforceIf(presence)
                
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

        # [INDUSTRY 5.0] No consecutive periods rule
        # period_len = 480 (8 hours)
        period_len = 480
        num_periods = (horizon // period_len) + 1
        
        for w_id in range(n_workers):
            worker_in_period = {}
            for p in range(num_periods):
                worker_in_period[p] = model.NewBoolVar(f'w{w_id}_p{p}')
                
                # Check if any operation assigned to this worker starts in this period
                p_start = p * period_len
                p_end = (p + 1) * period_len
                
                overlaps = []
                for job_id, op_idx in all_operations:
                    if (job_id, op_idx, w_id) in self.assign_worker:
                        is_assigned = self.assign_worker[(job_id, op_idx, w_id)]
                        
                        # Op overlaps with period if start < p_end AND end > p_start
                        # Simplified: start is in period
                        start_in_p = model.NewBoolVar(f'j{job_id}_o{op_idx}_w{w_id}_p{p}')
                        model.AddLinearConstraint(
                            self.start[(job_id, op_idx)],
                            p_start,
                            p_end - 1,
                        ).OnlyEnforceIf([is_assigned, start_in_p])
                        model.Add(start_in_p == 0).OnlyEnforceIf(is_assigned.Not())
                        overlaps.append(start_in_p)
                
                if overlaps:
                    # worker_in_period[p] is True IF AND ONLY IF sum(overlaps) > 0
                    model.Add(sum(overlaps) > 0).OnlyEnforceIf(worker_in_period[p])
                    model.Add(sum(overlaps) == 0).OnlyEnforceIf(worker_in_period[p].Not())
                    # [FIX] Force worker_in_period[p] to be true if any overlap exists
                    for overlap_var in overlaps:
                        model.AddImplication(overlap_var, worker_in_period[p])
                else:
                    model.Add(worker_in_period[p] == 0)

            # Constraint: No back-to-back periods
            for p in range(num_periods - 1):
                model.Add(worker_in_period[p] + worker_in_period[p+1] <= 1)

            # [INDUSTRY 5.0] OCRA limit per period (shift)
            for p in range(num_periods):
                p_start = p * period_len
                p_end = (p + 1) * period_len
                
                period_exposure = []
                for job_id, op_idx in all_operations:
                    if (job_id, op_idx, w_id) in self.assign_worker:
                        is_assigned = self.assign_worker[(job_id, op_idx, w_id)]
                        # scaled_risk = risk * ERGO_SCALE (1000)
                        risk_rate = int(instance.get_ergonomic_risk(job_id, op_idx) * ERGO_SCALE)
                        
                        start_in_p = model.NewBoolVar(f'ergo_j{job_id}_o{op_idx}_w{w_id}_p{p}')
                        model.AddLinearConstraint(
                            self.start[(job_id, op_idx)],
                            p_start,
                            p_end - 1,
                        ).OnlyEnforceIf([is_assigned, start_in_p])
                        model.Add(start_in_p == 0).OnlyEnforceIf(is_assigned.Not())
                        
                        # Boundary constraint: Tasks starting in a period must finish in the same period
                        model.Add(self.end[(job_id, op_idx)] <= p_end).OnlyEnforceIf(start_in_p)
                        
                        exposure = model.NewIntVar(0, 5000, f'exp_j{job_id}_o{op_idx}_w{w_id}_p{p}')
                        # [FIX] Ensure assignment is factored into exposure calculation
                        model.Add(exposure == self.duration[(job_id, op_idx)] * risk_rate).OnlyEnforceIf(start_in_p)
                        model.Add(exposure == 0).OnlyEnforceIf(start_in_p.Not())
                        period_exposure.append(exposure)
                
                if period_exposure:
                    # Limit: 2.2 * 1000 = 2200
                    model.Add(sum(period_exposure) <= 2200)

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
                model.Add(
                    self.start[(job.job_id, op_idx)] >=
                    self.end[(job.job_id, prev_op_idx)]
                )

        # 4. No-overlap on machines
        for m_id, intervals in self.machine_intervals.items():
            if intervals:
                model.AddNoOverlap(intervals)

        # 5. No-overlap on workers
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

    def _extract_solution(self, instance: Any, objective: str) -> Any:
        """Extract solution from solver"""
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

            schedule.add_operation(
                job_id=job_id,
                op_id=op_idx,
                machine_id=machine_id,
                worker_id=worker_id,
                mode_id=mode_id,
                start_time=start_time,
                completion_time=end_time,
                processing_time=duration,
            )

        schedule.compute_makespan()
        schedule.check_feasibility(instance)

        # Store additional objectives
        schedule.objectives['total_energy_cp'] = total_energy
        schedule.objectives['total_ergonomic_cp'] = total_ergonomic

        return schedule


class MIPScheduler:
    """
    Mixed-Integer Programming scheduler for SFJSSP

    Evidence: MIP for FJSSP confirmed from literature
    Application to full SFJSSP: PROPOSED

    Requires: OR-Tools linear solver or similar
    """

    def __init__(
        self,
        time_limit: int = 300,
        solver_name: str = "SCIP",
        energy_weights: Optional[Dict[str, float]] = None,
    ):
        _require_ortools()
        self.available = True
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
        """
        Solve using full SFJSSP MIP formulation

        Includes:
        - Machine mode selection
        - Worker shift constraints (Industry 5.0)
        - OCRA ergonomic limits
        - Energy optimization
        """
        raise NotImplementedError(
            "MIP exact solving is currently quarantined. "
            "The formulation is not yet validated against the current "
            "schedule feasibility semantics; use CPScheduler instead."
        )

        from sfjssp_model.schedule import Schedule

        # Create solver
        solver = pywraplp.Solver.CreateSolver(self.solver_name)
        if solver is None:
            print(f"Solver {self.solver_name} not available. Try 'SCIP' or 'CBC'")
            return None

        solver.SetTimeLimit(self.time_limit * 1000)  # ms

        # Data
        jobs = instance.jobs
        machines = instance.machines
        workers = instance.workers
        n_workers = len(workers)

        all_operations = []
        for job in jobs:
            for op_idx in range(len(job.operations)):
                all_operations.append((job.job_id, op_idx))

        # Time horizon
        max_pt = 0
        for job in jobs:
            for op in job.operations:
                for modes in op.processing_times.values():
                    if modes:
                        max_pt = max(max_pt, max(modes.values()))
        horizon = len(all_operations) * max_pt * 2
        
        # Industry 5.0 Period Constants
        period_len = 480  # 8 hours
        num_periods = int(horizon // period_len) + 1

        # Variables
        # x[j,o,m,mode,w] = 1 if operation (j,o) on machine m with mode and worker w
        x = {}
        # s[j,o] = start time
        s = {}
        # C[j,o] = completion time
        C = {}
        # shift[w,p] = 1 if worker w works in period p
        shift = {}

        for job_id, op_idx in all_operations:
            job = instance.get_job(job_id)
            op = job.operations[op_idx]

            for m_id in op.eligible_machines:
                machine = instance.get_machine(m_id)
                mode_times = op.processing_times.get(m_id, {})
                
                # Get machine modes
                if machine and machine.modes:
                    modes = machine.modes
                else:
                    from sfjssp_model.machine import MachineMode
                    modes = [MachineMode(mode_id=0, mode_name="default", speed_factor=1.0, power_multiplier=1.0)]

                for mode in modes:
                    if mode.mode_id not in mode_times and mode.mode_id != 0:
                        continue
                        
                    for w_id in op.eligible_workers:
                        x[(job_id, op_idx, m_id, mode.mode_id, w_id)] = solver.BoolVar(
                            f'x_{job_id}_{op_idx}_{m_id}_{mode.mode_id}_{w_id}'
                        )

            s[(job_id, op_idx)] = solver.NumVar(0, horizon, f's_{job_id}_{op_idx}')
            C[(job_id, op_idx)] = solver.NumVar(0, horizon, f'C_{job_id}_{op_idx}')

        for w_idx in range(n_workers):
            for p in range(num_periods):
                shift[(w_idx, p)] = solver.BoolVar(f'shift_w{w_idx}_p{p}')

        # Objective variables
        makespan = solver.NumVar(0, horizon, 'makespan')
        total_energy = solver.NumVar(0, horizon * 10000, 'total_energy')
        total_ergonomic = solver.NumVar(0, horizon * 1000, 'total_ergonomic')

        # Constraints

        # 1. Assignment: Each operation assigned to exactly one (machine, mode, worker)
        for job_id, op_idx in all_operations:
            job = instance.get_job(job_id)
            op = job.operations[op_idx]
            
            assignment_vars = []
            for m_id in op.eligible_machines:
                machine = instance.get_machine(m_id)
                mode_times = op.processing_times.get(m_id, {})
                
                if machine and machine.modes:
                    modes = machine.modes
                else:
                    from sfjssp_model.machine import MachineMode
                    modes = [MachineMode(mode_id=0, mode_name="default", speed_factor=1.0, power_multiplier=1.0)]

                for mode in modes:
                    if mode.mode_id not in mode_times and mode.mode_id != 0:
                        continue
                    for w_id in op.eligible_workers:
                        assignment_vars.append(x[(job_id, op_idx, m_id, mode.mode_id, w_id)])
            
            solver.Add(sum(assignment_vars) == 1)

        # 2. Precedence (within job)
        for job in jobs:
            if job.operations:
                solver.Add(s[(job.job_id, 0)] >= job.arrival_time)
            for op_idx in range(1, len(job.operations)):
                solver.Add(s[(job.job_id, op_idx)] >= C[(job.job_id, op_idx - 1)])

        # 3. Completion time with mandatory rest (12.5% for workers)
        for job_id, op_idx in all_operations:
            job = instance.get_job(job_id)
            op = job.operations[op_idx]
            
            durations = []
            for m_id in op.eligible_machines:
                machine = instance.get_machine(m_id)
                mode_times = op.processing_times.get(m_id, {})
                
                if machine and machine.modes:
                    modes = machine.modes
                else:
                    from sfjssp_model.machine import MachineMode
                    modes = [MachineMode(mode_id=0, mode_name="default", speed_factor=1.0, power_multiplier=1.0)]

                for mode in modes:
                    if mode.mode_id not in mode_times and mode.mode_id != 0:
                        continue
                    
                    base_pt = mode_times.get(mode.mode_id, list(mode_times.values())[0])
                    pt = base_pt / mode.speed_factor
                    
                    for w_id in op.eligible_workers:
                        # [INDUSTRY 5.0] include 12.5% rest in worker time
                        total_pt = pt * 1.125
                        durations.append(x[(job_id, op_idx, m_id, mode.mode_id, w_id)] * total_pt)
            
            solver.Add(C[(job_id, op_idx)] >= s[(job_id, op_idx)] + sum(durations))

        # 4. Industry 5.0: No consecutive shifts
        for w_idx in range(n_workers):
            for p in range(num_periods - 1):
                solver.Add(shift[(w_idx, p)] + shift[(w_idx, p+1)] <= 1)

        # 5. Link operations to shifts and OCRA limit
        M = horizon
        for w_idx, worker in enumerate(workers):
            w_id = worker.worker_id
            for p in range(num_periods):
                p_start = p * period_len
                p_end = (p + 1) * period_len
                
                period_ops = []
                for job_id, op_idx in all_operations:
                    job = instance.get_job(job_id)
                    op = job.operations[op_idx]
                    
                    if w_id not in op.eligible_workers:
                        continue
                        
                    # Find all machine/mode combos for this worker
                    worker_vars = []
                    for m_id in op.eligible_machines:
                        machine = instance.get_machine(m_id)
                        mode_times = op.processing_times.get(m_id, {})
                        if machine and machine.modes:
                            modes = machine.modes
                        else:
                            from sfjssp_model.machine import MachineMode
                            modes = [MachineMode(mode_id=0, mode_name="default", speed_factor=1.0, power_multiplier=1.0)]
                        for mode in modes:
                            if (job_id, op_idx, m_id, mode.mode_id, w_id) in x:
                                worker_vars.append(x[(job_id, op_idx, m_id, mode.mode_id, w_id)])
                    
                    if not worker_vars:
                        continue
                        
                    is_on_worker = solver.BoolVar(f'on_w{w_id}_j{job_id}_o{op_idx}')
                    solver.Add(is_on_worker == sum(worker_vars))
                    
                    # indicator if op starts in this period
                    start_in_p = solver.BoolVar(f'start_w{w_id}_j{job_id}_o{op_idx}_p{p}')
                    solver.Add(s[(job_id, op_idx)] >= p_start - M * (1 - start_in_p))
                    solver.Add(s[(job_id, op_idx)] <= p_end - 1 + M * (1 - start_in_p))
                    
                    # If start_in_p and is_on_worker, then shift must be active
                    op_in_shift = solver.BoolVar(f'in_shift_w{w_id}_j{job_id}_o{op_idx}_p{p}')
                    solver.Add(op_in_shift >= start_in_p + is_on_worker - 1)
                    solver.Add(shift[(w_idx, p)] >= op_in_shift)
                    
                    # OCRA contribution
                    risk = instance.get_ergonomic_risk(job_id, op_idx)
                    # We need duration. For MIP simplicity we use min duration or link it.
                    # Linking is better: sum(x * pt * risk)
                    for m_id in op.eligible_machines:
                        mode_times = op.processing_times.get(m_id, {})
                        machine = instance.get_machine(m_id)
                        if machine and machine.modes:
                            modes = machine.modes
                        else:
                            from sfjssp_model.machine import MachineMode
                            modes = [MachineMode(mode_id=0, mode_name="default", speed_factor=1.0, power_multiplier=1.0)]
                        for mode in modes:
                            if (job_id, op_idx, m_id, mode.mode_id, w_id) in x:
                                var = x[(job_id, op_idx, m_id, mode.mode_id, w_id)]
                                base_pt = mode_times.get(mode.mode_id, list(mode_times.values())[0])
                                pt = base_pt / mode.speed_factor
                                
                                # Only counts if start_in_p is also true
                                active_var = solver.BoolVar(f'act_w{w_id}_j{job_id}_o{op_idx}_m{m_id}_p{p}')
                                solver.Add(active_var >= var + start_in_p - 1)
                                period_ops.append(active_var * pt * risk)
                
                if period_ops:
                    solver.Add(sum(period_ops) <= 2200) # OCRA Limit 2.2 * 1000 equivalent

        # 6. Machine Capacity (Disjunctive)
        for m_id in [m.machine_id for m in machines]:
            m_ops = []
            for job_id, op_idx in all_operations:
                op = instance.get_job(job_id).operations[op_idx]
                if m_id in op.eligible_machines:
                    # check if assigned to this machine
                    machine_vars = []
                    for key, var in x.items():
                        if key[0] == job_id and key[1] == op_idx and key[2] == m_id:
                            machine_vars.append(var)
                    if machine_vars:
                        is_on_m = solver.BoolVar(f'on_m{m_id}_j{job_id}_o{op_idx}')
                        solver.Add(is_on_m == sum(machine_vars))
                        m_ops.append(((job_id, op_idx), is_on_m))
            
            for i, ((j1, o1), v1) in enumerate(m_ops):
                for ((j2, o2), v2) in m_ops[i+1:]:
                    y = solver.BoolVar(f'y_m{m_id}_j{j1}o{o1}_j{j2}o{o2}')
                    # If both on machine, one must precede the other
                    # s2 >= C1 - M(1-y) - M(2-v1-v2)
                    solver.Add(s[(j2, o2)] >= C[(j1, o1)] - M * (1 - y) - M * (2 - v1 - v2))
                    solver.Add(s[(j1, o1)] >= C[(j2, o2)] - M * y - M * (2 - v1 - v2))

        # 7. Worker Capacity (Disjunctive)
        for w_idx, worker in enumerate(workers):
            w_id = worker.worker_id
            w_ops = []
            for job_id, op_idx in all_operations:
                op = instance.get_job(job_id).operations[op_idx]
                if w_id in op.eligible_workers:
                    worker_vars = []
                    for key, var in x.items():
                        if key[0] == job_id and key[1] == op_idx and key[4] == w_id:
                            worker_vars.append(var)
                    if worker_vars:
                        is_on_w = solver.BoolVar(f'on_w{w_id}_j{job_id}_o{op_idx}')
                        solver.Add(is_on_w == sum(worker_vars))
                        w_ops.append(((job_id, op_idx), is_on_w))

            for i, ((j1, o1), v1) in enumerate(w_ops):
                for ((j2, o2), v2) in w_ops[i+1:]:
                    yw = solver.BoolVar(f'yw_w{w_id}_j{j1}o{o1}_j{j2}o{o2}')
                    solver.Add(s[(j2, o2)] >= C[(j1, o1)] - M * (1 - yw) - M * (2 - v1 - v2))
                    solver.Add(s[(j1, o1)] >= C[(j2, o2)] - M * yw - M * (2 - v1 - v2))

        # 8. Objectives
        for job in jobs:
            last_op = len(job.operations) - 1
            solver.Add(makespan >= C[(job.job_id, last_op)])

        energy_terms = []
        ergo_terms = []
        for (job_id, op_idx, m_id, mode_id, w_id), var in x.items():
            op = instance.get_job(job_id).operations[op_idx]
            machine = instance.get_machine(m_id)
            mode_times = op.processing_times.get(m_id, {})
            
            base_pt = mode_times.get(mode_id, list(mode_times.values())[0])
            pt = base_pt
            
            power = machine.power_processing if machine else 10.0
            if machine and machine.get_mode(mode_id):
                power *= machine.get_mode(mode_id).power_multiplier
            
            energy_terms.append(var * pt * power)
            ergo_terms.append(var * pt * instance.get_ergonomic_risk(job_id, op_idx))

        solver.Add(total_energy == sum(energy_terms))
        solver.Add(total_ergonomic == sum(ergo_terms))

        if objective == "energy":
            solver.Minimize(total_energy)
        elif objective == "ergonomic":
            solver.Minimize(total_ergonomic)
        elif objective == "composite":
            composite = (
                self.energy_weights.get('makespan', 0.4) * makespan +
                self.energy_weights.get('energy', 0.3) * (total_energy / 100) +
                self.energy_weights.get('ergonomic', 0.2) * (total_ergonomic / 10)
            )
            solver.Minimize(composite)
        else:
            solver.Minimize(makespan)

        # Solve
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            schedule = Schedule(instance_id=instance.instance_id)

            for (job_id, op_idx, m_id, mode_id, w_id), var in x.items():
                if var.solution_value() > 0.5:
                    start_time = s[(job_id, op_idx)].solution_value()
                    completion_time = C[(job_id, op_idx)].solution_value()

                    schedule.add_operation(
                        job_id=job_id,
                        op_id=op_idx,
                        machine_id=m_id,
                        worker_id=w_id,
                        mode_id=mode_id,
                        start_time=start_time,
                        completion_time=completion_time,
                        processing_time=completion_time - start_time,
                    )

            schedule.compute_makespan()
            schedule.check_feasibility(instance)
            schedule.objectives['total_energy_mip'] = total_energy.solution_value()
            schedule.objectives['total_ergonomic_mip'] = total_ergonomic.solution_value()

            if verbose:
                print(f"MIP solution: makespan={schedule.makespan:.1f}")

            return schedule
        else:
            if verbose:
                print(f"No MIP solution found. Status: {status}")
            return None


class EnergyAwareCPScheduler(CPScheduler):
    """
    Energy-aware CP-SAT scheduler with explicit energy modeling.

    Evidence: Energy-aware scheduling from E-DFJSP 2025 [CONFIRMED]
    Verification status: experimental objective path

    Extends CPScheduler with:
    - Machine mode selection for energy optimization
    - Time-of-use electricity pricing
    - Peak power constraints
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

        Experimental helper path. Uses machine modes to trade off speed vs.
        energy and inherits the same verification warning as `solve(...,
        objective="energy")`.
        """
        return self.solve(instance, objective="energy", verbose=verbose)

    def solve_peak_shaving(
        self,
        instance: Any,
        max_peak_power: float,
        verbose: bool = True,
    ) -> Optional[Any]:
        """
        Solve with peak power constraint

        Adds constraint on maximum simultaneous power consumption
        """
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
        objective: Objective name. For `method="cp"`, only `"makespan"` is
            currently verified; `"energy"`, `"ergonomic"`, and `"composite"`
            remain experimental and emit a warning.
        time_limit: Time limit in seconds
        verbose: Print progress

    Returns:
        Schedule object or None
    """
    if method == "cp":
        solver = CPScheduler(time_limit=time_limit)
        return solver.solve(instance, objective=objective, verbose=verbose)

    elif method == "mip":
        solver = MIPScheduler(time_limit=time_limit)
        return solver.solve(instance, verbose=verbose)

    elif method == "greedy":
        from baseline_solver.greedy_solvers import GreedyScheduler, spt_rule
        scheduler = GreedyScheduler(job_rule=spt_rule)
        return scheduler.schedule(instance, verbose=verbose)

    else:
        raise ValueError(f"Unknown method: {method}")
