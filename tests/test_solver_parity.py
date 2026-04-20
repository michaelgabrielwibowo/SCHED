import numpy as np
import pytest

try:
    from ..baseline_solver.greedy_solvers import GreedyScheduler, fifo_rule
    from ..environment.sfjssp_env import SFJSSPEnv
    from ..exact_solvers.cp_solver import solve_sfjssp
    from ..moea.nsga3 import evaluate_sfjssp_genome_detailed, schedule_to_sfjssp_genome
    from ..sfjssp_model.instance import SFJSSPInstance
    from ..sfjssp_model.job import Job, Operation
    from ..sfjssp_model.machine import Machine, MachineMode
    from ..sfjssp_model.worker import Worker
except ImportError:  # pragma: no cover - supports repo-root imports
    from baseline_solver.greedy_solvers import GreedyScheduler, fifo_rule
    from environment.sfjssp_env import SFJSSPEnv
    from exact_solvers.cp_solver import solve_sfjssp
    from moea.nsga3 import evaluate_sfjssp_genome_detailed, schedule_to_sfjssp_genome
    from sfjssp_model.instance import SFJSSPInstance
    from sfjssp_model.job import Job, Operation
    from sfjssp_model.machine import Machine, MachineMode
    from sfjssp_model.worker import Worker


def _clone_instance(instance: SFJSSPInstance) -> SFJSSPInstance:
    return SFJSSPInstance.from_dict(instance.to_dict())


def _schedule_signature(schedule) -> dict:
    return {
        (job_id, op_id): {
            "machine_id": scheduled_op.machine_id,
            "worker_id": scheduled_op.worker_id,
            "mode_id": scheduled_op.mode_id,
            "start_time": round(scheduled_op.start_time, 6),
            "completion_time": round(scheduled_op.completion_time, 6),
            "processing_time": round(scheduled_op.processing_time, 6),
            "setup_time": round(scheduled_op.setup_time, 6),
            "transport_time": round(scheduled_op.transport_time, 6),
        }
        for (job_id, op_id), scheduled_op in schedule.scheduled_ops.items()
    }


def _metric_signature(instance: SFJSSPInstance, schedule) -> dict:
    metrics = schedule.evaluate(instance)
    return {
        "makespan": round(metrics["makespan"], 6),
        "total_energy": round(metrics["total_energy"], 6),
        "max_ergonomic_exposure": round(metrics["max_ergonomic_exposure"], 6),
        "total_labor_cost": round(metrics["total_labor_cost"], 6),
        "weighted_tardiness": round(metrics["weighted_tardiness"], 6),
        "n_tardy_jobs": int(metrics["n_tardy_jobs"]),
    }


def _run_env_on_reference_schedule(instance: SFJSSPInstance, reference_schedule):
    env = SFJSSPEnv(instance)
    env.reset(seed=7)
    for scheduled_op in reference_schedule.scheduled_ops.values():
        action = {
            "job_idx": scheduled_op.job_id,
            "op_idx": scheduled_op.op_id,
            "machine_idx": scheduled_op.machine_id,
            "worker_idx": scheduled_op.worker_id,
            "mode_idx": scheduled_op.mode_id,
        }
        for _ in range(20):
            _, _, _, truncated, info = env.step(action)
            assert truncated is False
            if info.get("invalid") is not True:
                break
            assert info["reason"] == "Operation precedence not satisfied"
            env._advance_time()
        else:
            raise AssertionError(f"Environment never accepted action {action}")
    assert env.schedule.check_feasibility(instance) is True
    return env.schedule


def _build_transport_waiting_instance() -> SFJSSPInstance:
    instance = SFJSSPInstance(instance_id="PARITY_TRANSPORT_WAITING")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)], setup_time=0.0))
    instance.add_machine(Machine(machine_id=1, modes=[MachineMode(mode_id=0)], setup_time=0.0))
    instance.add_worker(Worker(worker_id=0, min_rest_fraction=0.0, ocra_max_per_shift=999.0))

    first = Operation(
        job_id=0,
        op_id=0,
        processing_times={0: {0: 10.0}},
        eligible_machines={0},
        eligible_workers={0},
        transport_time=4.0,
        waiting_time=2.0,
    )
    second = Operation(
        job_id=0,
        op_id=1,
        processing_times={1: {0: 7.0}},
        eligible_machines={1},
        eligible_workers={0},
    )
    instance.add_job(
        Job(
            job_id=0,
            arrival_time=5.0,
            due_date=100.0,
            operations=[first, second],
        )
    )
    instance.ergonomic_risk_map[(0, 0)] = 0.0
    instance.ergonomic_risk_map[(0, 1)] = 0.0
    return instance


def _build_setup_rest_ocra_instance() -> SFJSSPInstance:
    instance = SFJSSPInstance(instance_id="PARITY_SETUP_REST_OCRA")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)], setup_time=3.0))
    instance.add_worker(
        Worker(
            worker_id=0,
            min_rest_fraction=0.5,
            ocra_max_per_shift=2.2,
            base_efficiency=1.0,
        )
    )

    for job_id, arrival_time in enumerate((0.0, 1.0, 2.0)):
        instance.add_job(
            Job(
                job_id=job_id,
                arrival_time=arrival_time,
                due_date=500.0,
                operations=[
                    Operation(
                        job_id=job_id,
                        op_id=0,
                        processing_times={0: {0: 6.0}},
                        eligible_machines={0},
                        eligible_workers={0},
                    )
                ],
            )
        )
        instance.ergonomic_risk_map[(job_id, 0)] = 0.1
    return instance


def _build_single_operation_instance() -> SFJSSPInstance:
    instance = SFJSSPInstance(instance_id="PARITY_CP_SINGLE_OP")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)], setup_time=0.0))
    instance.add_worker(
        Worker(
            worker_id=0,
            min_rest_fraction=0.0,
            ocra_max_per_shift=999.0,
            fatigue_rate=0.0,
            recovery_rate=0.0,
            learning_coefficient=0.0,
        )
    )
    instance.add_job(
        Job(
            job_id=0,
            arrival_time=0.0,
            due_date=100.0,
            operations=[
                Operation(
                    job_id=0,
                    op_id=0,
                    processing_times={0: {0: 10.0}},
                    eligible_machines={0},
                    eligible_workers={0},
                )
            ],
        )
    )
    instance.ergonomic_risk_map[(0, 0)] = 0.0
    return instance


def _build_simple_precedence_instance() -> SFJSSPInstance:
    instance = SFJSSPInstance(instance_id="PARITY_CP_SIMPLE_PRECEDENCE")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)], setup_time=0.0))
    instance.add_worker(
        Worker(
            worker_id=0,
            min_rest_fraction=0.0,
            ocra_max_per_shift=999.0,
            fatigue_rate=0.0,
            recovery_rate=0.0,
            learning_coefficient=0.0,
        )
    )
    instance.add_job(
        Job(
            job_id=0,
            arrival_time=0.0,
            due_date=100.0,
            operations=[
                Operation(
                    job_id=0,
                    op_id=0,
                    processing_times={0: {0: 4.0}},
                    eligible_machines={0},
                    eligible_workers={0},
                ),
                Operation(
                    job_id=0,
                    op_id=1,
                    processing_times={0: {0: 6.0}},
                    eligible_machines={0},
                    eligible_workers={0},
                ),
            ],
        )
    )
    instance.ergonomic_risk_map[(0, 0)] = 0.0
    instance.ergonomic_risk_map[(0, 1)] = 0.0
    return instance


def _build_machine_calendar_instance() -> SFJSSPInstance:
    instance = _build_single_operation_instance()
    instance.instance_id = "PARITY_MACHINE_CALENDAR"
    instance.add_machine_unavailability(
        0,
        0.0,
        20.0,
        reason="maintenance",
        source="calendar",
    )
    return instance


def _build_worker_calendar_instance() -> SFJSSPInstance:
    instance = _build_single_operation_instance()
    instance.instance_id = "PARITY_WORKER_CALENDAR"
    instance.add_worker_unavailability(
        0,
        0.0,
        15.0,
        reason="training",
        source="calendar",
    )
    return instance


def _build_machine_breakdown_instance() -> SFJSSPInstance:
    instance = _build_single_operation_instance()
    instance.instance_id = "PARITY_MACHINE_BREAKDOWN"
    instance.add_machine_breakdown_event(
        0,
        0.0,
        12.0,
        source="event",
        details={"event_id": "BD-1"},
    )
    return instance


def _build_worker_absence_instance() -> SFJSSPInstance:
    instance = _build_single_operation_instance()
    instance.instance_id = "PARITY_WORKER_ABSENCE"
    instance.add_worker_absence_event(
        0,
        0.0,
        14.0,
        source="event",
        details={"event_id": "ABS-1"},
    )
    return instance


def test_greedy_env_and_nsga_match_on_transport_waiting_case():
    base_instance = _build_transport_waiting_instance()

    greedy_instance = _clone_instance(base_instance)
    greedy_schedule = GreedyScheduler(job_rule=fifo_rule).schedule(greedy_instance, verbose=False)

    env_instance = _clone_instance(base_instance)
    env_schedule = _run_env_on_reference_schedule(env_instance, greedy_schedule)

    nsga_instance = _clone_instance(base_instance)
    genome = schedule_to_sfjssp_genome(_clone_instance(base_instance), greedy_schedule)
    nsga_details = evaluate_sfjssp_genome_detailed(nsga_instance, genome)

    assert greedy_schedule.check_feasibility(greedy_instance) is True
    assert env_schedule.check_feasibility(env_instance) is True
    assert nsga_details["is_feasible"] is True
    assert nsga_details["constraint_violations"] == []
    assert _schedule_signature(greedy_schedule) == _schedule_signature(env_schedule)
    assert _schedule_signature(greedy_schedule) == _schedule_signature(nsga_details["schedule"])
    assert _metric_signature(greedy_instance, greedy_schedule) == _metric_signature(env_instance, env_schedule)
    assert _metric_signature(greedy_instance, greedy_schedule) == {
        "makespan": round(nsga_details["metrics"]["makespan"], 6),
        "total_energy": round(nsga_details["metrics"]["total_energy"], 6),
        "max_ergonomic_exposure": round(nsga_details["metrics"]["max_ergonomic_exposure"], 6),
        "total_labor_cost": round(nsga_details["metrics"]["total_labor_cost"], 6),
        "weighted_tardiness": round(nsga_details["metrics"]["weighted_tardiness"], 6),
        "n_tardy_jobs": int(nsga_details["metrics"]["n_tardy_jobs"]),
    }
    assert _schedule_signature(greedy_schedule)[(0, 1)]["start_time"] == 21.0


def test_greedy_env_and_nsga_match_on_setup_rest_and_ocra_case():
    base_instance = _build_setup_rest_ocra_instance()

    greedy_instance = _clone_instance(base_instance)
    greedy_schedule = GreedyScheduler(job_rule=fifo_rule).schedule(greedy_instance, verbose=False)

    env_instance = _clone_instance(base_instance)
    env_schedule = _run_env_on_reference_schedule(env_instance, greedy_schedule)

    nsga_instance = _clone_instance(base_instance)
    genome = schedule_to_sfjssp_genome(_clone_instance(base_instance), greedy_schedule)
    nsga_details = evaluate_sfjssp_genome_detailed(nsga_instance, genome)

    assert greedy_schedule.check_feasibility(greedy_instance) is True
    assert env_schedule.check_feasibility(env_instance) is True
    assert nsga_details["is_feasible"] is True
    env_signature = _schedule_signature(env_schedule)
    nsga_signature = _schedule_signature(nsga_details["schedule"])
    greedy_signature = _schedule_signature(greedy_schedule)
    assert env_signature == nsga_signature
    for key, greedy_op in greedy_signature.items():
        env_op = env_signature[key]
        assert greedy_op["machine_id"] == env_op["machine_id"]
        assert greedy_op["worker_id"] == env_op["worker_id"]
        assert greedy_op["mode_id"] == env_op["mode_id"]
        assert greedy_op["processing_time"] == env_op["processing_time"]
        assert greedy_op["setup_time"] == env_op["setup_time"]
        assert greedy_op["start_time"] == pytest.approx(env_op["start_time"], abs=0.1)
        assert greedy_op["completion_time"] == pytest.approx(env_op["completion_time"], abs=0.1)

    greedy_metrics = _metric_signature(greedy_instance, greedy_schedule)
    env_metrics = _metric_signature(env_instance, env_schedule)
    for metric_name in (
        "makespan",
        "total_energy",
        "max_ergonomic_exposure",
        "total_labor_cost",
        "weighted_tardiness",
    ):
        assert greedy_metrics[metric_name] == pytest.approx(env_metrics[metric_name], abs=0.1)
    assert greedy_metrics["n_tardy_jobs"] == env_metrics["n_tardy_jobs"]
    assert env_metrics == {
        "makespan": round(nsga_details["metrics"]["makespan"], 6),
        "total_energy": round(nsga_details["metrics"]["total_energy"], 6),
        "max_ergonomic_exposure": round(nsga_details["metrics"]["max_ergonomic_exposure"], 6),
        "total_labor_cost": round(nsga_details["metrics"]["total_labor_cost"], 6),
        "weighted_tardiness": round(nsga_details["metrics"]["weighted_tardiness"], 6),
        "n_tardy_jobs": int(nsga_details["metrics"]["n_tardy_jobs"]),
    }
    assert _metric_signature(greedy_instance, greedy_schedule)["max_ergonomic_exposure"] > 0.0


@pytest.mark.parametrize(
    ("builder", "expected_start"),
    [
        (_build_machine_calendar_instance, 20.0),
        (_build_worker_calendar_instance, 15.0),
        (_build_machine_breakdown_instance, 12.0),
        (_build_worker_absence_instance, 14.0),
    ],
)
def test_greedy_and_env_match_on_explicit_unavailability_cases(builder, expected_start):
    base_instance = builder()

    greedy_instance = _clone_instance(base_instance)
    greedy_schedule = GreedyScheduler(job_rule=fifo_rule).schedule(
        greedy_instance,
        verbose=False,
    )

    env_instance = _clone_instance(base_instance)
    env_schedule = _run_env_on_reference_schedule(env_instance, greedy_schedule)

    nsga_instance = _clone_instance(base_instance)
    genome = schedule_to_sfjssp_genome(_clone_instance(base_instance), greedy_schedule)
    nsga_details = evaluate_sfjssp_genome_detailed(nsga_instance, genome)

    greedy_signature = _schedule_signature(greedy_schedule)
    env_signature = _schedule_signature(env_schedule)
    nsga_signature = _schedule_signature(nsga_details["schedule"])

    assert greedy_schedule.check_feasibility(greedy_instance) is True
    assert env_schedule.check_feasibility(env_instance) is True
    assert nsga_details["is_feasible"] is True
    assert nsga_details["constraint_violations"] == []
    assert greedy_signature == env_signature == nsga_signature
    assert greedy_signature[(0, 0)]["start_time"] == expected_start
    assert _metric_signature(greedy_instance, greedy_schedule) == _metric_signature(
        env_instance,
        env_schedule,
    )
    assert _metric_signature(greedy_instance, greedy_schedule) == {
        "makespan": round(nsga_details["metrics"]["makespan"], 6),
        "total_energy": round(nsga_details["metrics"]["total_energy"], 6),
        "max_ergonomic_exposure": round(nsga_details["metrics"]["max_ergonomic_exposure"], 6),
        "total_labor_cost": round(nsga_details["metrics"]["total_labor_cost"], 6),
        "weighted_tardiness": round(nsga_details["metrics"]["weighted_tardiness"], 6),
        "n_tardy_jobs": int(nsga_details["metrics"]["n_tardy_jobs"]),
    }


def test_cp_matches_canonical_single_operation_case_when_ortools_installed():
    pytest.importorskip("ortools")

    base_instance = _build_single_operation_instance()
    greedy_instance = _clone_instance(base_instance)
    greedy_schedule = GreedyScheduler(job_rule=fifo_rule).schedule(greedy_instance, verbose=False)

    cp_instance = _clone_instance(base_instance)
    cp_schedule = solve_sfjssp(
        cp_instance,
        method="cp",
        objective="makespan",
        time_limit=5,
        verbose=False,
    )

    assert cp_schedule is not None
    assert cp_schedule.check_feasibility(cp_instance) is True
    assert cp_schedule.metadata["surrogate_timing"] is False
    assert _schedule_signature(greedy_schedule) == _schedule_signature(cp_schedule)
    assert _metric_signature(greedy_instance, greedy_schedule) == _metric_signature(cp_instance, cp_schedule)


def test_cp_matches_simple_precedence_fixture_when_ortools_installed():
    pytest.importorskip("ortools")

    base_instance = _build_simple_precedence_instance()
    greedy_instance = _clone_instance(base_instance)
    greedy_schedule = GreedyScheduler(job_rule=fifo_rule).schedule(greedy_instance, verbose=False)

    cp_instance = _clone_instance(base_instance)
    cp_schedule = solve_sfjssp(
        cp_instance,
        method="cp",
        objective="makespan",
        time_limit=5,
        verbose=False,
    )

    assert cp_schedule is not None
    assert cp_schedule.check_feasibility(cp_instance) is True
    assert cp_schedule.metadata["surrogate_timing"] is False
    assert _schedule_signature(greedy_schedule) == _schedule_signature(cp_schedule)
    assert _metric_signature(greedy_instance, greedy_schedule) == _metric_signature(cp_instance, cp_schedule)


def test_cp_matches_transport_waiting_fixture_when_ortools_installed():
    pytest.importorskip("ortools")

    base_instance = _build_transport_waiting_instance()
    greedy_instance = _clone_instance(base_instance)
    greedy_schedule = GreedyScheduler(job_rule=fifo_rule).schedule(greedy_instance, verbose=False)

    cp_instance = _clone_instance(base_instance)
    cp_schedule = solve_sfjssp(
        cp_instance,
        method="cp",
        objective="makespan",
        time_limit=5,
        verbose=False,
    )

    assert cp_schedule is not None
    assert cp_schedule.check_feasibility(cp_instance) is True
    assert cp_schedule.metadata["surrogate_timing"] is False
    assert cp_schedule.metadata["verified_scope"] == "narrow_canonical_makespan_fixtures"
    assert _schedule_signature(greedy_schedule) == _schedule_signature(cp_schedule)
    assert _metric_signature(greedy_instance, greedy_schedule) == _metric_signature(cp_instance, cp_schedule)
    assert _schedule_signature(cp_schedule)[(0, 1)]["start_time"] == 21.0


@pytest.mark.parametrize(
    ("builder", "expected_start"),
    [
        (_build_machine_calendar_instance, 20.0),
        (_build_worker_calendar_instance, 15.0),
        (_build_machine_breakdown_instance, 12.0),
        (_build_worker_absence_instance, 14.0),
    ],
)
def test_cp_matches_fixed_blackout_unavailability_fixtures_when_ortools_installed(
    builder,
    expected_start,
):
    pytest.importorskip("ortools")

    base_instance = builder()
    greedy_instance = _clone_instance(base_instance)
    greedy_schedule = GreedyScheduler(job_rule=fifo_rule).schedule(greedy_instance, verbose=False)

    cp_instance = _clone_instance(base_instance)
    cp_schedule = solve_sfjssp(
        cp_instance,
        method="cp",
        objective="makespan",
        time_limit=5,
        verbose=False,
    )

    assert cp_schedule is not None
    assert cp_schedule.check_feasibility(cp_instance) is True
    assert cp_schedule.metadata["surrogate_timing"] is False
    assert cp_schedule.metadata["calendar_event_semantics"] == "fixed_blackout_intervals"
    assert cp_schedule.metadata["calendar_event_semantics_verified"] is True
    assert _schedule_signature(greedy_schedule) == _schedule_signature(cp_schedule)
    assert _metric_signature(greedy_instance, greedy_schedule) == _metric_signature(cp_instance, cp_schedule)
    assert _schedule_signature(cp_schedule)[(0, 0)]["start_time"] == expected_start


def test_nsga_infeasible_payload_preserves_assignment_reason():
    instance = _build_single_operation_instance()
    invalid_genome = {
        "sequence": np.array([0], dtype=int),
        "machines": np.array([1], dtype=int),
        "workers": np.array([0], dtype=int),
        "modes": np.array([0], dtype=int),
        "offsets": np.array([0], dtype=int),
        "op_list": [(0, 0)],
    }

    details = evaluate_sfjssp_genome_detailed(instance, invalid_genome)

    assert details["is_feasible"] is False
    assert details["penalties"]["hard_violations"] == 1
    assert details["schedule"] is None
    assert any("Ineligible machine assignment" in reason for reason in details["constraint_violations"])
