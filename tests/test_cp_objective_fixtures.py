import pytest

try:
    from ..exact_solvers.cp_solver import CPScheduler, solve_sfjssp
    from ..sfjssp_model.instance import InstanceType, SFJSSPInstance
    from ..sfjssp_model.job import Job, Operation
    from ..sfjssp_model.machine import Machine, MachineMode
    from ..sfjssp_model.schedule import Schedule
    from ..sfjssp_model.worker import Worker
except ImportError:  # pragma: no cover - supports repo-root imports
    from exact_solvers.cp_solver import CPScheduler, solve_sfjssp
    from sfjssp_model.instance import InstanceType, SFJSSPInstance
    from sfjssp_model.job import Job, Operation
    from sfjssp_model.machine import Machine, MachineMode
    from sfjssp_model.schedule import Schedule
    from sfjssp_model.worker import Worker


def _single_op_schedule(instance: SFJSSPInstance, machine_id: int) -> Schedule:
    op = instance.jobs[0].operations[0]
    processing_time = op.processing_times[machine_id][0]
    schedule = Schedule(instance_id=instance.instance_id)
    schedule.add_operation(
        job_id=0,
        op_id=0,
        machine_id=machine_id,
        worker_id=0,
        mode_id=0,
        start_time=0.0,
        completion_time=processing_time,
        processing_time=processing_time,
    )
    return schedule


def _build_energy_tradeoff_instance() -> SFJSSPInstance:
    instance = SFJSSPInstance(
        instance_id="CP_ENERGY_FIXTURE",
        instance_type=InstanceType.STATIC,
        auxiliary_power_total=0.0,
    )
    instance.add_machine(
        Machine(
            machine_id=0,
            power_processing=50.0,
            power_idle=0.0,
            power_setup=0.0,
            startup_energy=0.0,
            setup_time=0.0,
            modes=[MachineMode(mode_id=0, speed_factor=1.0, power_multiplier=1.0)],
        )
    )
    instance.add_machine(
        Machine(
            machine_id=1,
            power_processing=1.0,
            power_idle=0.0,
            power_setup=0.0,
            startup_energy=20.0,
            setup_time=0.0,
            modes=[MachineMode(mode_id=0, speed_factor=1.0, power_multiplier=1.0)],
        )
    )
    instance.add_worker(
        Worker(
            worker_id=0,
            labor_cost_per_hour=0.0,
            fatigue_rate=0.0,
            recovery_rate=0.0,
            ocra_max_per_shift=1000.0,
            min_rest_fraction=0.0,
        )
    )
    operation = Operation(
        job_id=0,
        op_id=0,
        eligible_machines={0, 1},
        eligible_workers={0},
        processing_times={
            0: {0: 10.0},
            1: {0: 20.0},
        },
    )
    instance.add_job(Job(job_id=0, operations=[operation], arrival_time=0.0, due_date=1000.0))
    instance.ergonomic_risk_map[(0, 0)] = 0.0
    return instance


def _build_composite_tradeoff_instance() -> SFJSSPInstance:
    instance = SFJSSPInstance(
        instance_id="CP_COMPOSITE_FIXTURE",
        instance_type=InstanceType.STATIC,
        auxiliary_power_total=0.0,
    )
    instance.add_machine(
        Machine(
            machine_id=0,
            power_processing=1.0,
            power_idle=0.0,
            power_setup=0.0,
            startup_energy=100.0,
            setup_time=0.0,
            modes=[MachineMode(mode_id=0, speed_factor=1.0, power_multiplier=1.0)],
        )
    )
    instance.add_machine(
        Machine(
            machine_id=1,
            power_processing=1.0,
            power_idle=0.0,
            power_setup=0.0,
            startup_energy=0.0,
            setup_time=0.0,
            modes=[MachineMode(mode_id=0, speed_factor=1.0, power_multiplier=1.0)],
        )
    )
    instance.add_worker(
        Worker(
            worker_id=0,
            labor_cost_per_hour=0.0,
            fatigue_rate=0.0,
            recovery_rate=0.0,
            ocra_max_per_shift=1000.0,
            min_rest_fraction=0.0,
        )
    )
    operation = Operation(
        job_id=0,
        op_id=0,
        eligible_machines={0, 1},
        eligible_workers={0},
        processing_times={
            0: {0: 10.0},
            1: {0: 20.0},
        },
    )
    instance.add_job(Job(job_id=0, operations=[operation], arrival_time=0.0, due_date=1000.0, weight=1.0))
    instance.ergonomic_risk_map[(0, 0)] = 0.0
    return instance


def test_cp_energy_objective_matches_schedule_energy_ranking():
    pytest.importorskip("ortools")

    instance = _build_energy_tradeoff_instance()
    manual_energies = {}
    for machine_id in (0, 1):
        schedule = _single_op_schedule(instance, machine_id)
        manual_energies[machine_id] = schedule.evaluate(instance)["total_energy"]

    expected_machine = min(manual_energies, key=manual_energies.get)

    with pytest.warns(UserWarning, match="experimental"):
        schedule = solve_sfjssp(
            instance,
            method="cp",
            objective="energy",
            time_limit=5,
            verbose=False,
        )

    assert schedule is not None
    assert schedule.get_operation(0, 0).machine_id == expected_machine


def test_cp_composite_objective_matches_schedule_composite_ranking():
    pytest.importorskip("ortools")

    instance = _build_composite_tradeoff_instance()
    score_weights = {
        "makespan": 0.4,
        "total_energy": 0.3,
        "max_ergonomic_exposure": 0.2,
        "total_labor_cost": 0.1,
    }
    solver_weights = {
        "makespan": 0.4,
        "energy": 0.3,
        "ergonomic": 0.2,
        "labor_cost": 0.1,
    }

    manual_scores = {}
    for machine_id in (0, 1):
        schedule = _single_op_schedule(instance, machine_id)
        manual_scores[machine_id] = schedule.evaluate(
            instance,
            weights=score_weights,
        )["composite_score"]

    expected_machine = min(manual_scores, key=manual_scores.get)

    scheduler = CPScheduler(time_limit=5, energy_weights=solver_weights)
    with pytest.warns(UserWarning, match="experimental"):
        schedule = scheduler.solve(
            instance,
            objective="composite",
            verbose=False,
        )

    assert schedule is not None
    assert schedule.get_operation(0, 0).machine_id == expected_machine
