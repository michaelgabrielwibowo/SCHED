from pathlib import Path

import pytest

try:
    from ..experiments.compare_solvers import load_benchmark
    from ..exact_solvers.cp_solver import MIPScheduler, solve_sfjssp
    from ..sfjssp_model.instance import SFJSSPInstance
    from ..sfjssp_model.job import Job, Operation
    from ..sfjssp_model.machine import Machine, MachineMode
    from ..sfjssp_model.worker import Worker
    from ..training.train_drl import (
        TrainingConfig,
        TrainingPipeline,
        run_training,
    )
    from ..environment.sfjssp_env import SFJSSPEnv
    from ..experiments.generate_benchmarks import (
        BenchmarkGenerator,
        GeneratorConfig,
        InstanceSize,
    )
except ImportError:  # pragma: no cover - supports repo-root imports
    from experiments.compare_solvers import load_benchmark
    from exact_solvers.cp_solver import MIPScheduler, solve_sfjssp
    from sfjssp_model.instance import SFJSSPInstance
    from sfjssp_model.job import Job, Operation
    from sfjssp_model.machine import Machine, MachineMode
    from sfjssp_model.worker import Worker
    from training.train_drl import (
        TrainingConfig,
        TrainingPipeline,
        run_training,
    )
    from environment.sfjssp_env import SFJSSPEnv
    from experiments.generate_benchmarks import (
        BenchmarkGenerator,
        GeneratorConfig,
        InstanceSize,
    )


def _build_cp_fixture(setup_time: float = 0.0) -> SFJSSPInstance:
    instance = SFJSSPInstance(instance_id=f"cp_fixture_setup_{setup_time}")
    instance.add_machine(
        Machine(
            machine_id=0,
            setup_time=setup_time,
            modes=[MachineMode(mode_id=0)],
        )
    )
    instance.add_worker(
        Worker(
            worker_id=0,
            min_rest_fraction=0.0,
            ocra_max_per_shift=999.0,
            fatigue_rate=0.0,
            recovery_rate=0.0,
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


def test_cp_solver_smoke_when_ortools_installed():
    pytest.importorskip("ortools")

    instance = _build_cp_fixture(setup_time=0.0)
    schedule = solve_sfjssp(
        instance,
        method="cp",
        objective="makespan",
        time_limit=5,
        verbose=False,
    )

    assert schedule is not None
    assert schedule.is_feasible
    assert len(schedule.scheduled_ops) == 1
    assert schedule.metadata["solver"] == "cp_sat"
    assert schedule.metadata["solver_objective"] == "makespan"
    assert schedule.metadata["surrogate_objective"] is False
    assert schedule.metadata["surrogate_timing"] is False


def test_mip_solver_is_quarantined():
    scheduler = MIPScheduler(time_limit=1)
    with pytest.raises(NotImplementedError, match="quarantined"):
        scheduler.solve(None, verbose=False)


def test_cp_non_makespan_objective_warns_when_ortools_installed():
    pytest.importorskip("ortools")

    instance = _build_cp_fixture(setup_time=0.0)

    with pytest.warns(UserWarning, match="experimental"):
        schedule = solve_sfjssp(
            instance,
            method="cp",
            objective="energy",
            time_limit=1,
            verbose=False,
        )

    assert schedule is not None
    assert schedule.metadata["surrogate_objective"] is True
    assert schedule.metadata["solver_objective"] == "energy"


def test_cp_setup_gap_truth_metadata_when_ortools_installed():
    pytest.importorskip("ortools")

    instance = _build_cp_fixture(setup_time=5.0)
    with pytest.warns(UserWarning, match="infeasible under the canonical"):
        schedule = solve_sfjssp(
            instance,
            method="cp",
            objective="makespan",
            time_limit=5,
            verbose=False,
        )

    assert schedule is not None
    assert schedule.is_feasible is False
    assert schedule.metadata["surrogate_timing"] is True
    assert schedule.metadata["surrogate_timing_reason"] == "machine_setup_not_encoded_in_cp_intervals"


def test_solve_sfjssp_mip_method_is_quarantined():
    with pytest.raises(NotImplementedError, match="quarantined"):
        solve_sfjssp(_build_cp_fixture(), method="mip", verbose=False)


def test_torch_training_smoke_when_torch_installed(tmp_path):
    pytest.importorskip("torch")

    result = run_training(output_dir=str(tmp_path), n_episodes=1)

    assert result["episodes_trained"] == 1
    assert result["best_makespan"] is not None
    assert result["best_makespan"] > 0
    assert (tmp_path / "history.json").exists()


def test_torch_checkpoint_round_trip_when_torch_installed(tmp_path):
    pytest.importorskip("torch")

    generator = BenchmarkGenerator(GeneratorConfig(size=InstanceSize.SMALL, seed=7))
    instance = generator.generate()
    env = SFJSSPEnv(instance, use_graph_state=True)

    pipeline = TrainingPipeline(
        TrainingConfig(n_episodes=1, max_steps_per_episode=10, log_interval=1)
    )
    history = pipeline.train(env, n_episodes=1, verbose=False)
    pipeline.save(str(tmp_path))

    restored = TrainingPipeline(
        TrainingConfig(n_episodes=1, max_steps_per_episode=10, log_interval=1)
    )
    restored.load(str(tmp_path))

    assert len(history["rewards"]) == 1
    assert (tmp_path / "agent.pt").exists()
    assert (tmp_path / "config.json").exists()
    assert restored.training_history == pipeline.training_history
