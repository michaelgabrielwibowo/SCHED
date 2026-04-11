from pathlib import Path

import pytest

try:
    from ..experiments.compare_solvers import load_benchmark
    from ..exact_solvers.cp_solver import MIPScheduler, solve_sfjssp
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


def test_cp_solver_smoke_when_ortools_installed():
    pytest.importorskip("ortools")

    benchmark_dir = Path(__file__).resolve().parents[1] / "benchmarks" / "small"
    benchmark_paths = [
        benchmark_dir / "SFJSSP_small_000.json",
        benchmark_dir / "SFJSSP_small_001.json",
    ]

    for benchmark in benchmark_paths:
        instance = load_benchmark(str(benchmark))
        schedule = solve_sfjssp(
            instance,
            method="cp",
            objective="makespan",
            time_limit=5,
            verbose=False,
        )

        assert schedule is not None
        assert schedule.is_feasible
        assert len(schedule.scheduled_ops) == sum(len(job.operations) for job in instance.jobs)


def test_mip_solver_is_quarantined_when_ortools_installed():
    pytest.importorskip("ortools")

    scheduler = MIPScheduler(time_limit=1)
    with pytest.raises(NotImplementedError, match="quarantined"):
        scheduler.solve(None, verbose=False)


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
