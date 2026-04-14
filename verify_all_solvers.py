import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np

try:
    import torch  # noqa: F401
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

try:
    import ortools  # noqa: F401
    ORTOOLS_INSTALLED = True
except ImportError:
    ORTOOLS_INSTALLED = False

try:
    from .environment.sfjssp_env import SFJSSPEnv
    from .experiments.generate_benchmarks import (
        BenchmarkGenerator,
        GeneratorConfig,
        InstanceSize,
    )
    from .experiments.compare_solvers import load_benchmark
    from .exact_solvers.cp_solver import MIPScheduler, solve_sfjssp
    from .training.train_drl import TrainingConfig, TrainingPipeline, run_training
except ImportError:  # pragma: no cover - supports repo-root imports
    from environment.sfjssp_env import SFJSSPEnv
    from experiments.generate_benchmarks import (
        BenchmarkGenerator,
        GeneratorConfig,
        InstanceSize,
    )
    from experiments.compare_solvers import load_benchmark
    from exact_solvers.cp_solver import MIPScheduler, solve_sfjssp
    from training.train_drl import TrainingConfig, TrainingPipeline, run_training


REPO_ROOT = Path(__file__).resolve().parent


@contextmanager
def _repo_temp_dir(prefix: str):
    """Create a repo-local temp directory for sandbox-safe smoke tests."""

    root = REPO_ROOT / ".tmp_verify"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{prefix}{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_waiting_logic():
    print("Testing environment waiting logic...")
    config = GeneratorConfig(seed=42, n_jobs=1, n_machines=5, n_workers=5)
    gen = BenchmarkGenerator(config)
    instance = gen.generate()

    op = instance.jobs[0].operations[0]
    m_id = list(op.eligible_machines)[0]
    w_id = list(op.eligible_workers)[0]
    worker = instance.get_worker(w_id)

    env = SFJSSPEnv(instance)
    env.reset()
    env.current_time = instance.jobs[0].arrival_time

    worker.record_work(470, current_time=0.0)
    op.processing_times[m_id] = {0: 30.0}

    print(f"  Testing Op(J0,O0) on M{m_id} by W{w_id}")

    job_mask = env._compute_job_mask()
    res_mask = env.compute_resource_mask(0)

    if job_mask[0] == 1.0 and res_mask[m_id, w_id, 0] == 1.0:
        print("  Masks allow the task with a deferred start.")
    else:
        print("  Action mask incorrectly blocks the task.")
        return False

    action = {"job_idx": 0, "machine_idx": m_id, "worker_idx": w_id, "mode_idx": 0}
    env.step(action)

    print(f"  Task start time: {op.start_time}")
    print(f"  Env current time: {env.current_time}")

    if op.start_time >= 480:
        print("  Waiting logic behaves as expected.")
        return True

    print(f"  Task started too early at {op.start_time}")
    return False


def test_counters():
    print("\nTesting machine counters...")
    config = GeneratorConfig(seed=42, n_jobs=1, n_machines=1, n_workers=1)
    gen = BenchmarkGenerator(config)
    instance = gen.generate()

    op = instance.jobs[0].operations[0]
    m_id = list(op.eligible_machines)[0]
    w_id = list(op.eligible_workers)[0]

    machine = instance.get_machine(m_id)
    machine.setup_time = 15.0

    env = SFJSSPEnv(instance)
    env.reset()
    env.current_time = 50.0

    action = {"job_idx": 0, "machine_idx": m_id, "worker_idx": w_id, "mode_idx": 0}
    env.step(action)

    print(f"  Total idle: {machine.total_idle_time}")
    print(f"  Total setup: {machine.total_setup_time}")

    if np.isclose(machine.total_idle_time, 35.0) and np.isclose(machine.total_setup_time, 15.0):
        print("  Machine counters are consistent.")
        return True

    print("  Machine counters do not match the expected split.")
    return False


def test_cp_solver_optional():
    print("\nTesting CP exact solver (makespan only)...")
    if not ORTOOLS_INSTALLED:
        print("  Skipped: OR-Tools is not installed.")
        return True

    benchmark_paths = [
        Path("benchmarks/small/SFJSSP_small_000.json"),
        Path("benchmarks/small/SFJSSP_small_001.json"),
    ]

    for benchmark_path in benchmark_paths:
        instance = load_benchmark(str(benchmark_path))
        schedule = solve_sfjssp(
            instance,
            method="cp",
            objective="makespan",
            time_limit=60,
            verbose=False,
        )

        if schedule is None:
            print(f"  CP solver returned no schedule for {benchmark_path.name}.")
            return False

        print(f"  {benchmark_path.name}: makespan={schedule.compute_makespan():.1f}, feasible={schedule.is_feasible}")

        if not (
            schedule.is_feasible
            and len(schedule.scheduled_ops) == sum(len(job.operations) for job in instance.jobs)
        ):
            print(f"  CP exact solver returned an infeasible or incomplete schedule for {benchmark_path.name}.")
            return False

    print("  CP exact solver smoke passed on multiple stored benchmarks for objective='makespan'.")
    return True


def test_mip_quarantine_optional():
    print("\nTesting MIP quarantine...")
    if not ORTOOLS_INSTALLED:
        print("  Skipped: OR-Tools is not installed.")
        return True

    try:
        scheduler = MIPScheduler(time_limit=1)
        scheduler.solve(None, verbose=False)
    except NotImplementedError as exc:
        print(f"  Expected quarantine message: {exc}")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"  Unexpected MIP failure type: {type(exc).__name__}: {exc}")
        return False

    print("  MIP scheduler should not be callable while quarantined.")
    return False


def test_torch_training_optional():
    print("\nTesting torch-backed DRL smoke...")
    if not TORCH_INSTALLED:
        print("  Skipped: torch is not installed.")
        return True

    with _repo_temp_dir("sfjssp_drl_") as tmp_dir:
        result = run_training(output_dir=tmp_dir, n_episodes=1)
        history_path = Path(tmp_dir) / "history.json"
        history_exists = history_path.exists()

    if result["episodes_trained"] != 1:
        print("  Training did not report a single completed episode.")
        return False

    if result["best_makespan"] is None or result["best_makespan"] <= 0:
        print("  Training reported an invalid makespan.")
        return False

    if not history_exists:
        print("  Training history.json was not written.")
        return False

    print(f"  Torch training makespan: {result['best_makespan']:.1f}")
    print("  Torch-backed DRL smoke passed.")
    return True


def test_torch_checkpoint_round_trip_optional():
    print("\nTesting torch checkpoint round-trip...")
    if not TORCH_INSTALLED:
        print("  Skipped: torch is not installed.")
        return True

    generator = BenchmarkGenerator(GeneratorConfig(size=InstanceSize.SMALL, seed=7))
    instance = generator.generate()
    env = SFJSSPEnv(instance, use_graph_state=True)

    with _repo_temp_dir("sfjssp_ckpt_") as tmp_dir:
        pipeline = TrainingPipeline(
            TrainingConfig(n_episodes=1, max_steps_per_episode=10, log_interval=1)
        )
        pipeline.train(env, n_episodes=1, verbose=False)
        pipeline.save(tmp_dir)

        restored = TrainingPipeline(
            TrainingConfig(n_episodes=1, max_steps_per_episode=10, log_interval=1)
        )
        restored.load(tmp_dir)

        if restored.training_history != pipeline.training_history:
            print("  Training history did not survive checkpoint round-trip.")
            return False

        if not (Path(tmp_dir) / "agent.pt").exists():
            print("  agent.pt was not written.")
            return False

    print("  Torch checkpoint round-trip passed.")
    return True


if __name__ == "__main__":
    checks = [
        test_waiting_logic,
        test_counters,
        test_cp_solver_optional,
        test_mip_quarantine_optional,
        test_torch_training_optional,
        test_torch_checkpoint_round_trip_optional,
    ]
    passed = all(check() for check in checks)
    if passed:
        print("\nVerification checks passed.")
    else:
        sys.exit(1)
