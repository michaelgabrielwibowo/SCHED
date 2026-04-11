try:
    from ..baseline_solver.greedy_solvers import GreedyScheduler
except ImportError:  # pragma: no cover - supports repo-root imports
    from baseline_solver.greedy_solvers import GreedyScheduler


def test_greedy_scheduler_small_benchmark_regression(small_benchmark_instance):
    schedule = GreedyScheduler().schedule(small_benchmark_instance, verbose=False)

    assert small_benchmark_instance.n_jobs > 0
    assert schedule.is_feasible is True
    assert schedule.makespan > 0.0


def test_greedy_scheduler_medium_benchmark_regression(medium_benchmark_instance):
    schedule = GreedyScheduler().schedule(medium_benchmark_instance, verbose=False)

    assert medium_benchmark_instance.n_jobs > 0
    assert schedule.is_feasible is True
    assert schedule.makespan > 0.0
