#!/usr/bin/env python
"""
Sweep NSGA-III budget settings on the stored benchmark slice.

This keeps the current decoder and warm-start semantics fixed and answers a
single question: how much residual tardiness drops as search budget increases?
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from ..experiments.artifact_schemas import BUDGET_SWEEP_ARTIFACT_SCHEMA
    from ..baseline_solver.greedy_solvers import edt_rule, fifo_rule, spt_rule
    from ..experiments.compare_solvers import (
        _get_git_commit,
        _get_git_status_short,
        load_benchmark,
        run_greedy_experiment,
        run_nsga3_experiment,
    )
    from ..moea.nsga3 import NSGA3_DEFAULT_CROSSOVER_POLICY
    from ..moea.nsga3 import NSGA3_DEFAULT_SEQUENCE_MUTATION
    from ..moea.nsga3 import NSGA3_DEFAULT_IMMIGRANT_POLICY
except ImportError:  # pragma: no cover - supports repo-root imports
    from experiments.artifact_schemas import BUDGET_SWEEP_ARTIFACT_SCHEMA
    from baseline_solver.greedy_solvers import edt_rule, fifo_rule, spt_rule
    from experiments.compare_solvers import (
        _get_git_commit,
        _get_git_status_short,
        load_benchmark,
        run_greedy_experiment,
        run_nsga3_experiment,
    )
    from moea.nsga3 import NSGA3_DEFAULT_CROSSOVER_POLICY
    from moea.nsga3 import NSGA3_DEFAULT_SEQUENCE_MUTATION
    from moea.nsga3 import NSGA3_DEFAULT_IMMIGRANT_POLICY


def _build_sweep_provenance(
    benchmark_dir: str,
    output_path: str,
    generations: Sequence[int],
    population_sizes: Sequence[int],
    warm_start: bool,
    seed: int,
    constraint_handling: str,
    crossover_policy: str,
    sequence_mutation: str,
    immigrant_policy: str,
    immigrant_count: int,
    immigrant_period: int,
    immigrant_archive_size: int,
    report_member_policy: str,
    command: str = "",
) -> Dict[str, Any]:
    """Build provenance metadata for the NSGA budget sweep artifact."""
    git_status_short = _get_git_status_short()
    return {
        "artifact_schema": BUDGET_SWEEP_ARTIFACT_SCHEMA,
        "git_commit": _get_git_commit(),
        "git_dirty": bool(git_status_short),
        "git_status_short": git_status_short,
        "command": command or "python -m experiments.sweep_nsga_budget",
        "benchmark_dir": benchmark_dir,
        "output_path": output_path,
        "generations": list(generations),
        "population_sizes": list(population_sizes),
        "nsga3_warm_start": warm_start,
        "nsga3_seed": seed,
        "nsga3_constraint_handling": constraint_handling,
        "nsga3_crossover_policy": crossover_policy,
        "nsga3_sequence_mutation": sequence_mutation,
        "nsga3_immigrant_policy": immigrant_policy,
        "nsga3_immigrant_count": immigrant_count,
        "nsga3_immigrant_period": immigrant_period,
        "nsga3_immigrant_archive_size": immigrant_archive_size,
        "nsga3_report_member_policy": report_member_policy,
        "baseline_scope": "published greedy comparison slice (SPT, FIFO, EDD)",
        "python_version": sys.version.split()[0],
        "timestamp": datetime.now().isoformat(),
    }


def _parse_csv_ints(raw: str) -> List[int]:
    """Parse a comma-separated integer list."""
    values = [chunk.strip() for chunk in raw.split(",")]
    return [int(value) for value in values if value]


def _average(values: Iterable[float]) -> Optional[float]:
    items = [float(value) for value in values]
    if not items:
        return None
    return sum(items) / len(items)


def _json_ready(value: Any) -> Any:
    """Recursively convert NumPy-backed scalars into JSON-native Python values."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _build_greedy_baseline(instance: Any) -> Dict[str, Any]:
    """Compute the published greedy baseline slice used by comparison artifacts."""
    runs = [
        run_greedy_experiment(instance, "SPT", spt_rule),
        run_greedy_experiment(instance, "FIFO", fifo_rule),
        run_greedy_experiment(instance, "EDD", edt_rule),
    ]
    best = min(runs, key=lambda result: result["makespan"])
    return {
        "best_method": best["method"],
        "best_makespan": best["makespan"],
        "runs": runs,
    }


def _normalize_budget_run(
    result: Dict[str, Any],
    generations: int,
    population_size: int,
    best_greedy_makespan: float,
) -> Dict[str, Any]:
    """Normalize one NSGA budget run into the sweep artifact schema."""
    report_metrics = result.get("report_member_metrics") or {}
    report_penalties = result.get("report_member_penalties") or {}
    representative_members = result.get("representative_members") or {}
    tardiness_best = representative_members.get("min_weighted_tardiness_feasible") or {}
    tardiness_best_metrics = tardiness_best.get("metrics") or {}
    tardiness_best_penalties = tardiness_best.get("penalties") or {}
    report_makespan = result.get("report_makespan")
    if report_makespan is None:
        report_makespan = report_metrics.get("makespan")
    tardiness_best_n_tardy_jobs = result.get("tardiness_best_n_tardy_jobs")
    return {
        "generations": generations,
        "population_size": population_size,
        "seed": result.get("seed"),
        "time_seconds": result.get("time_seconds"),
        "feasible": result.get("feasible"),
        "warm_start": result.get("warm_start"),
        "constraint_handling": result.get("constraint_handling"),
        "report_member_key": result.get("report_member_key"),
        "report_member_selection_policy": result.get("report_member_selection_policy"),
        "report_member_is_feasible": result.get("report_member_is_feasible"),
        "report_member_metrics": report_metrics or None,
        "report_member_penalties": report_penalties or None,
        "report_member_constraint_violations": result.get("report_member_constraint_violations") or [],
        "report_makespan": report_makespan,
        "report_total_energy": result.get("report_total_energy"),
        "report_max_ergonomic_exposure": result.get("report_max_ergonomic_exposure"),
        "report_total_labor_cost": result.get("report_total_labor_cost"),
        "report_total_tardiness": result.get("report_total_tardiness"),
        "report_n_tardy_jobs": result.get("report_n_tardy_jobs"),
        "report_weighted_tardiness": result.get("report_weighted_tardiness"),
        "report_total_penalty": result.get("report_total_penalty"),
        "tardiness_best_member_metrics": tardiness_best_metrics or None,
        "tardiness_best_member_penalties": tardiness_best_penalties or None,
        "tardiness_best_member_constraint_violations": result.get("tardiness_best_member_constraint_violations") or [],
        "tardiness_best_n_tardy_jobs": tardiness_best_n_tardy_jobs,
        "tardiness_best_weighted_tardiness": result.get("tardiness_best_weighted_tardiness"),
        "tardiness_best_makespan": result.get("tardiness_best_makespan"),
        "min_n_tardy_jobs": result.get("min_n_tardy_jobs"),
        "min_weighted_tardiness": result.get("min_weighted_tardiness"),
        "min_total_penalty": result.get("min_total_penalty"),
        "zero_tardy_feasible_pareto_size": result.get("zero_tardy_feasible_pareto_size"),
        "zero_penalty_pareto_size": result.get("zero_penalty_pareto_size"),
        "feasible_pareto_size": result.get("feasible_pareto_size"),
        "pareto_size": result.get("pareto_size"),
        "beats_best_published_greedy_makespan": (
            bool(report_makespan < best_greedy_makespan)
            if report_makespan is not None
            else False
        ),
        "report_member_zero_tardy": bool(result.get("report_n_tardy_jobs") == 0) if result.get("report_n_tardy_jobs") is not None else False,
        "tardiness_best_zero_tardy": bool(tardiness_best_n_tardy_jobs == 0) if tardiness_best_n_tardy_jobs is not None else False,
        "tardiness_best_improves_report": (
            bool(
                result.get("tardiness_best_weighted_tardiness") is not None
                and result.get("report_weighted_tardiness") is not None
                and result["tardiness_best_weighted_tardiness"] < result["report_weighted_tardiness"]
            )
        ),
    }


def _summarize_budget_sweep(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate per-instance sweep results into per-config summary rows."""
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for instance_result in results:
        for run in instance_result["runs"]:
            key = (run["generations"], run["population_size"])
            grouped.setdefault(key, []).append(run)

    summary: List[Dict[str, Any]] = []
    for generations, population_size in sorted(grouped):
        runs = grouped[(generations, population_size)]
        summary.append(
            {
                "generations": generations,
                "population_size": population_size,
                "instances": len(runs),
                "constraint_handling": runs[0].get("constraint_handling"),
                "report_member_policy": runs[0].get("report_member_key"),
                "avg_report_makespan": _average(
                    run["report_member_metrics"]["makespan"]
                    for run in runs
                    if run.get("report_member_metrics")
                ),
                "avg_report_total_energy": _average(
                    run["report_total_energy"]
                    for run in runs
                    if run.get("report_total_energy") is not None
                ),
                "avg_report_max_ergonomic_exposure": _average(
                    run["report_max_ergonomic_exposure"]
                    for run in runs
                    if run.get("report_max_ergonomic_exposure") is not None
                ),
                "avg_report_total_labor_cost": _average(
                    run["report_total_labor_cost"]
                    for run in runs
                    if run.get("report_total_labor_cost") is not None
                ),
                "avg_report_weighted_tardiness": _average(
                    run["report_weighted_tardiness"]
                    for run in runs
                    if run.get("report_weighted_tardiness") is not None
                ),
                "avg_tardiness_best_makespan": _average(
                    run["tardiness_best_member_metrics"]["makespan"]
                    for run in runs
                    if run.get("tardiness_best_member_metrics")
                ),
                "avg_tardiness_best_weighted_tardiness": _average(
                    run["tardiness_best_weighted_tardiness"]
                    for run in runs
                    if run.get("tardiness_best_weighted_tardiness") is not None
                ),
                "avg_report_n_tardy_jobs": _average(
                    run["report_n_tardy_jobs"]
                    for run in runs
                    if run.get("report_n_tardy_jobs") is not None
                ),
                "avg_tardiness_best_n_tardy_jobs": _average(
                    run["tardiness_best_n_tardy_jobs"]
                    for run in runs
                    if run.get("tardiness_best_n_tardy_jobs") is not None
                ),
                "avg_report_total_penalty": _average(
                    run["report_total_penalty"]
                    for run in runs
                    if run.get("report_total_penalty") is not None
                ),
                "avg_min_total_penalty": _average(
                    run["min_total_penalty"]
                    for run in runs
                    if run.get("min_total_penalty") is not None
                ),
                "instances_beating_best_published_greedy_makespan": sum(
                    1 for run in runs if run["beats_best_published_greedy_makespan"]
                ),
                "instances_with_zero_tardy_feasible_member": sum(
                    1 for run in runs if (run.get("zero_tardy_feasible_pareto_size") or 0) > 0
                ),
                "instances_with_report_zero_tardy_member": sum(
                    1 for run in runs if run["report_member_zero_tardy"]
                ),
                "instances_with_tardiness_best_zero_tardy_member": sum(
                    1 for run in runs if run["tardiness_best_zero_tardy"]
                ),
                "instances_where_tardiness_best_improves_report": sum(
                    1 for run in runs if run["tardiness_best_improves_report"]
                ),
                "avg_time_seconds": _average(
                    run["time_seconds"] for run in runs if run.get("time_seconds") is not None
                ),
            }
        )
    return summary


def _recommend_budget(summary: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose a budget recommendation with tardiness first, then makespan, then time."""
    if not summary:
        return None
    return min(
        summary,
        key=lambda row: (
            row["avg_report_weighted_tardiness"] if row["avg_report_weighted_tardiness"] is not None else float("inf"),
            row["avg_report_n_tardy_jobs"] if row["avg_report_n_tardy_jobs"] is not None else float("inf"),
            row["avg_tardiness_best_weighted_tardiness"] if row["avg_tardiness_best_weighted_tardiness"] is not None else float("inf"),
            row["avg_tardiness_best_n_tardy_jobs"] if row["avg_tardiness_best_n_tardy_jobs"] is not None else float("inf"),
            row["avg_report_makespan"] if row["avg_report_makespan"] is not None else float("inf"),
            row["avg_time_seconds"] if row["avg_time_seconds"] is not None else float("inf"),
        ),
    )


def print_summary(summary: Sequence[Dict[str, Any]], recommended: Optional[Dict[str, Any]]) -> None:
    """Print the sweep summary table."""
    print("\n" + "=" * 132)
    print("NSGA-III BUDGET SWEEP SUMMARY")
    print("=" * 132)
    print(
        f"{'Gen':>6} {'Pop':>6} {'Rpt M':>10} {'Rpt WTd':>12} {'Best WTd':>12} "
        f"{'Rpt Tardy':>10} {'Best Tardy':>11} {'Improve':>9} {'Time(s)':>10}"
    )
    print("-" * 132)

    for row in summary:
        avg_makespan = (
            f"{row['avg_report_makespan']:.1f}"
            if row["avg_report_makespan"] is not None
            else "N/A"
        )
        avg_report_weighted_tardiness = (
            f"{row['avg_report_weighted_tardiness']:.1f}"
            if row["avg_report_weighted_tardiness"] is not None
            else "N/A"
        )
        avg_tardiness_best_weighted_tardiness = (
            f"{row['avg_tardiness_best_weighted_tardiness']:.1f}"
            if row["avg_tardiness_best_weighted_tardiness"] is not None
            else "N/A"
        )
        avg_report_n_tardy_jobs = (
            f"{row['avg_report_n_tardy_jobs']:.2f}"
            if row["avg_report_n_tardy_jobs"] is not None
            else "N/A"
        )
        avg_tardiness_best_n_tardy_jobs = (
            f"{row['avg_tardiness_best_n_tardy_jobs']:.2f}"
            if row["avg_tardiness_best_n_tardy_jobs"] is not None
            else "N/A"
        )
        avg_time_seconds = (
            f"{row['avg_time_seconds']:.2f}"
            if row["avg_time_seconds"] is not None
            else "N/A"
        )
        print(
            f"{row['generations']:>6} {row['population_size']:>6} "
            f"{avg_makespan:>10} {avg_report_weighted_tardiness:>12} {avg_tardiness_best_weighted_tardiness:>12} "
            f"{avg_report_n_tardy_jobs:>10} {avg_tardiness_best_n_tardy_jobs:>11} "
            f"{row['instances_where_tardiness_best_improves_report']:>9} "
            f"{avg_time_seconds:>10}"
        )

    if recommended is not None:
        print("-" * 132)
        print(
            "Recommended config: "
            f"{recommended['generations']} generations, pop {recommended['population_size']} "
            f"(report policy {recommended['report_member_policy']}, "
            f"avg report weighted tardiness {recommended['avg_report_weighted_tardiness']:.1f}, "
            f"avg report makespan {recommended['avg_report_makespan']:.1f})"
        )
    print("=" * 132)


def run_budget_sweep(
    benchmark_dir: str,
    output_path: str,
    generations: Sequence[int],
    population_sizes: Sequence[int],
    warm_start: bool,
    seed: int,
    constraint_handling: str,
    report_member_policy: str,
    crossover_policy: str = NSGA3_DEFAULT_CROSSOVER_POLICY,
    immigrant_policy: str = NSGA3_DEFAULT_IMMIGRANT_POLICY,
    immigrant_count: int = 2,
    immigrant_period: int = 5,
    immigrant_archive_size: int = 8,
    command: str = "",
) -> Dict[str, Any]:
    """Run the NSGA budget sweep and return the serialized payload."""
    import glob

    print("=" * 60)
    print("NSGA-III Budget Sweep")
    print("=" * 60)

    pattern = os.path.join(benchmark_dir, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No benchmark files found in {benchmark_dir}")

    print(f"Found {len(files)} benchmark files")
    instance_results: List[Dict[str, Any]] = []

    for filepath in files:
        instance_name = os.path.basename(filepath).replace(".json", "")
        instance = load_benchmark(filepath)
        baseline = _build_greedy_baseline(instance)
        print(
            f"\nInstance: {instance_name} | baseline={baseline['best_method']} "
            f"({baseline['best_makespan']:.1f})"
        )

        runs = []
        for population_size in population_sizes:
            for n_generations in generations:
                print(
                    f"  NSGA-III gen={n_generations}, pop={population_size}...",
                    end=" ",
                    flush=True,
                )
                result = run_nsga3_experiment(
                    instance,
                    n_generations=n_generations,
                    population_size=population_size,
                    warm_start=warm_start,
                    seed=seed,
                    constraint_handling=constraint_handling,
                    crossover_policy=crossover_policy,
                    sequence_mutation=NSGA3_DEFAULT_SEQUENCE_MUTATION,
                    immigrant_policy=immigrant_policy,
                    immigrant_count=immigrant_count,
                    immigrant_period=immigrant_period,
                    immigrant_archive_size=immigrant_archive_size,
                    report_member_policy=report_member_policy,
                )
                normalized = _normalize_budget_run(
                    result,
                    generations=n_generations,
                    population_size=population_size,
                    best_greedy_makespan=baseline["best_makespan"],
                )
                runs.append(normalized)
                report_makespan = normalized["report_makespan"]
                report_tardy_jobs = normalized["report_n_tardy_jobs"]
                print(
                    f"report={normalized['report_member_key']}, "
                    f"makespan={report_makespan:.1f}, tardy_jobs={report_tardy_jobs}, "
                    f"time={normalized['time_seconds']:.2f}s"
                )

        instance_results.append(
            {
                "instance": instance_name,
                "best_published_greedy_method": baseline["best_method"],
                "best_published_greedy_makespan": baseline["best_makespan"],
                "published_greedy_runs": baseline["runs"],
                "runs": runs,
            }
        )

    summary = _summarize_budget_sweep(instance_results)
    recommended = _recommend_budget(summary)
    payload = {
        "provenance": _build_sweep_provenance(
            benchmark_dir=benchmark_dir,
            output_path=output_path,
            generations=generations,
            population_sizes=population_sizes,
            warm_start=warm_start,
            seed=seed,
            constraint_handling=constraint_handling,
            crossover_policy=crossover_policy,
            sequence_mutation=NSGA3_DEFAULT_SEQUENCE_MUTATION,
            immigrant_policy=immigrant_policy,
            immigrant_count=immigrant_count,
            immigrant_period=immigrant_period,
            immigrant_archive_size=immigrant_archive_size,
            report_member_policy=report_member_policy,
            command=command,
        ),
        "results": instance_results,
        "summary": summary,
        "recommended_budget": recommended,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)

    print(f"\nResults saved to {output_path}")
    print_summary(summary, recommended)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sweep NSGA-III generations and population size")
    parser.add_argument("--benchmark-dir", type=str, default="benchmarks/small")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/nsga_budget_sweep.json",
    )
    parser.add_argument(
        "--generations",
        type=str,
        default="30,60,120",
        help="Comma-separated generation counts",
    )
    parser.add_argument(
        "--population-sizes",
        type=str,
        default="30,60",
        help="Comma-separated population sizes",
    )
    parser.add_argument("--seed", type=int, default=42, help="NSGA-III RNG seed")
    parser.add_argument(
        "--nsga-constraint-handling",
        type=str,
        default="legacy_scalar_penalized_objectives",
        help="Constraint handling policy used inside NSGA-III survival ranking",
    )
    parser.add_argument(
        "--nsga-report-member-policy",
        type=str,
        default="best_makespan_feasible",
        help="Representative feasible Pareto member to aggregate as the report member",
    )
    parser.add_argument(
        "--nsga-crossover-policy",
        type=str,
        default=NSGA3_DEFAULT_CROSSOVER_POLICY,
        help="Sequence crossover policy for the swept NSGA runs",
    )
    parser.add_argument(
        "--nsga-immigrant-policy",
        type=str,
        default=NSGA3_DEFAULT_IMMIGRANT_POLICY,
        help="Optional immigrant archive policy for the swept NSGA runs",
    )
    parser.add_argument(
        "--nsga-immigrant-count",
        type=int,
        default=2,
        help="Maximum number of immigrant archive clones injected per event",
    )
    parser.add_argument(
        "--nsga-immigrant-period",
        type=int,
        default=5,
        help="Inject archive immigrants every N generations when enabled",
    )
    parser.add_argument(
        "--nsga-immigrant-archive-size",
        type=int,
        default=8,
        help="Maximum number of hard-feasible tardiness archive entries to retain",
    )
    parser.add_argument(
        "--no-nsga-warm-start",
        action="store_true",
        help="Disable deterministic greedy warm-start seeds",
    )

    args = parser.parse_args()

    command = "python -m experiments.sweep_nsga_budget"
    if len(sys.argv) > 1:
        command = f"{command} {subprocess.list2cmdline(sys.argv[1:])}"

    run_budget_sweep(
        benchmark_dir=args.benchmark_dir,
        output_path=args.output,
        generations=_parse_csv_ints(args.generations),
        population_sizes=_parse_csv_ints(args.population_sizes),
        warm_start=not args.no_nsga_warm_start,
        seed=args.seed,
        constraint_handling=args.nsga_constraint_handling,
        report_member_policy=args.nsga_report_member_policy,
        crossover_policy=args.nsga_crossover_policy,
        immigrant_policy=args.nsga_immigrant_policy,
        immigrant_count=args.nsga_immigrant_count,
        immigrant_period=args.nsga_immigrant_period,
        immigrant_archive_size=args.nsga_immigrant_archive_size,
        command=command,
    )
