#!/usr/bin/env python
"""
Replay NSGA ranking policies on populations produced by the stable legacy run.

This isolates ranking-policy effects from search dynamics by keeping the
evolution path fixed and only re-sorting already evaluated populations.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Sequence

try:
    from ..experiments.artifact_schemas import POLICY_ANALYSIS_ARTIFACT_SCHEMA
    from ..experiments.compare_solvers import (
        _get_git_status_short,
        _details_like_payload,
        _summarize_feasible_member_pool,
        load_benchmark,
    )
    from ..moea.nsga3 import (
        Individual,
        NSGA3,
        NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        NSGA3_DEFAULT_CROSSOVER_POLICY,
        NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        NSGA3_DEFAULT_SEQUENCE_MUTATION,
        NSGA3_DEFAULT_IMMIGRANT_POLICY,
        Population,
        create_sfjssp_genome,
        create_sfjssp_seed_genomes,
        evaluate_sfjssp_genome,
        evaluate_sfjssp_genome_detailed,
    )
except ImportError:  # pragma: no cover - supports repo-root imports
    from experiments.artifact_schemas import POLICY_ANALYSIS_ARTIFACT_SCHEMA
    from experiments.compare_solvers import (
        _get_git_status_short,
        _details_like_payload,
        _summarize_feasible_member_pool,
        load_benchmark,
    )
    from moea.nsga3 import (
        Individual,
        NSGA3,
        NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        NSGA3_DEFAULT_CROSSOVER_POLICY,
        NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        NSGA3_DEFAULT_SEQUENCE_MUTATION,
        NSGA3_DEFAULT_IMMIGRANT_POLICY,
        Population,
        create_sfjssp_genome,
        create_sfjssp_seed_genomes,
        evaluate_sfjssp_genome,
        evaluate_sfjssp_genome_detailed,
    )


ANALYSIS_POLICIES: Sequence[str] = (
    "legacy_scalar_penalized_objectives",
    "hard_feasible_first_soft_penalties",
    "feasibility_first_constrained_domination",
    "feasibility_first_lexicographic",
)


def _get_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _record_to_individual(record: Dict[str, Any], policy: str) -> Individual:
    metrics = dict(record.get("metrics") or {})
    raw_objectives = list(record.get("raw_objectives") or [])
    penalized_objectives = list(record.get("penalized_objectives") or raw_objectives)
    objectives = (
        list(penalized_objectives)
        if policy in {"legacy_scalar_penalized_objectives", "hard_feasible_first_soft_penalties"}
        else list(raw_objectives)
    )
    return Individual(
        genome={},
        objectives=objectives,
        raw_objectives=raw_objectives,
        penalized_objectives=penalized_objectives,
        penalties=dict(record.get("penalties") or {}),
        metrics=metrics,
        constraint_violations=list(record.get("constraint_violations") or []),
        constraint_key=tuple(record.get("constraint_key") or (0.0, 0.0, 0.0, 0.0)),
        is_feasible=bool(record.get("is_feasible", True)),
        makespan=float(record.get("makespan", 0.0) or 0.0),
        energy=float(metrics.get("total_energy", record.get("energy", 0.0)) or 0.0),
        ergonomic_risk=float(
            metrics.get("max_ergonomic_exposure", record.get("ergonomic_risk", 0.0)) or 0.0
        ),
        labor_cost=float(metrics.get("total_labor_cost", record.get("labor_cost", 0.0)) or 0.0),
    )


def _build_policy_analysis_provenance(
    benchmark_dir: str,
    output_path: str,
    generations: int,
    population_size: int,
    seed: int,
    warm_start: bool,
    report_member_policy: str,
    command: str = "",
) -> Dict[str, Any]:
    """Build artifact provenance for the policy analysis output."""
    git_status_short = _get_git_status_short()
    return {
        "artifact_schema": POLICY_ANALYSIS_ARTIFACT_SCHEMA,
        "timestamp": datetime.now().isoformat(),
        "git_commit": _get_git_commit(),
        "git_dirty": bool(git_status_short),
        "git_status_short": git_status_short,
        "command": command or "python -m experiments.analyze_nsga_policies",
        "benchmark_dir": benchmark_dir,
        "output_path": output_path,
        "nsga3_generations": generations,
        "nsga3_population_size": population_size,
        "nsga3_seed": seed,
        "nsga3_warm_start": warm_start,
        "baseline_constraint_handling": NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        "analysis_policies": list(ANALYSIS_POLICIES),
        "nsga3_report_member_policy": report_member_policy,
        "python_version": sys.version.split()[0],
    }


def _replay_snapshot(
    population_records: Sequence[Dict[str, Any]],
    policy: str,
    report_member_policy: str,
) -> Dict[str, Any]:
    replay = NSGA3(
        n_objectives=4,
        population_size=len(population_records),
        n_generations=0,
        seed=42,
        constraint_handling=policy,
        crossover_policy=NSGA3_DEFAULT_CROSSOVER_POLICY,
        local_improvement=NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        sequence_mutation=NSGA3_DEFAULT_SEQUENCE_MUTATION,
        immigrant_policy=NSGA3_DEFAULT_IMMIGRANT_POLICY,
    )
    population = Population(size=len(population_records))
    for record in population_records:
        population.add(_record_to_individual(record, policy))

    fronts = replay._non_dominated_sort(population)
    front_zero = fronts[0] if fronts else []
    front_details = [
        _details_like_payload(population[idx].__dict__)
        for idx in front_zero
    ]
    summary = _summarize_feasible_member_pool(
        front_details,
        report_member_policy=report_member_policy,
    )
    return {
        "constraint_handling": policy,
        "population_size": len(population_records),
        "pareto_size": len(front_zero),
        "hard_feasible_count": sum(
            1 for record in population_records
            if float((record.get("constraint_key") or [0.0])[0]) <= 0.0
        ),
        "report_member_key": summary.get("report_member_key"),
        "report_makespan": summary.get("report_makespan"),
        "report_weighted_tardiness": summary.get("report_weighted_tardiness"),
        "report_n_tardy_jobs": summary.get("report_n_tardy_jobs"),
        "tardiness_best_makespan": summary.get("tardiness_best_makespan"),
        "tardiness_best_weighted_tardiness": summary.get("tardiness_best_weighted_tardiness"),
        "tardiness_best_n_tardy_jobs": summary.get("tardiness_best_n_tardy_jobs"),
        "feasible_pareto_size": summary.get("feasible_pareto_size"),
        "zero_tardy_feasible_pareto_size": summary.get("zero_tardy_feasible_pareto_size"),
    }


def _average(values: Sequence[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def run_policy_analysis(
    benchmark_dir: str,
    output_path: str,
    generations: int,
    population_size: int,
    seed: int,
    warm_start: bool,
    report_member_policy: str,
    command: str = "",
) -> Dict[str, Any]:
    files = sorted(
        os.path.join(benchmark_dir, name)
        for name in os.listdir(benchmark_dir)
        if name.endswith(".json")
    )
    results: List[Dict[str, Any]] = []

    for filepath in files:
        instance = load_benchmark(filepath)
        seed_genomes = create_sfjssp_seed_genomes(instance) if warm_start else []
        optimizer = NSGA3(
            n_objectives=4,
            population_size=population_size,
            n_generations=generations,
            mutation_rate=0.2,
            crossover_rate=0.9,
            seed=seed,
            constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
            crossover_policy=NSGA3_DEFAULT_CROSSOVER_POLICY,
            local_improvement=NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
            sequence_mutation=NSGA3_DEFAULT_SEQUENCE_MUTATION,
            immigrant_policy=NSGA3_DEFAULT_IMMIGRANT_POLICY,
        )
        optimizer.set_problem(
            evaluate_fn=evaluate_sfjssp_genome,
            evaluate_details_fn=evaluate_sfjssp_genome_detailed,
            create_individual_fn=create_sfjssp_genome,
            seed_individuals_fn=(lambda _instance, seeds=seed_genomes: seeds),
        )
        optimizer.evolve(
            instance,
            verbose=False,
            record_history=True,
            include_population_records=True,
        )
        final_snapshot = optimizer.history[-1]
        generation_replays = []
        for snapshot in optimizer.history:
            generation_replays.append(
                {
                    "generation": snapshot["generation"],
                    "policies": {
                        policy: _replay_snapshot(
                            snapshot.get("population_records") or [],
                            policy=policy,
                            report_member_policy=report_member_policy,
                        )
                        for policy in ANALYSIS_POLICIES
                    },
                }
            )
        results.append(
            {
                "instance": os.path.basename(filepath).replace(".json", ""),
                "legacy_constraint_handling": NSGA3_DEFAULT_CONSTRAINT_HANDLING,
                "generations": generations,
                "population_size": population_size,
                "final_generation": final_snapshot["generation"],
                "final_population_replay": {
                    policy: _replay_snapshot(
                        final_snapshot.get("population_records") or [],
                        policy=policy,
                        report_member_policy=report_member_policy,
                    )
                    for policy in ANALYSIS_POLICIES
                },
                "generation_replay": generation_replays,
            }
        )

    summary: List[Dict[str, Any]] = []
    for policy in ANALYSIS_POLICIES:
        final_runs = [item["final_population_replay"][policy] for item in results]
        summary.append(
            {
                "constraint_handling": policy,
                "instances": len(final_runs),
                "avg_report_makespan": _average(
                    [run.get("report_makespan") for run in final_runs]
                ),
                "avg_report_weighted_tardiness": _average(
                    [run.get("report_weighted_tardiness") for run in final_runs]
                ),
                "avg_report_n_tardy_jobs": _average(
                    [run.get("report_n_tardy_jobs") for run in final_runs]
                ),
                "instances_with_zero_tardy_feasible_member": sum(
                    1
                    for run in final_runs
                    if (run.get("zero_tardy_feasible_pareto_size") or 0) > 0
                ),
            }
        )

    return {
        "provenance": _build_policy_analysis_provenance(
            benchmark_dir=benchmark_dir,
            output_path=output_path,
            generations=generations,
            population_size=population_size,
            seed=seed,
            warm_start=warm_start,
            report_member_policy=report_member_policy,
            command=command,
        ),
        "results": results,
        "summary": summary,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Replay NSGA policy rankings on stored populations")
    parser.add_argument("--benchmark-dir", type=str, default="benchmarks/small")
    parser.add_argument("--output", type=str, default="experiments/results/nsga_policy_analysis.json")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nsga-report-member-policy",
        type=str,
        default="best_makespan_feasible",
    )
    parser.add_argument("--no-nsga-warm-start", action="store_true")
    args = parser.parse_args()

    payload = run_policy_analysis(
        benchmark_dir=args.benchmark_dir,
        output_path=args.output,
        generations=args.generations,
        population_size=args.population_size,
        seed=args.seed,
        warm_start=not args.no_nsga_warm_start,
        report_member_policy=args.nsga_report_member_policy,
        command=" ".join(sys.argv),
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved: {args.output}")
    for row in payload["summary"]:
        print(
            f"{row['constraint_handling']}: "
            f"avg_mk={row['avg_report_makespan']}, "
            f"avg_wtd={row['avg_report_weighted_tardiness']}, "
            f"avg_tardy={row['avg_report_n_tardy_jobs']}"
        )


if __name__ == "__main__":
    main()
