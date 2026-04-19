#!/usr/bin/env python
"""
Comparative experiments for SFJSSP solvers.

This script serializes solver outputs using the canonical schedule metric names
and an explicit report-member contract for NSGA artifacts.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

try:
    from ..experiments.artifact_schemas import (
        COMPARISON_ARTIFACT_SCHEMA,
        CANONICAL_SCHEDULE_METRIC_FIELDS,
        CANONICAL_SCHEDULE_PENALTY_FIELDS,
        derive_canonical_feasibility,
    )
except ImportError:  # pragma: no cover - supports repo-root imports
    from experiments.artifact_schemas import (
        COMPARISON_ARTIFACT_SCHEMA,
        CANONICAL_SCHEDULE_METRIC_FIELDS,
        CANONICAL_SCHEDULE_PENALTY_FIELDS,
        derive_canonical_feasibility,
    )

try:
    from ..sfjssp_model.instance import SFJSSPInstance
    from ..baseline_solver.greedy_solvers import (
        GreedyScheduler,
        spt_rule,
        fifo_rule,
        edt_rule,
        composite_rule,
    )
    from ..moea.nsga3 import (
        NSGA3,
        NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        NSGA3_CONSTRAINT_HANDLING_POLICIES,
        NSGA3_DEFAULT_CROSSOVER_POLICY,
        NSGA3_CROSSOVER_POLICIES,
        NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        NSGA3_LOCAL_IMPROVEMENT_POLICIES,
        NSGA3_DEFAULT_SEQUENCE_MUTATION,
        NSGA3_SEQUENCE_MUTATION_POLICIES,
        NSGA3_DEFAULT_IMMIGRANT_POLICY,
        NSGA3_IMMIGRANT_POLICIES,
        create_sfjssp_genome,
        create_sfjssp_seed_genomes,
        evaluate_sfjssp_genome,
        evaluate_sfjssp_genome_detailed,
    )
except ImportError:  # pragma: no cover - supports repo-root imports
    from sfjssp_model.instance import SFJSSPInstance
    from baseline_solver.greedy_solvers import (
        GreedyScheduler,
        spt_rule,
        fifo_rule,
        edt_rule,
        composite_rule,
    )
    from moea.nsga3 import (
        NSGA3,
        NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        NSGA3_CONSTRAINT_HANDLING_POLICIES,
        NSGA3_DEFAULT_CROSSOVER_POLICY,
        NSGA3_CROSSOVER_POLICIES,
        NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        NSGA3_LOCAL_IMPROVEMENT_POLICIES,
        NSGA3_DEFAULT_SEQUENCE_MUTATION,
        NSGA3_SEQUENCE_MUTATION_POLICIES,
        NSGA3_DEFAULT_IMMIGRANT_POLICY,
        NSGA3_IMMIGRANT_POLICIES,
        create_sfjssp_genome,
        create_sfjssp_seed_genomes,
        evaluate_sfjssp_genome,
        evaluate_sfjssp_genome_detailed,
    )


NSGA3_REPORT_MEMBER_POLICIES: Sequence[str] = (
    "best_makespan_feasible",
    "min_n_tardy_feasible",
    "min_weighted_tardiness_feasible",
)

# This is the only public projection from canonical Schedule.evaluate() metrics
# into persisted experiment artifacts. Public artifact names must stay identical
# to the canonical schedule oracle names listed here.
ARTIFACT_CANONICAL_METRIC_MAP: Dict[str, str] = {
    metric_name: metric_name
    for metric_name in CANONICAL_SCHEDULE_METRIC_FIELDS
}

ARTIFACT_CANONICAL_PENALTY_MAP: Dict[str, str] = {
    penalty_name: penalty_name
    for penalty_name in CANONICAL_SCHEDULE_PENALTY_FIELDS
}


def load_benchmark(filepath: str) -> SFJSSPInstance:
    """Load a benchmark instance from its stored JSON representation."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SFJSSPInstance.from_dict(data)


def _clone_instance(instance: SFJSSPInstance) -> SFJSSPInstance:
    """Create a fresh instance copy so each solver runs on clean mutable state."""
    return SFJSSPInstance.from_dict(instance.to_dict())


def _get_git_commit() -> Optional[str]:
    """Return the current Git commit hash when available."""
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


def _get_git_status_short() -> List[str]:
    """Return short git status lines for local, uncommitted changes."""
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]


def _build_provenance(
    benchmark_dir: str,
    output_path: str,
    run_cp: bool,
    nsga3_generations: int,
    nsga3_population_size: int,
    nsga3_warm_start: bool,
    nsga3_seed: int,
    nsga3_constraint_handling: str,
    nsga3_parent_selection: str,
    nsga3_crossover_policy: str,
    nsga3_local_improvement: str,
    nsga3_sequence_mutation: str,
    nsga3_immigrant_policy: str,
    nsga3_immigrant_count: int,
    nsga3_immigrant_period: int,
    nsga3_immigrant_archive_size: int,
    nsga3_report_member_policy: str,
    command: str = "",
) -> Dict[str, Any]:
    """Build provenance metadata for a comparison artifact."""
    git_status_short = _get_git_status_short()
    return {
        "artifact_schema": COMPARISON_ARTIFACT_SCHEMA,
        "git_commit": _get_git_commit(),
        "git_dirty": bool(git_status_short),
        "git_status_short": git_status_short,
        "command": command or "python -m experiments.compare_solvers",
        "benchmark_dir": benchmark_dir,
        "output_path": output_path,
        "nsga3_generations": nsga3_generations,
        "nsga3_population_size": nsga3_population_size,
        "nsga3_warm_start": nsga3_warm_start,
        "nsga3_seed_source": "greedy dispatch rules" if nsga3_warm_start else "disabled",
        "nsga3_seed": nsga3_seed,
        "nsga3_constraint_handling": nsga3_constraint_handling,
        "nsga3_parent_selection": nsga3_parent_selection,
        "nsga3_crossover_policy": nsga3_crossover_policy,
        "nsga3_local_improvement": nsga3_local_improvement,
        "nsga3_sequence_mutation": nsga3_sequence_mutation,
        "nsga3_immigrant_policy": nsga3_immigrant_policy,
        "nsga3_immigrant_count": nsga3_immigrant_count,
        "nsga3_immigrant_period": nsga3_immigrant_period,
        "nsga3_immigrant_archive_size": nsga3_immigrant_archive_size,
        "nsga3_report_member_policy": nsga3_report_member_policy,
        "cp_enabled": run_cp,
        "cp_verified_scope": "makespan only",
        "python_version": sys.version.split()[0],
        "timestamp": datetime.now().isoformat(),
    }


def _validate_report_member_policy(report_member_policy: str) -> str:
    """Validate the requested NSGA report-member policy."""
    if report_member_policy not in NSGA3_REPORT_MEMBER_POLICIES:
        raise ValueError(
            "Unsupported NSGA report-member policy "
            f"{report_member_policy!r}; expected one of {list(NSGA3_REPORT_MEMBER_POLICIES)}"
        )
    return report_member_policy


def _resolve_report_member(
    representative_members: Dict[str, Optional[Dict[str, Any]]],
    report_member_policy: str,
) -> Optional[Dict[str, Any]]:
    """Return the representative member selected for artifact/report publication."""
    _validate_report_member_policy(report_member_policy)
    return representative_members.get(report_member_policy)


def _details_like_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a stored population record into the details-like shape used downstream."""
    return {
        "metrics": dict(payload.get("metrics") or {}),
        "penalties": dict(payload.get("penalties") or {}),
        "constraint_violations": list(payload.get("constraint_violations") or []),
        "is_feasible": bool(payload.get("is_feasible", True)),
    }


def _details_are_canonically_feasible(details: Dict[str, Any]) -> bool:
    """Return full feasibility using explicit penalties and violation evidence."""
    return derive_canonical_feasibility(
        details.get("is_feasible", True),
        details.get("penalties") or {},
        details.get("constraint_violations") or [],
    )


def _project_canonical_metric_payload(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Project canonical schedule metrics into the persisted artifact surface."""
    return {
        artifact_field: metrics.get(schedule_field)
        for schedule_field, artifact_field in ARTIFACT_CANONICAL_METRIC_MAP.items()
    }


def _project_canonical_penalty_payload(penalties: Dict[str, Any]) -> Dict[str, Any]:
    """Project canonical penalties into the persisted artifact surface."""
    return {
        artifact_field: penalties.get(schedule_field)
        for schedule_field, artifact_field in ARTIFACT_CANONICAL_PENALTY_MAP.items()
    }


def _build_representative_member(details: Dict[str, Any], policy: str) -> Dict[str, Any]:
    """Project one detailed-evaluation payload into artifact-ready representative fields."""
    metrics = details.get('metrics', {})
    penalties = details.get('penalties', {})
    constraint_violations = list(details.get('constraint_violations', []))
    is_feasible = derive_canonical_feasibility(
        details.get('is_feasible', True),
        penalties,
        constraint_violations,
    )
    return {
        'policy': policy,
        'is_feasible': is_feasible,
        'metrics': _project_canonical_metric_payload(metrics),
        'penalties': _project_canonical_penalty_payload(penalties),
        'constraint_violations': constraint_violations,
    }


def _build_representative_members(
    feasible_members: Sequence[Dict[str, Any]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Select the named feasible NSGA representative members from a detail payload list."""
    if not feasible_members:
        return {
            'best_makespan_feasible': None,
            'min_n_tardy_feasible': None,
            'min_weighted_tardiness_feasible': None,
        }

    best_makespan_member = min(
        feasible_members,
        key=lambda details: details['metrics'].get('makespan', float('inf')),
    )
    min_n_tardy_member = min(
        feasible_members,
        key=lambda details: (
            details['metrics'].get('n_tardy_jobs', float('inf')),
            details['metrics'].get('weighted_tardiness', float('inf')),
            details['metrics'].get('makespan', float('inf')),
        ),
    )
    min_weighted_tardiness_member = min(
        feasible_members,
        key=lambda details: (
            details['metrics'].get('weighted_tardiness', float('inf')),
            details['metrics'].get('n_tardy_jobs', float('inf')),
            details['metrics'].get('makespan', float('inf')),
        ),
    )
    return {
        'best_makespan_feasible': _build_representative_member(
            best_makespan_member,
            'best_makespan_feasible_pareto',
        ),
        'min_n_tardy_feasible': _build_representative_member(
            min_n_tardy_member,
            'min_n_tardy_feasible_pareto',
        ),
        'min_weighted_tardiness_feasible': _build_representative_member(
            min_weighted_tardiness_member,
            'min_weighted_tardiness_feasible_pareto',
        ),
    }


def _summarize_feasible_member_pool(
    detailed_members: Sequence[Dict[str, Any]],
    report_member_policy: str,
) -> Dict[str, Any]:
    """Summarize one evaluated feasible-member pool using the artifact report policy."""
    feasible_members = [
        details for details in detailed_members
        if _details_are_canonically_feasible(details)
    ]
    representative_members = _build_representative_members(feasible_members)
    summary: Dict[str, Any] = {
        'representative_members': representative_members,
        'feasible_pareto_size': len(feasible_members),
        'zero_tardy_feasible_pareto_size': sum(
            1
            for details in feasible_members
            if details['metrics'].get('n_tardy_jobs', 0) == 0
        ),
        'min_n_tardy_jobs': (
            min(
                details['metrics'].get('n_tardy_jobs', float('inf'))
                for details in feasible_members
            )
            if feasible_members
            else None
        ),
        'min_weighted_tardiness': (
            min(
                details['metrics'].get('weighted_tardiness', float('inf'))
                for details in feasible_members
            )
            if feasible_members
            else None
        ),
        'makespan': representative_members['best_makespan_feasible']['metrics']['makespan']
        if representative_members['best_makespan_feasible'] is not None
        else None,
        'total_energy': representative_members['best_makespan_feasible']['metrics']['total_energy']
        if representative_members['best_makespan_feasible'] is not None
        else None,
        'max_ergonomic_exposure': representative_members['best_makespan_feasible']['metrics']['max_ergonomic_exposure']
        if representative_members['best_makespan_feasible'] is not None
        else None,
        'total_labor_cost': representative_members['best_makespan_feasible']['metrics']['total_labor_cost']
        if representative_members['best_makespan_feasible'] is not None
        else None,
    }
    _apply_report_member_policy(
        summary,
        representative_members=representative_members,
        report_member_policy=report_member_policy,
    )
    return summary


def _summarize_generation_diagnostics(
    history: Sequence[Dict[str, Any]],
    report_member_policy: str,
) -> List[Dict[str, Any]]:
    """Collapse replayable generation snapshots into report-member diagnostics."""
    diagnostics: List[Dict[str, Any]] = []
    for snapshot in history:
        population_records = snapshot.get("population_records") or []
        detailed_members = [_details_like_payload(record) for record in population_records]
        summary = _summarize_feasible_member_pool(
            detailed_members,
            report_member_policy=report_member_policy,
        )
        diagnostics.append(
            {
                "generation": snapshot.get("generation"),
                "constraint_handling": snapshot.get("constraint_handling"),
                "crossover_policy": snapshot.get("crossover_policy", NSGA3_DEFAULT_CROSSOVER_POLICY),
                "local_improvement": snapshot.get("local_improvement", NSGA3_DEFAULT_LOCAL_IMPROVEMENT),
                "sequence_mutation": snapshot.get("sequence_mutation", NSGA3_DEFAULT_SEQUENCE_MUTATION),
                "immigrant_policy": snapshot.get("immigrant_policy", NSGA3_DEFAULT_IMMIGRANT_POLICY),
                "population_size": snapshot.get("population_size"),
                "hard_feasible_count": snapshot.get("hard_feasible_count"),
                "min_hard_feasible_makespan": snapshot.get("min_hard_feasible_makespan"),
                "min_hard_feasible_weighted_tardiness": snapshot.get("min_hard_feasible_weighted_tardiness"),
                "zero_tardy_hard_feasible_count": snapshot.get("zero_tardy_hard_feasible_count"),
                "report_member_key": summary.get("report_member_key"),
                "report_member_selection_policy": summary.get("report_member_selection_policy"),
                "report_member_is_feasible": summary.get("report_member_is_feasible"),
                "report_makespan": summary.get("report_makespan"),
                "report_weighted_tardiness": summary.get("report_weighted_tardiness"),
                "report_n_tardy_jobs": summary.get("report_n_tardy_jobs"),
                "tardiness_best_makespan": summary.get("tardiness_best_makespan"),
                "tardiness_best_weighted_tardiness": summary.get("tardiness_best_weighted_tardiness"),
                "tardiness_best_n_tardy_jobs": summary.get("tardiness_best_n_tardy_jobs"),
                "repair_attempted_children": snapshot.get("repair_attempted_children", 0),
                "repair_accepted_children": snapshot.get("repair_accepted_children", 0),
                "repair_improved_n_tardy_jobs_count": snapshot.get("repair_improved_n_tardy_jobs_count", 0),
                "repair_improved_weighted_tardiness_count": snapshot.get("repair_improved_weighted_tardiness_count", 0),
                "repair_rejected_due_to_makespan_cap_count": snapshot.get("repair_rejected_due_to_makespan_cap_count", 0),
                "urgent_sequence_mutation_attempts": snapshot.get("urgent_sequence_mutation_attempts", 0),
                "urgent_sequence_mutation_applied": snapshot.get("urgent_sequence_mutation_applied", 0),
                "urgent_sequence_mutation_fallback_random_swap_count": snapshot.get(
                    "urgent_sequence_mutation_fallback_random_swap_count",
                    0,
                ),
                "urgent_sequence_mutation_noop_count": snapshot.get("urgent_sequence_mutation_noop_count", 0),
                "urgent_sequence_mutation_changed_position_distance_sum": snapshot.get(
                    "urgent_sequence_mutation_changed_position_distance_sum",
                    0,
                ),
                "urgent_crossover_attempts": snapshot.get("urgent_crossover_attempts", 0),
                "urgent_crossover_applied": snapshot.get("urgent_crossover_applied", 0),
                "urgent_crossover_fallback_legacy_count": snapshot.get(
                    "urgent_crossover_fallback_legacy_count",
                    0,
                ),
                "urgent_crossover_prefix_total": snapshot.get("urgent_crossover_prefix_total", 0),
                "urgent_crossover_children_from_tardiness_better_parent_count": snapshot.get(
                    "urgent_crossover_children_from_tardiness_better_parent_count",
                    0,
                ),
                "children_evaluated_count": snapshot.get("children_evaluated_count", 0),
                "children_hard_feasible_count": snapshot.get("children_hard_feasible_count", 0),
                "children_zero_tardy_count": snapshot.get("children_zero_tardy_count", 0),
                "children_improve_both_parents_n_tardy_jobs_count": snapshot.get(
                    "children_improve_both_parents_n_tardy_jobs_count",
                    0,
                ),
                "children_improve_both_parents_weighted_tardiness_count": snapshot.get(
                    "children_improve_both_parents_weighted_tardiness_count",
                    0,
                ),
                "children_improve_both_parents_makespan_count": snapshot.get(
                    "children_improve_both_parents_makespan_count",
                    0,
                ),
                "best_child_weighted_tardiness": snapshot.get("best_child_weighted_tardiness"),
                "best_child_generation": snapshot.get("best_child_generation"),
                "best_child_n_tardy_jobs": snapshot.get("best_child_n_tardy_jobs"),
                "best_child_makespan": snapshot.get("best_child_makespan"),
                "immigrant_archive_size": snapshot.get("immigrant_archive_size", 0),
                "immigrant_archive_admission_count": snapshot.get("immigrant_archive_admission_count", 0),
                "immigrant_archive_replaced_count": snapshot.get("immigrant_archive_replaced_count", 0),
                "immigrant_injection_events": snapshot.get("immigrant_injection_events", 0),
                "immigrant_injected_individuals": snapshot.get("immigrant_injected_individuals", 0),
                "immigrant_skipped_duplicate_count": snapshot.get("immigrant_skipped_duplicate_count", 0),
                "immigrant_survivors_in_next_population": snapshot.get(
                    "immigrant_survivors_in_next_population",
                    0,
                ),
            }
        )
    return diagnostics


def _is_same_member(
    left: Optional[Dict[str, Any]],
    right: Optional[Dict[str, Any]],
) -> bool:
    """Check whether two representative-member payloads describe the same solution."""
    if left is None or right is None:
        return False
    return (
        left.get("metrics") == right.get("metrics")
        and left.get("penalties") == right.get("penalties")
        and left.get("constraint_violations") == right.get("constraint_violations")
    )


def _apply_report_member_policy(
    result: Dict[str, Any],
    representative_members: Dict[str, Optional[Dict[str, Any]]],
    report_member_policy: str,
) -> Dict[str, Any]:
    """Attach the explicit public report-member contract."""
    report_member_key = _validate_report_member_policy(report_member_policy)
    report_member = _resolve_report_member(representative_members, report_member_key)
    tardiness_member = representative_members.get("min_weighted_tardiness_feasible")

    result["report_member_key"] = report_member_key
    if report_member is not None:
        report_metrics = report_member["metrics"]
        report_penalties = report_member["penalties"]
        result.update(
            {
                "report_member_selection_policy": report_member["policy"],
                "report_member_is_feasible": report_member["is_feasible"],
                "report_member_metrics": report_metrics,
                "report_member_penalties": report_penalties,
                "report_member_constraint_violations": report_member["constraint_violations"],
                "report_makespan": report_metrics.get("makespan"),
                "report_total_energy": report_metrics.get("total_energy"),
                "report_max_ergonomic_exposure": report_metrics.get("max_ergonomic_exposure"),
                "report_total_labor_cost": report_metrics.get("total_labor_cost"),
                "report_total_tardiness": report_metrics.get("total_tardiness"),
                "report_n_tardy_jobs": report_metrics.get("n_tardy_jobs"),
                "report_weighted_tardiness": report_metrics.get("weighted_tardiness"),
                "report_total_penalty": report_penalties.get("total_penalty"),
                "report_member_zero_tardy": report_metrics.get("n_tardy_jobs") == 0,
                "makespan": report_metrics.get("makespan"),
                "total_energy": report_metrics.get("total_energy"),
                "max_ergonomic_exposure": report_metrics.get("max_ergonomic_exposure"),
                "total_labor_cost": report_metrics.get("total_labor_cost"),
                "total_tardiness": report_metrics.get("total_tardiness"),
                "weighted_tardiness": report_metrics.get("weighted_tardiness"),
                "n_tardy_jobs": report_metrics.get("n_tardy_jobs"),
            }
        )
    else:
        default_policy_name = f"{report_member_key}_pareto"
        result.update(
            {
                "report_member_selection_policy": default_policy_name,
                "report_member_is_feasible": False,
                "report_member_metrics": None,
                "report_member_penalties": None,
                "report_member_constraint_violations": [],
                "report_makespan": None,
                "report_total_energy": None,
                "report_max_ergonomic_exposure": None,
                "report_total_labor_cost": None,
                "report_total_tardiness": None,
                "report_n_tardy_jobs": None,
                "report_weighted_tardiness": None,
                "report_total_penalty": None,
                "report_member_zero_tardy": False,
                "makespan": None,
                "total_energy": None,
                "max_ergonomic_exposure": None,
                "total_labor_cost": None,
                "total_tardiness": None,
                "weighted_tardiness": None,
                "n_tardy_jobs": None,
            }
        )

    if tardiness_member is not None:
        tardiness_metrics = tardiness_member["metrics"]
        tardiness_penalties = tardiness_member["penalties"]
        result.update(
            {
                "tardiness_best_member_policy": tardiness_member["policy"],
                "tardiness_best_member_metrics": tardiness_metrics,
                "tardiness_best_member_penalties": tardiness_penalties,
                "tardiness_best_member_constraint_violations": tardiness_member["constraint_violations"],
                "tardiness_best_n_tardy_jobs": tardiness_metrics.get("n_tardy_jobs"),
                "tardiness_best_weighted_tardiness": tardiness_metrics.get("weighted_tardiness"),
                "tardiness_best_makespan": tardiness_metrics.get("makespan"),
                "tardiness_best_zero_tardy": tardiness_metrics.get("n_tardy_jobs") == 0,
                "tardiness_best_is_same_as_report_member": _is_same_member(tardiness_member, report_member),
            }
        )
    else:
        result.update(
            {
                "tardiness_best_member_policy": "min_weighted_tardiness_feasible_pareto",
                "tardiness_best_member_metrics": None,
                "tardiness_best_member_penalties": None,
                "tardiness_best_member_constraint_violations": [],
                "tardiness_best_n_tardy_jobs": None,
                "tardiness_best_weighted_tardiness": None,
                "tardiness_best_makespan": None,
                "tardiness_best_zero_tardy": False,
                "tardiness_best_is_same_as_report_member": False,
            }
        )

    return result


def run_greedy_experiment(instance: SFJSSPInstance, rule_name: str, rule_fn) -> dict:
    """Run greedy scheduler experiment"""
    instance = _clone_instance(instance)
    start_time = time.time()

    scheduler = GreedyScheduler(job_rule=rule_fn)
    schedule = scheduler.schedule(instance, verbose=True)

    elapsed = time.time() - start_time

    # Evaluate
    objectives = schedule.evaluate(instance)
    
    if not schedule.is_feasible and schedule.constraint_violations:
        print(f"    (Infeasible: {len(schedule.constraint_violations)} violations, first: {schedule.constraint_violations[0]})")
        if any("overlap" in v for v in schedule.constraint_violations):
            print(f"    WARNING: Overlap violation detected!")

    return {
        'method': f'Greedy ({rule_name})',
        'makespan': schedule.makespan,
        'total_energy': objectives.get('total_energy', 0),
        'max_ergonomic_exposure': objectives.get('max_ergonomic_exposure', 0),
        'total_labor_cost': objectives.get('total_labor_cost', 0),
        'total_tardiness': objectives.get('total_tardiness', 0),
        'weighted_tardiness': objectives.get('weighted_tardiness', 0),
        'n_tardy_jobs': objectives.get('n_tardy_jobs', 0),
        'time_seconds': elapsed,
        'feasible': schedule.is_feasible,
        'constraint_violations': list(schedule.constraint_violations),
    }


def run_nsga3_experiment(
    instance: SFJSSPInstance,
    n_generations: int = 50,
    population_size: int = 30,
    warm_start: bool = True,
    seed: int = 42,
    constraint_handling: str = NSGA3_DEFAULT_CONSTRAINT_HANDLING,
    parent_selection: str = "random_pairing",
    crossover_policy: str = NSGA3_DEFAULT_CROSSOVER_POLICY,
    local_improvement: str = NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
    sequence_mutation: str = NSGA3_DEFAULT_SEQUENCE_MUTATION,
    immigrant_policy: str = NSGA3_DEFAULT_IMMIGRANT_POLICY,
    immigrant_count: int = 2,
    immigrant_period: int = 5,
    immigrant_archive_size: int = 8,
    report_member_policy: str = "best_makespan_feasible",
    collect_generation_diagnostics: bool = False,
) -> dict:
    """Run NSGA-III experiment"""
    report_member_policy = _validate_report_member_policy(report_member_policy)
    instance = _clone_instance(instance)
    start_time = time.time()
    seed_genomes = create_sfjssp_seed_genomes(instance) if warm_start else []

    nsga3 = NSGA3(
        n_objectives=4,
        population_size=population_size,
        n_generations=n_generations,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=seed,
        constraint_handling=constraint_handling,
        parent_selection=parent_selection,
        crossover_policy=crossover_policy,
        local_improvement=local_improvement,
        sequence_mutation=sequence_mutation,
        immigrant_policy=immigrant_policy,
        immigrant_count=immigrant_count,
        immigrant_period=immigrant_period,
        immigrant_archive_size=immigrant_archive_size,
    )

    set_problem_kwargs = dict(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
        seed_individuals_fn=(lambda _instance, seeds=seed_genomes: seeds),
    )
    nsga3.set_problem(**set_problem_kwargs)

    nsga3.evolve(
        instance,
        verbose=False,
        record_history=collect_generation_diagnostics,
        include_population_records=collect_generation_diagnostics,
    )

    elapsed = time.time() - start_time

    # Get best solutions for each objective
    pareto = nsga3.get_pareto_solutions()
    detailed_pareto = [
        evaluate_sfjssp_genome_detailed(instance, sol.genome)
        for sol in pareto
    ]
    feasible_pareto = [
        details for details in detailed_pareto
        if _details_are_canonically_feasible(details)
    ]

    feasible_summary = _summarize_feasible_member_pool(
        feasible_pareto,
        report_member_policy=report_member_policy,
    )
    representative_members = feasible_summary['representative_members']
    best_makespan = feasible_summary['makespan']
    best_total_energy = feasible_summary['total_energy']
    best_max_ergonomic_exposure = feasible_summary['max_ergonomic_exposure']
    best_total_labor_cost = feasible_summary['total_labor_cost']
    min_n_tardy_jobs = feasible_summary['min_n_tardy_jobs']
    min_weighted_tardiness = feasible_summary['min_weighted_tardiness']
    zero_tardy_feasible_pareto_size = feasible_summary['zero_tardy_feasible_pareto_size']

    min_penalty = min(
        (details['penalties']['total_penalty'] for details in detailed_pareto),
        default=float('inf'),
    )
    zero_penalty_count = sum(
        1 for details in detailed_pareto
        if details['penalties']['total_penalty'] == 0.0
    )

    result = {
        'method': f'NSGA-III ({n_generations} gen)',
        'makespan': best_makespan,
        'total_energy': best_total_energy,
        'max_ergonomic_exposure': best_max_ergonomic_exposure,
        'total_labor_cost': best_total_labor_cost,
        'total_tardiness': feasible_summary.get('report_total_tardiness'),
        'weighted_tardiness': feasible_summary.get('report_weighted_tardiness'),
        'n_tardy_jobs': feasible_summary.get('report_n_tardy_jobs'),
        'time_seconds': elapsed,
        'population_size': population_size,
        'seed': seed,
        'warm_start': warm_start,
        'warm_start_seed_count': len(seed_genomes),
        'constraint_handling': nsga3.constraint_handling,
        'parent_selection': nsga3.parent_selection,
        'crossover_policy': nsga3.crossover_policy,
        'local_improvement': nsga3.local_improvement,
        'sequence_mutation': nsga3.sequence_mutation,
        'immigrant_policy': nsga3.immigrant_policy,
        'immigrant_count': nsga3.immigrant_count,
        'immigrant_period': nsga3.immigrant_period,
        'immigrant_archive_size': nsga3.immigrant_archive_size,
        'pareto_size': len(pareto),
        'feasible_pareto_size': feasible_summary['feasible_pareto_size'],
        'zero_penalty_pareto_size': zero_penalty_count,
        'zero_tardy_feasible_pareto_size': zero_tardy_feasible_pareto_size,
        'min_n_tardy_jobs': min_n_tardy_jobs,
        'min_weighted_tardiness': min_weighted_tardiness,
        'min_total_penalty': min_penalty,
        'representative_members': representative_members,
        'feasible': bool(feasible_pareto),
    }
    result.update(nsga3.last_run_diagnostics)
    result.update(feasible_summary)

    if collect_generation_diagnostics:
        result['generation_diagnostics'] = _summarize_generation_diagnostics(
            nsga3.history,
            report_member_policy=report_member_policy,
        )

    if not feasible_pareto:
        result['error'] = (
            "No hard-feasible Pareto members found under the current "
            f"NSGA-III configuration ({n_generations} generations)."
        )

    return result


def run_cp_experiment(instance: SFJSSPInstance, time_limit: int = 30) -> dict:
    """Run CP-SAT experiment"""
    instance = _clone_instance(instance)
    try:
        from ..exact_solvers.cp_solver import CPScheduler
    except ImportError:  # pragma: no cover - supports repo-root imports
        try:
            from exact_solvers.cp_solver import CPScheduler
        except ImportError:
            return {
                'method': 'CP-SAT',
                'error': 'OR-Tools not available',
                'makespan': None,
                'total_energy': None,
                'max_ergonomic_exposure': None,
                'total_labor_cost': None,
                'total_tardiness': None,
                'weighted_tardiness': None,
                'n_tardy_jobs': None,
                'time_seconds': None,
                'feasible': False,
                'objective': 'makespan',
                'support_status': 'verified for makespan only',
            }
    except Exception as exc:
        return {
            'method': 'CP-SAT',
            'error': str(exc),
            'makespan': None,
            'total_energy': None,
            'max_ergonomic_exposure': None,
            'total_labor_cost': None,
            'total_tardiness': None,
            'weighted_tardiness': None,
            'n_tardy_jobs': None,
            'time_seconds': None,
            'feasible': False,
            'objective': 'makespan',
            'support_status': 'verified for makespan only',
        }

    start_time = time.time()

    try:
        cp_solver = CPScheduler(time_limit=time_limit, num_workers=2)
        schedule = cp_solver.solve(instance, objective='makespan', verbose=False)
    except ImportError as exc:
        return {
            'method': 'CP-SAT',
            'error': str(exc),
            'makespan': None,
            'total_energy': None,
            'max_ergonomic_exposure': None,
            'total_labor_cost': None,
            'total_tardiness': None,
            'weighted_tardiness': None,
            'n_tardy_jobs': None,
            'feasible': False,
            'objective': 'makespan',
            'support_status': 'verified for makespan only',
            'time_seconds': None,
        }

    elapsed = time.time() - start_time

    if schedule is None:
        return {
            'method': 'CP-SAT',
            'error': 'No solution found',
            'makespan': None,
            'total_energy': None,
            'max_ergonomic_exposure': None,
            'total_labor_cost': None,
            'total_tardiness': None,
            'weighted_tardiness': None,
            'n_tardy_jobs': None,
            'time_seconds': elapsed,
            'feasible': False,
            'objective': 'makespan',
            'support_status': 'verified for makespan only',
        }

    objectives = schedule.evaluate(instance)

    return {
        'method': 'CP-SAT',
        'objective': 'makespan',
        'support_status': 'verified for makespan only',
        'makespan': schedule.makespan,
        'total_energy': objectives.get('total_energy', 0),
        'max_ergonomic_exposure': objectives.get('max_ergonomic_exposure', 0),
        'total_labor_cost': objectives.get('total_labor_cost', 0),
        'total_tardiness': objectives.get('total_tardiness', 0),
        'weighted_tardiness': objectives.get('weighted_tardiness', 0),
        'n_tardy_jobs': objectives.get('n_tardy_jobs', 0),
        'time_seconds': elapsed,
        'feasible': schedule.is_feasible,
    }


def run_comparison(
    instance: SFJSSPInstance,
    instance_name: str,
    run_cp: bool = False,
    nsga3_generations: int = 50,
    nsga3_population_size: int = 30,
    nsga3_warm_start: bool = True,
    nsga3_seed: int = 42,
    nsga3_constraint_handling: str = "legacy_scalar_penalized_objectives",
    nsga3_parent_selection: str = "random_pairing",
    nsga3_crossover_policy: str = NSGA3_DEFAULT_CROSSOVER_POLICY,
    nsga3_local_improvement: str = NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
    nsga3_sequence_mutation: str = NSGA3_DEFAULT_SEQUENCE_MUTATION,
    nsga3_immigrant_policy: str = NSGA3_DEFAULT_IMMIGRANT_POLICY,
    nsga3_immigrant_count: int = 2,
    nsga3_immigrant_period: int = 5,
    nsga3_immigrant_archive_size: int = 8,
    nsga3_report_member_policy: str = "best_makespan_feasible",
    nsga3_collect_generation_diagnostics: bool = False,
) -> dict:
    """Run full comparison on instance"""
    print(f"\n{'='*60}")
    print(f"Instance: {instance_name}")
    print(f"Jobs: {instance.n_jobs}, Machines: {instance.n_machines}, Workers: {instance.n_workers}")
    print(f"{'='*60}")

    results = {
        'instance': instance_name,
        'n_jobs': instance.n_jobs,
        'n_machines': instance.n_machines,
        'n_workers': instance.n_workers,
        'timestamp': datetime.now().isoformat(),
        'experiments': [],
    }

    # Greedy methods
    print("\nRunning greedy heuristics...")

    for rule_name, rule_fn in [('SPT', spt_rule), ('FIFO', fifo_rule), ('EDD', edt_rule)]:
        print(f"  {rule_name}...", end=' ', flush=True)
        result = run_greedy_experiment(instance, rule_name, rule_fn)
        results['experiments'].append(result)
        print(f"makespan={result['makespan']:.1f}, time={result['time_seconds']:.3f}s")

    # NSGA-III
    print(
        f"\nRunning NSGA-III ({nsga3_generations} generations, "
        f"population {nsga3_population_size})..."
    )
    result = run_nsga3_experiment(
        instance,
        nsga3_generations,
        population_size=nsga3_population_size,
        warm_start=nsga3_warm_start,
        seed=nsga3_seed,
        constraint_handling=nsga3_constraint_handling,
        parent_selection=nsga3_parent_selection,
        crossover_policy=nsga3_crossover_policy,
        local_improvement=nsga3_local_improvement,
        sequence_mutation=nsga3_sequence_mutation,
        immigrant_policy=nsga3_immigrant_policy,
        immigrant_count=nsga3_immigrant_count,
        immigrant_period=nsga3_immigrant_period,
        immigrant_archive_size=nsga3_immigrant_archive_size,
        report_member_policy=nsga3_report_member_policy,
        collect_generation_diagnostics=nsga3_collect_generation_diagnostics,
    )
    results['experiments'].append(result)
    if result['feasible']:
        tardiness_note = ""
        if (
            result.get('tardiness_best_weighted_tardiness') is not None
            and result.get('report_weighted_tardiness') is not None
            and result.get('tardiness_best_weighted_tardiness') < result.get('report_weighted_tardiness')
        ):
            tardiness_note = (
                f", tardiness-best={result['tardiness_best_n_tardy_jobs']} "
                f"(WTd {result['tardiness_best_weighted_tardiness']:.1f}, "
                f"mk {result['tardiness_best_makespan']:.1f})"
            )
        print(
            f"  report={result['report_member_key']}, "
            f"makespan={result['report_makespan']:.1f}, "
            f"energy={result['report_total_energy']:.1f}, "
            f"tardy_jobs={result['report_n_tardy_jobs']}{tardiness_note}, "
            f"seeds={result['warm_start_seed_count']}, "
            f"crossover={result['crossover_policy']}, "
            f"repair={result['local_improvement']}, "
            f"mutation={result['sequence_mutation']}, "
            f"immigrants={result['immigrant_policy']}, "
            f"time={result['time_seconds']:.1f}s"
        )
    else:
        print(f"  {result['error']}")

    # CP-SAT (optional)
    if run_cp:
        print("\nRunning CP-SAT...")
        result = run_cp_experiment(instance, time_limit=60)
        results['experiments'].append(result)
        if 'error' in result:
            print(f"  {result['error']}")
        else:
            print(f"  makespan={result['makespan']:.1f}, time={result['time_seconds']:.1f}s")

    return results


def run_suite_comparison(
    benchmark_dir: str = "benchmarks/small",
    output_path: str = "experiments/results/comparison.json",
    run_cp: bool = False,
    nsga3_generations: int = 30,
    nsga3_population_size: int = 30,
    nsga3_warm_start: bool = True,
    nsga3_seed: int = 42,
    nsga3_constraint_handling: str = "legacy_scalar_penalized_objectives",
    nsga3_parent_selection: str = "random_pairing",
    nsga3_crossover_policy: str = NSGA3_DEFAULT_CROSSOVER_POLICY,
    nsga3_local_improvement: str = NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
    nsga3_sequence_mutation: str = NSGA3_DEFAULT_SEQUENCE_MUTATION,
    nsga3_immigrant_policy: str = NSGA3_DEFAULT_IMMIGRANT_POLICY,
    nsga3_immigrant_count: int = 2,
    nsga3_immigrant_period: int = 5,
    nsga3_immigrant_archive_size: int = 8,
    nsga3_report_member_policy: str = "best_makespan_feasible",
    nsga3_collect_generation_diagnostics: bool = False,
    command: str = "",
):
    """Run comparison on all instances in directory"""
    import glob

    print("=" * 60)
    print("SFJSSP Solver Comparison Experiment")
    print("=" * 60)

    # Find all benchmark files
    pattern = os.path.join(benchmark_dir, "*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No benchmark files found in {benchmark_dir}")
        return

    print(f"Found {len(files)} benchmark files")

    all_results: List[Dict[str, Any]] = []

    for filepath in files:
        instance_name = os.path.basename(filepath).replace('.json', '')
        instance = load_benchmark(filepath)

        results = run_comparison(
            instance,
            instance_name,
            run_cp=run_cp,
            nsga3_generations=nsga3_generations,
            nsga3_population_size=nsga3_population_size,
            nsga3_warm_start=nsga3_warm_start,
            nsga3_seed=nsga3_seed,
            nsga3_constraint_handling=nsga3_constraint_handling,
            nsga3_parent_selection=nsga3_parent_selection,
            nsga3_crossover_policy=nsga3_crossover_policy,
            nsga3_local_improvement=nsga3_local_improvement,
            nsga3_sequence_mutation=nsga3_sequence_mutation,
            nsga3_immigrant_policy=nsga3_immigrant_policy,
            nsga3_immigrant_count=nsga3_immigrant_count,
            nsga3_immigrant_period=nsga3_immigrant_period,
            nsga3_immigrant_archive_size=nsga3_immigrant_archive_size,
            nsga3_report_member_policy=nsga3_report_member_policy,
            nsga3_collect_generation_diagnostics=nsga3_collect_generation_diagnostics,
        )
        all_results.append(results)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "provenance": _build_provenance(
            benchmark_dir=benchmark_dir,
            output_path=output_path,
            run_cp=run_cp,
            nsga3_generations=nsga3_generations,
            nsga3_population_size=nsga3_population_size,
            nsga3_warm_start=nsga3_warm_start,
            nsga3_seed=nsga3_seed,
            nsga3_constraint_handling=nsga3_constraint_handling,
            nsga3_parent_selection=nsga3_parent_selection,
            nsga3_crossover_policy=nsga3_crossover_policy,
            nsga3_local_improvement=nsga3_local_improvement,
            nsga3_sequence_mutation=nsga3_sequence_mutation,
            nsga3_immigrant_policy=nsga3_immigrant_policy,
            nsga3_immigrant_count=nsga3_immigrant_count,
            nsga3_immigrant_period=nsga3_immigrant_period,
            nsga3_immigrant_archive_size=nsga3_immigrant_archive_size,
            nsga3_report_member_policy=nsga3_report_member_policy,
            command=command,
        ),
        "results": all_results,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")

    # Print summary
    print_summary(all_results)

    return payload


def print_summary(results: list):
    """Print summary table"""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Instance':<25} {'Method':<20} {'Makespan':>12} {'Energy':>12} {'Time(s)':>10}")
    print("-" * 80)

    for r in results:
        instance = r['instance']
        for exp in r['experiments']:
            method = exp['method']
            makespan_value = exp.get('makespan')
            energy_value = exp.get('total_energy')
            if method.startswith('NSGA-III') and exp.get('report_member_metrics'):
                method = f"{method} [{exp.get('report_member_key')}]"
                makespan_value = exp.get('report_makespan')
                energy_value = exp.get('report_total_energy')
            makespan = (
                f"{makespan_value:.1f}"
                if makespan_value is not None
                else "N/A"
            )
            energy = (
                f"{energy_value:.0f}"
                if energy_value is not None
                else "N/A"
            )
            time_s = f"{exp['time_seconds']:.2f}" if 'time_seconds' in exp else "N/A"

            print(f"{instance:<25} {method:<20} {makespan:>12} {energy:>12} {time_s:>10}")

    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFJSSP Solver Comparison")
    parser.add_argument("--benchmark-dir", type=str, default="benchmarks/small",
                       help="Directory with benchmark JSON files")
    parser.add_argument("--output", type=str, default="experiments/results/comparison.json",
                       help="Output JSON path")
    parser.add_argument("--cp", action="store_true", help="Run CP-SAT solver")
    parser.add_argument("--generations", type=int, default=30,
                       help="NSGA-III generations")
    parser.add_argument("--population-size", type=int, default=30,
                       help="NSGA-III population size")
    parser.add_argument("--seed", type=int, default=42,
                       help="NSGA-III RNG seed")
    parser.add_argument(
        "--nsga-constraint-handling",
        type=str,
        default="legacy_scalar_penalized_objectives",
        choices=list(NSGA3_CONSTRAINT_HANDLING_POLICIES),
        help="Constraint handling policy used inside NSGA-III survival ranking",
    )
    parser.add_argument(
        "--nsga-report-member-policy",
        type=str,
        default="best_makespan_feasible",
        choices=list(NSGA3_REPORT_MEMBER_POLICIES),
        help="Representative feasible Pareto member to publish as the NSGA report member",
    )
    parser.add_argument(
        "--nsga-parent-selection",
        type=str,
        default="random_pairing",
        choices=["random_pairing", "feasible_tardiness_tournament"],
        help="Parent selection policy used during NSGA-III mating",
    )
    parser.add_argument(
        "--nsga-crossover-policy",
        type=str,
        default=NSGA3_DEFAULT_CROSSOVER_POLICY,
        choices=list(NSGA3_CROSSOVER_POLICIES),
        help="Sequence crossover policy used during NSGA-III offspring construction",
    )
    parser.add_argument(
        "--nsga-local-improvement",
        type=str,
        default=NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        choices=list(NSGA3_LOCAL_IMPROVEMENT_POLICIES),
        help="Optional post-mutation local improvement policy used before NSGA-III survival",
    )
    parser.add_argument(
        "--nsga-sequence-mutation",
        type=str,
        default=NSGA3_DEFAULT_SEQUENCE_MUTATION,
        choices=list(NSGA3_SEQUENCE_MUTATION_POLICIES),
        help="Sequence mutation policy used during NSGA-III offspring construction",
    )
    parser.add_argument(
        "--nsga-immigrant-policy",
        type=str,
        default=NSGA3_DEFAULT_IMMIGRANT_POLICY,
        choices=list(NSGA3_IMMIGRANT_POLICIES),
        help="Optional immigrant injection policy used before NSGA-III survival",
    )
    parser.add_argument(
        "--nsga-immigrant-count",
        type=int,
        default=2,
        help="Maximum number of archive immigrants injected at each immigrant event",
    )
    parser.add_argument(
        "--nsga-immigrant-period",
        type=int,
        default=5,
        help="Inject immigrants every N generations when immigrant policy is enabled",
    )
    parser.add_argument(
        "--nsga-immigrant-archive-size",
        type=int,
        default=8,
        help="Maximum number of hard-feasible tardiness candidates retained in the immigrant archive",
    )
    parser.add_argument(
        "--no-nsga-warm-start",
        action="store_true",
        help="Disable deterministic greedy warm-start seeds for NSGA-III",
    )
    parser.add_argument(
        "--nsga-collect-generation-diagnostics",
        action="store_true",
        help="Embed per-generation NSGA feasibility/tardiness diagnostics in experiment rows",
    )

    args = parser.parse_args()

    command = "python -m experiments.compare_solvers"
    if len(sys.argv) > 1:
        command = f"{command} {subprocess.list2cmdline(sys.argv[1:])}"

    run_suite_comparison(
        benchmark_dir=args.benchmark_dir,
        output_path=args.output,
        run_cp=args.cp,
        nsga3_generations=args.generations,
        nsga3_population_size=args.population_size,
        nsga3_warm_start=not args.no_nsga_warm_start,
        nsga3_seed=args.seed,
        nsga3_constraint_handling=args.nsga_constraint_handling,
        nsga3_parent_selection=args.nsga_parent_selection,
        nsga3_crossover_policy=args.nsga_crossover_policy,
        nsga3_local_improvement=args.nsga_local_improvement,
        nsga3_sequence_mutation=args.nsga_sequence_mutation,
        nsga3_immigrant_policy=args.nsga_immigrant_policy,
        nsga3_immigrant_count=args.nsga_immigrant_count,
        nsga3_immigrant_period=args.nsga_immigrant_period,
        nsga3_immigrant_archive_size=args.nsga_immigrant_archive_size,
        nsga3_report_member_policy=args.nsga_report_member_policy,
        nsga3_collect_generation_diagnostics=args.nsga_collect_generation_diagnostics,
        command=command,
    )
