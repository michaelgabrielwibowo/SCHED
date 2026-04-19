"""Versioned artifact contracts for experiment reporting."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


COMPARISON_ARTIFACT_SCHEMA = "comparison_results_v5"
LEGACY_COMPARISON_ARTIFACT_SCHEMAS = frozenset({"comparison_results_v4"})
BUDGET_SWEEP_ARTIFACT_SCHEMA = "nsga_budget_sweep_v4"
POLICY_ANALYSIS_ARTIFACT_SCHEMA = "nsga_policy_analysis_v2"
REPRESENTATION_AUDIT_ARTIFACT_SCHEMA = "nsga_representation_audit_v2"


# Canonical schedule metrics must keep the same names from Schedule.evaluate().
CANONICAL_SCHEDULE_METRIC_FIELDS: Sequence[str] = (
    "makespan",
    "total_energy",
    "max_ergonomic_exposure",
    "total_labor_cost",
    "total_tardiness",
    "weighted_tardiness",
    "n_tardy_jobs",
)

CANONICAL_SCHEDULE_PENALTY_FIELDS: Sequence[str] = (
    "hard_violations",
    "n_tardy_jobs",
    "weighted_tardiness",
    "ocra_penalty",
    "total_penalty",
)

LEGACY_METRIC_ALIASES = frozenset({"energy", "ergonomic_risk", "labor_cost"})
LEGACY_SELECTED_PREFIX = "selected_"

COMPARISON_PROVENANCE_FIELDS: Sequence[str] = (
    "artifact_schema",
    "git_commit",
    "git_dirty",
    "git_status_short",
    "command",
    "benchmark_dir",
    "output_path",
    "nsga3_generations",
    "nsga3_population_size",
    "nsga3_warm_start",
    "nsga3_seed_source",
    "nsga3_seed",
    "nsga3_constraint_handling",
    "nsga3_parent_selection",
    "nsga3_crossover_policy",
    "nsga3_local_improvement",
    "nsga3_sequence_mutation",
    "nsga3_immigrant_policy",
    "nsga3_immigrant_count",
    "nsga3_immigrant_period",
    "nsga3_immigrant_archive_size",
    "nsga3_report_member_policy",
    "cp_enabled",
    "cp_verified_scope",
    "python_version",
    "timestamp",
)

BUDGET_SWEEP_PROVENANCE_FIELDS: Sequence[str] = (
    "artifact_schema",
    "git_commit",
    "git_dirty",
    "git_status_short",
    "command",
    "benchmark_dir",
    "output_path",
    "generations",
    "population_sizes",
    "nsga3_warm_start",
    "nsga3_seed",
    "nsga3_constraint_handling",
    "nsga3_crossover_policy",
    "nsga3_sequence_mutation",
    "nsga3_immigrant_policy",
    "nsga3_immigrant_count",
    "nsga3_immigrant_period",
    "nsga3_immigrant_archive_size",
    "nsga3_report_member_policy",
    "baseline_scope",
    "python_version",
    "timestamp",
)

POLICY_ANALYSIS_PROVENANCE_FIELDS: Sequence[str] = (
    "artifact_schema",
    "git_commit",
    "git_dirty",
    "git_status_short",
    "command",
    "benchmark_dir",
    "output_path",
    "nsga3_generations",
    "nsga3_population_size",
    "nsga3_seed",
    "nsga3_warm_start",
    "baseline_constraint_handling",
    "analysis_policies",
    "nsga3_report_member_policy",
    "python_version",
    "timestamp",
)

REPRESENTATION_AUDIT_PROVENANCE_FIELDS: Sequence[str] = (
    "artifact_schema",
    "git_commit",
    "git_dirty",
    "git_status_short",
    "command",
    "benchmark_dir",
    "output_path",
    "baseline_artifact",
    "reference_artifact",
    "seed",
    "python_version",
    "timestamp",
)


class ArtifactSchemaError(ValueError):
    """Raised when an artifact payload drifts from the declared contract."""


def derive_canonical_feasibility(
    feasibility_flag: Any,
    penalties: Optional[Mapping[str, Any]],
    constraint_violations: Optional[Sequence[Any]],
) -> bool:
    """Derive canonical feasibility from explicit payload evidence only."""
    if not bool(feasibility_flag):
        return False
    if penalties is None:
        return False
    hard_violations = float((penalties.get("hard_violations", 0.0) or 0.0))
    if hard_violations > 0.0:
        return False
    return not list(constraint_violations or [])


def is_current_comparison_schema(schema: Optional[str]) -> bool:
    """Return whether a comparison payload uses the current strict contract."""
    return schema == COMPARISON_ARTIFACT_SCHEMA


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactSchemaError(message)


def _validate_required_fields(
    payload: Mapping[str, Any],
    required_fields: Iterable[str],
    context: str,
) -> None:
    missing = [field for field in required_fields if field not in payload]
    _require(not missing, f"{context} missing required fields: {missing}")


def _validate_metrics(
    metrics: Optional[Mapping[str, Any]],
    context: str,
) -> None:
    if metrics is None:
        return
    _validate_required_fields(metrics, CANONICAL_SCHEDULE_METRIC_FIELDS, context)
    _require(
        not (set(metrics) & LEGACY_METRIC_ALIASES),
        f"{context} contains legacy metric aliases: {sorted(set(metrics) & LEGACY_METRIC_ALIASES)}",
    )


def _validate_penalties(
    penalties: Optional[Mapping[str, Any]],
    context: str,
) -> None:
    if penalties is None:
        return
    _validate_required_fields(penalties, CANONICAL_SCHEDULE_PENALTY_FIELDS, context)


def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    expected_schema: str,
    required_fields: Sequence[str],
) -> None:
    _validate_required_fields(provenance, required_fields, "artifact provenance")
    _require(
        provenance.get("artifact_schema") == expected_schema,
        f"artifact provenance expected schema {expected_schema!r}, got {provenance.get('artifact_schema')!r}",
    )


def validate_comparison_payload(payload: Mapping[str, Any]) -> None:
    """Validate the current comparison artifact contract."""
    _validate_required_fields(payload, ("provenance", "results"), "comparison payload")
    provenance = payload["provenance"]
    _require(isinstance(provenance, Mapping), "comparison provenance must be a mapping")
    _validate_provenance(
        provenance,
        expected_schema=COMPARISON_ARTIFACT_SCHEMA,
        required_fields=COMPARISON_PROVENANCE_FIELDS,
    )
    results = payload["results"]
    _require(isinstance(results, list), "comparison results must be a list")
    for index, result in enumerate(results):
        _validate_comparison_result(result, context=f"comparison results[{index}]")


def _validate_comparison_result(result: Mapping[str, Any], *, context: str) -> None:
    _validate_required_fields(result, ("instance", "experiments"), context)
    experiments = result["experiments"]
    _require(isinstance(experiments, list), f"{context}.experiments must be a list")
    for exp_index, experiment in enumerate(experiments):
        _validate_experiment_row(
            experiment,
            context=f"{context}.experiments[{exp_index}]",
        )


def _validate_experiment_row(experiment: Mapping[str, Any], *, context: str) -> None:
    _validate_required_fields(experiment, ("method", "time_seconds", "feasible"), context)
    _require(
        not (set(experiment) & LEGACY_METRIC_ALIASES),
        f"{context} uses legacy metric aliases: {sorted(set(experiment) & LEGACY_METRIC_ALIASES)}",
    )
    if str(experiment.get("method", "")).startswith("NSGA-III"):
        _validate_nsga_report_row(experiment, context=context)
        return
    if "error" not in experiment:
        _validate_required_fields(
            experiment,
            ("makespan", "total_energy", "max_ergonomic_exposure", "total_labor_cost"),
            context,
        )


def _validate_nsga_report_row(experiment: Mapping[str, Any], *, context: str) -> None:
    _require(
        not any(str(key).startswith(LEGACY_SELECTED_PREFIX) for key in experiment),
        f"{context} still exposes legacy selected_* aliases",
    )
    _validate_required_fields(
        experiment,
        (
            "report_member_key",
            "report_member_selection_policy",
            "report_member_is_feasible",
            "report_member_metrics",
            "report_member_penalties",
            "report_member_constraint_violations",
            "report_makespan",
            "report_total_energy",
            "report_max_ergonomic_exposure",
            "report_total_labor_cost",
            "report_weighted_tardiness",
            "report_n_tardy_jobs",
            "report_total_penalty",
            "representative_members",
            "tardiness_best_member_policy",
            "tardiness_best_member_metrics",
            "tardiness_best_member_penalties",
            "tardiness_best_member_constraint_violations",
        ),
        context,
    )
    report_metrics = experiment.get("report_member_metrics")
    report_penalties = experiment.get("report_member_penalties")
    report_violations = experiment.get("report_member_constraint_violations")
    _validate_metrics(report_metrics, f"{context}.report_member_metrics")
    _validate_penalties(report_penalties, f"{context}.report_member_penalties")
    derived_feasible = derive_canonical_feasibility(
        report_metrics is not None,
        report_penalties,
        report_violations,
    )
    if report_metrics is None:
        _require(
            experiment.get("report_member_is_feasible") is False,
            f"{context} cannot mark a missing report member as feasible",
        )
    else:
        _require(
            experiment.get("report_member_is_feasible") == derived_feasible,
            f"{context} report_member_is_feasible does not match penalties/violations evidence",
        )
        _require(
            experiment.get("makespan") == report_metrics.get("makespan"),
            f"{context}.makespan must mirror report member makespan",
        )
        _require(
            experiment.get("total_energy") == report_metrics.get("total_energy"),
            f"{context}.total_energy must mirror report member total_energy",
        )
        _require(
            experiment.get("max_ergonomic_exposure") == report_metrics.get("max_ergonomic_exposure"),
            f"{context}.max_ergonomic_exposure must mirror report member max_ergonomic_exposure",
        )
        _require(
            experiment.get("total_labor_cost") == report_metrics.get("total_labor_cost"),
            f"{context}.total_labor_cost must mirror report member total_labor_cost",
        )

    tardiness_metrics = experiment.get("tardiness_best_member_metrics")
    tardiness_penalties = experiment.get("tardiness_best_member_penalties")
    _validate_metrics(tardiness_metrics, f"{context}.tardiness_best_member_metrics")
    _validate_penalties(tardiness_penalties, f"{context}.tardiness_best_member_penalties")


def validate_budget_sweep_payload(payload: Mapping[str, Any]) -> None:
    """Validate the current budget sweep artifact contract."""
    _validate_required_fields(payload, ("provenance", "results", "summary"), "budget sweep payload")
    provenance = payload["provenance"]
    _require(isinstance(provenance, Mapping), "budget sweep provenance must be a mapping")
    _validate_provenance(
        provenance,
        expected_schema=BUDGET_SWEEP_ARTIFACT_SCHEMA,
        required_fields=BUDGET_SWEEP_PROVENANCE_FIELDS,
    )
    results = payload["results"]
    _require(isinstance(results, list), "budget sweep results must be a list")
    for result_index, result in enumerate(results):
        _validate_required_fields(
            result,
            ("instance", "best_published_greedy_method", "best_published_greedy_makespan", "published_greedy_runs", "runs"),
            f"budget sweep results[{result_index}]",
        )
        runs = result["runs"]
        _require(isinstance(runs, list), f"budget sweep results[{result_index}].runs must be a list")
        for run_index, run in enumerate(runs):
            _validate_budget_run(
                run,
                context=f"budget sweep results[{result_index}].runs[{run_index}]",
            )
    summary = payload["summary"]
    _require(isinstance(summary, list), "budget sweep summary must be a list")
    for index, row in enumerate(summary):
        _validate_budget_summary_row(row, context=f"budget sweep summary[{index}]")


def _validate_budget_run(run: Mapping[str, Any], *, context: str) -> None:
    _require(
        not any(str(key).startswith(LEGACY_SELECTED_PREFIX) for key in run),
        f"{context} still exposes legacy selected_* aliases",
    )
    _validate_required_fields(
        run,
        (
            "generations",
            "population_size",
            "report_member_key",
            "report_member_selection_policy",
            "report_member_metrics",
            "report_member_penalties",
            "report_makespan",
            "report_weighted_tardiness",
            "report_n_tardy_jobs",
            "report_total_penalty",
            "report_member_zero_tardy",
            "tardiness_best_member_metrics",
            "tardiness_best_member_penalties",
            "tardiness_best_weighted_tardiness",
            "tardiness_best_n_tardy_jobs",
            "tardiness_best_improves_report",
            "time_seconds",
        ),
        context,
    )
    _validate_metrics(run.get("report_member_metrics"), f"{context}.report_member_metrics")
    _validate_penalties(run.get("report_member_penalties"), f"{context}.report_member_penalties")
    _validate_metrics(run.get("tardiness_best_member_metrics"), f"{context}.tardiness_best_member_metrics")
    _validate_penalties(run.get("tardiness_best_member_penalties"), f"{context}.tardiness_best_member_penalties")


def _validate_budget_summary_row(row: Mapping[str, Any], *, context: str) -> None:
    _require(
        not any("selected" in str(key) for key in row),
        f"{context} still exposes selected-based aggregate fields",
    )
    _validate_required_fields(
        row,
        (
            "generations",
            "population_size",
            "instances",
            "constraint_handling",
            "report_member_policy",
            "avg_report_makespan",
            "avg_report_weighted_tardiness",
            "avg_tardiness_best_weighted_tardiness",
            "avg_report_n_tardy_jobs",
            "avg_tardiness_best_n_tardy_jobs",
            "avg_report_total_penalty",
            "avg_time_seconds",
            "instances_with_report_zero_tardy_member",
            "instances_with_tardiness_best_zero_tardy_member",
            "instances_where_tardiness_best_improves_report",
        ),
        context,
    )


def validate_policy_analysis_payload(payload: Mapping[str, Any]) -> None:
    """Validate provenance and top-level shape for the policy analysis artifact."""
    _validate_required_fields(payload, ("provenance", "results", "summary"), "policy analysis payload")
    provenance = payload["provenance"]
    _require(isinstance(provenance, Mapping), "policy analysis provenance must be a mapping")
    _validate_provenance(
        provenance,
        expected_schema=POLICY_ANALYSIS_ARTIFACT_SCHEMA,
        required_fields=POLICY_ANALYSIS_PROVENANCE_FIELDS,
    )


def validate_representation_audit_payload(payload: Mapping[str, Any]) -> None:
    """Validate provenance and top-level shape for the representation audit artifact."""
    _validate_required_fields(
        payload,
        ("provenance", "baseline_truth_table", "results", "aggregate_summary"),
        "representation audit payload",
    )
    provenance = payload["provenance"]
    _require(isinstance(provenance, Mapping), "representation audit provenance must be a mapping")
    _validate_provenance(
        provenance,
        expected_schema=REPRESENTATION_AUDIT_ARTIFACT_SCHEMA,
        required_fields=REPRESENTATION_AUDIT_PROVENANCE_FIELDS,
    )
