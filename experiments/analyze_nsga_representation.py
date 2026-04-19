#!/usr/bin/env python
"""
Offline NSGA representation bottleneck audit.

This script does not run live evolution. It reconstructs the best deterministic
generation-0 warm-start seed per benchmark, probes bounded neighborhoods around
that seed, and reports which genome dimensions still show untapped value.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from ..experiments.artifact_schemas import (
        REPRESENTATION_AUDIT_ARTIFACT_SCHEMA,
        derive_canonical_feasibility,
    )
    from ..experiments.compare_solvers import _get_git_status_short, load_benchmark
    from ..moea.nsga3 import (
        _clone_seed_genome,
        _collect_seed_genome_candidates,
        _iter_adjacent_urgent_swap_variants,
        _iter_pull_forward_variants,
        _seed_genome_signature,
        _sequence_urgency_records,
        evaluate_sfjssp_genome_detailed,
    )
except ImportError:  # pragma: no cover - supports repo-root imports
    from experiments.artifact_schemas import (
        REPRESENTATION_AUDIT_ARTIFACT_SCHEMA,
        derive_canonical_feasibility,
    )
    from experiments.compare_solvers import _get_git_status_short, load_benchmark
    from moea.nsga3 import (
        _clone_seed_genome,
        _collect_seed_genome_candidates,
        _iter_adjacent_urgent_swap_variants,
        _iter_pull_forward_variants,
        _seed_genome_signature,
        _sequence_urgency_records,
        evaluate_sfjssp_genome_detailed,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_ARTIFACT = REPO_ROOT / "experiments" / "results" / "comparison_2026-04-15-crossover-baseline.json"
DEFAULT_REFERENCE_ARTIFACT = REPO_ROOT / "experiments" / "results" / "comparison_2026-04-15-crossover-urgent.json"
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "results" / "nsga_representation_audit_2026-04-15.json"
SEQUENCE_FAMILY_NAME = "sequence_only"
ASSIGNMENT_FAMILY_NAME = "assignment_only"
OFFSET_FAMILY_NAME = "offset_only"
MIXED_FAMILY_NAME = "mixed_one_step"
MAX_SEQUENCE_CANDIDATES = 100
MAX_ASSIGNMENT_CANDIDATES = 100
MAX_OFFSET_CANDIDATES = 60
MAX_MIXED_CANDIDATES = 40
MAX_OFFSET_VALUE = 4


@dataclass
class AuditCandidate:
    """One bounded offline audit candidate."""

    name: str
    family: str
    move_type: str
    genome: Dict[str, np.ndarray]
    details: Dict[str, Any]


def _get_git_commit() -> Optional[str]:
    """Return the current git commit hash when available."""
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


def _build_provenance(
    benchmark_dir: str,
    output_path: str,
    baseline_artifact: str,
    reference_artifact: Optional[str],
    seed: int,
    command: str,
) -> Dict[str, Any]:
    """Build artifact provenance for the representation audit."""
    git_status_short = _get_git_status_short()
    return {
        "artifact_schema": REPRESENTATION_AUDIT_ARTIFACT_SCHEMA,
        "git_commit": _get_git_commit(),
        "git_dirty": bool(git_status_short),
        "git_status_short": git_status_short,
        "command": command,
        "benchmark_dir": benchmark_dir,
        "output_path": output_path,
        "baseline_artifact": baseline_artifact,
        "reference_artifact": reference_artifact,
        "seed": seed,
        "python_version": sys.version.split()[0],
        "timestamp": datetime.now().isoformat(),
    }


def _details_are_hard_feasible(details: Dict[str, Any]) -> bool:
    """Return whether a detailed payload is hard-feasible."""
    penalties = details.get("penalties") or {}
    return float(penalties.get("hard_violations", 0.0) or 0.0) <= 0.0


def _details_are_canonically_feasible(details: Dict[str, Any]) -> bool:
    """Return whether a detailed payload is fully feasible under the canonical oracle."""
    return derive_canonical_feasibility(
        details.get("is_feasible", True),
        details.get("penalties") or {},
        details.get("constraint_violations") or [],
    )


def _candidate_score(details: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    """Rank candidates by hard feasibility, tardiness, then efficiency."""
    metrics = details.get("metrics") or {}
    penalties = details.get("penalties") or {}
    hard_violations = float(penalties.get("hard_violations", 0.0))
    if hard_violations > 0.0 or not _details_are_canonically_feasible(details):
        return (
            hard_violations if hard_violations > 0.0 else 1.0,
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
        )
    return (
        0.0,
        float(metrics.get("n_tardy_jobs", penalties.get("n_tardy_jobs", float("inf")))),
        float(metrics.get("weighted_tardiness", penalties.get("weighted_tardiness", float("inf")))),
        float(metrics.get("makespan", float("inf"))),
        float(metrics.get("total_energy", float("inf"))),
    )


def _compact_details(details: Dict[str, Any]) -> Dict[str, Any]:
    """Convert detailed-evaluation payload into compact JSON-safe metrics."""
    metrics = dict(details.get("metrics") or {})
    penalties = dict(details.get("penalties") or {})
    constraint_violations = list(details.get("constraint_violations") or [])
    return {
        "hard_feasible": _details_are_hard_feasible(details),
        "is_feasible": derive_canonical_feasibility(
            details.get("is_feasible", True),
            penalties,
            constraint_violations,
        ),
        "n_tardy_jobs": metrics.get("n_tardy_jobs", penalties.get("n_tardy_jobs")),
        "weighted_tardiness": metrics.get("weighted_tardiness", penalties.get("weighted_tardiness")),
        "makespan": metrics.get("makespan"),
        "total_energy": metrics.get("total_energy"),
        "max_ergonomic_exposure": metrics.get("max_ergonomic_exposure"),
        "total_labor_cost": metrics.get("total_labor_cost"),
        "constraint_violations": constraint_violations,
        "hard_violations": penalties.get("hard_violations"),
    }


def _metric_signature(details: Dict[str, Any], precision: int = 6) -> Tuple[Any, ...]:
    """Build a rounded decoded-metric signature for collision tracking."""
    metrics = details.get("metrics") or {}
    penalties = details.get("penalties") or {}
    n_tardy_jobs = metrics.get("n_tardy_jobs", penalties.get("n_tardy_jobs"))
    weighted_tardiness = metrics.get("weighted_tardiness", penalties.get("weighted_tardiness"))
    makespan = metrics.get("makespan")
    total_energy = metrics.get("total_energy")
    return (
        int(n_tardy_jobs) if n_tardy_jobs is not None else None,
        round(float(weighted_tardiness), precision) if weighted_tardiness is not None else None,
        round(float(makespan), precision) if makespan is not None else None,
        round(float(total_energy), precision) if total_energy is not None else None,
    )


def _candidate_improves_over_seed(
    seed_details: Dict[str, Any],
    candidate_details: Dict[str, Any],
) -> bool:
    """Return whether one candidate beats the best hard-feasible seed."""
    return _candidate_score(candidate_details) < _candidate_score(seed_details)


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    """Return the median when values exist."""
    if not values:
        return None
    return float(statistics.median(values))


def _signature_list(signature: Tuple[Tuple[int, ...], ...]) -> List[List[int]]:
    """Convert a nested tuple signature into JSON-safe nested lists."""
    return [[int(value) for value in component] for component in signature]


def _build_candidate_record(candidate: AuditCandidate) -> Dict[str, Any]:
    """Serialize one audit candidate without embedding the full schedule."""
    return {
        "name": candidate.name,
        "family": candidate.family,
        "move_type": candidate.move_type,
        "genome_signature": _signature_list(_seed_genome_signature(candidate.genome)),
        "metric_signature": list(_metric_signature(candidate.details)),
        **_compact_details(candidate.details),
    }


def _summarize_collision_metrics(
    seed_genome: Dict[str, np.ndarray],
    seed_details: Dict[str, Any],
    candidates: Sequence[AuditCandidate],
) -> Dict[str, Any]:
    """Summarize genome-to-metric collision rate for one evaluated family."""
    genome_signatures = set()
    metric_clusters: Dict[Tuple[Any, ...], List[Tuple[Tuple[int, ...], ...]]] = defaultdict(list)

    baseline_signature = _seed_genome_signature(seed_genome)
    genome_signatures.add(baseline_signature)
    metric_clusters[_metric_signature(seed_details)].append(baseline_signature)

    for candidate in candidates:
        genome_signature = _seed_genome_signature(candidate.genome)
        genome_signatures.add(genome_signature)
        metric_clusters[_metric_signature(candidate.details)].append(genome_signature)

    distinct_metric_signatures = len(metric_clusters)
    distinct_genome_count = len(genome_signatures)
    decoded_collision_rate = 0.0
    if distinct_genome_count > 0:
        decoded_collision_rate = 1.0 - (
            distinct_metric_signatures / float(distinct_genome_count)
        )

    top_clusters = []
    for metric_signature, genomes in sorted(
        metric_clusters.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        if len(genomes) <= 1:
            continue
        top_clusters.append(
            {
                "metric_signature": list(metric_signature),
                "genome_count": len(genomes),
            }
        )
        if len(top_clusters) >= 5:
            break

    return {
        "distinct_genome_count": distinct_genome_count,
        "distinct_metric_signature_count": distinct_metric_signatures,
        "decoded_collision_rate": float(decoded_collision_rate),
        "top_collision_clusters": top_clusters,
    }


def _summarize_family(
    family: str,
    seed_genome: Dict[str, np.ndarray],
    seed_details: Dict[str, Any],
    candidates: Sequence[AuditCandidate],
) -> Dict[str, Any]:
    """Build the uniform JSON summary for one neighborhood family."""
    feasible_candidates = [
        candidate
        for candidate in candidates
        if _details_are_hard_feasible(candidate.details)
    ]
    best_candidate = min(
        feasible_candidates,
        key=lambda candidate: _candidate_score(candidate.details),
        default=None,
    )
    best_seed_score = _candidate_score(seed_details)

    delta_weighted_tardiness_values = []
    delta_makespan_values = []
    improvement_candidates: List[AuditCandidate] = []
    for candidate in feasible_candidates:
        metrics = candidate.details.get("metrics") or {}
        seed_metrics = seed_details.get("metrics") or {}
        if metrics.get("weighted_tardiness") is not None and seed_metrics.get("weighted_tardiness") is not None:
            delta_weighted_tardiness_values.append(
                float(metrics["weighted_tardiness"]) - float(seed_metrics["weighted_tardiness"])
            )
        if metrics.get("makespan") is not None and seed_metrics.get("makespan") is not None:
            delta_makespan_values.append(
                float(metrics["makespan"]) - float(seed_metrics["makespan"])
            )
        if _candidate_improves_over_seed(seed_details, candidate.details):
            improvement_candidates.append(candidate)

    best_improvement_delta_weighted_tardiness = None
    best_improvement_delta_n_tardy_jobs = None
    if improvement_candidates:
        best_improver = min(improvement_candidates, key=lambda candidate: _candidate_score(candidate.details))
        best_metrics = best_improver.details.get("metrics") or {}
        seed_metrics = seed_details.get("metrics") or {}
        if (
            best_metrics.get("weighted_tardiness") is not None
            and seed_metrics.get("weighted_tardiness") is not None
        ):
            best_improvement_delta_weighted_tardiness = (
                float(best_metrics["weighted_tardiness"]) - float(seed_metrics["weighted_tardiness"])
            )
        if best_metrics.get("n_tardy_jobs") is not None and seed_metrics.get("n_tardy_jobs") is not None:
            best_improvement_delta_n_tardy_jobs = (
                float(best_metrics["n_tardy_jobs"]) - float(seed_metrics["n_tardy_jobs"])
            )

    collision = _summarize_collision_metrics(seed_genome, seed_details, candidates)
    return {
        "family": family,
        "seed_score": list(best_seed_score),
        "candidates_evaluated": len(candidates),
        "hard_feasible_count": len(feasible_candidates),
        "best_n_tardy_jobs": None if best_candidate is None else _compact_details(best_candidate.details)["n_tardy_jobs"],
        "best_weighted_tardiness": None if best_candidate is None else _compact_details(best_candidate.details)["weighted_tardiness"],
        "best_makespan": None if best_candidate is None else _compact_details(best_candidate.details)["makespan"],
        "improves_over_best_seed_count": len(improvement_candidates),
        "improves_over_best_seed_best_delta_weighted_tardiness": best_improvement_delta_weighted_tardiness,
        "improves_over_best_seed_best_delta_n_tardy_jobs": best_improvement_delta_n_tardy_jobs,
        "median_delta_weighted_tardiness": _median_or_none(delta_weighted_tardiness_values),
        "median_delta_makespan": _median_or_none(delta_makespan_values),
        "collision": collision,
        "top_improvers": [
            _build_candidate_record(candidate)
            for candidate in sorted(
                improvement_candidates,
                key=lambda candidate: _candidate_score(candidate.details),
            )[:5]
        ],
    }


def _extract_nsga_experiment(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return the NSGA experiment row from one comparison result entry."""
    for experiment in result.get("experiments", []):
        method = str(experiment.get("method", ""))
        if method.startswith("NSGA-III"):
            return experiment
    raise ValueError(f"Missing NSGA experiment in comparison result for {result.get('instance')!r}")


def _load_baseline_truth_table(
    baseline_artifact_path: str,
    reference_artifact_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract the frozen April 15 baseline truth slice per instance."""
    with open(baseline_artifact_path, "r", encoding="utf-8") as f:
        baseline_artifact = json.load(f)

    reference_by_instance: Dict[str, Dict[str, Any]] = {}
    if reference_artifact_path and os.path.exists(reference_artifact_path):
        with open(reference_artifact_path, "r", encoding="utf-8") as f:
            reference_artifact = json.load(f)
        reference_by_instance = {
            entry["instance"]: _extract_nsga_experiment(entry)
            for entry in reference_artifact.get("results", [])
        }

    per_instance = []
    for entry in baseline_artifact.get("results", []):
        instance_name = entry["instance"]
        nsga = _extract_nsga_experiment(entry)
        generation_zero = {}
        diagnostics = nsga.get("generation_diagnostics") or []
        if diagnostics:
            generation_zero = diagnostics[0]

        truth = {
            "instance": instance_name,
            "legacy_baseline": {
                "generation0_best_hard_feasible_weighted_tardiness": generation_zero.get("min_hard_feasible_weighted_tardiness"),
                "generation0_report_weighted_tardiness": generation_zero.get("report_weighted_tardiness"),
                "final_best_hard_feasible_weighted_tardiness": nsga.get("tardiness_best_weighted_tardiness"),
                "final_report_weighted_tardiness": nsga.get("report_weighted_tardiness"),
                "best_child_weighted_tardiness": nsga.get("best_child_weighted_tardiness"),
                "best_child_generation": nsga.get("best_child_generation"),
            },
        }
        if instance_name in reference_by_instance:
            ref = reference_by_instance[instance_name]
            truth["reference_candidate"] = {
                "final_best_hard_feasible_weighted_tardiness": ref.get("tardiness_best_weighted_tardiness"),
                "final_report_weighted_tardiness": ref.get("report_weighted_tardiness"),
                "best_child_weighted_tardiness": ref.get("best_child_weighted_tardiness"),
                "best_child_generation": ref.get("best_child_generation"),
            }
        per_instance.append(truth)

    aggregate = {
        "instances": len(per_instance),
        "median_generation0_best_hard_feasible_weighted_tardiness": _median_or_none(
            [
                item["legacy_baseline"]["generation0_best_hard_feasible_weighted_tardiness"]
                for item in per_instance
                if item["legacy_baseline"]["generation0_best_hard_feasible_weighted_tardiness"] is not None
            ]
        ),
        "median_final_best_hard_feasible_weighted_tardiness": _median_or_none(
            [
                item["legacy_baseline"]["final_best_hard_feasible_weighted_tardiness"]
                for item in per_instance
                if item["legacy_baseline"]["final_best_hard_feasible_weighted_tardiness"] is not None
            ]
        ),
    }
    return {
        "baseline_artifact": baseline_artifact_path,
        "reference_artifact": reference_artifact_path,
        "per_instance": per_instance,
        "aggregate": aggregate,
    }


def load_benchmark_from_instance(instance: Any) -> Any:
    """Clone an instance through its serialized form so evaluation stays isolated."""
    return instance.__class__.from_dict(instance.to_dict())


def _evaluate_genome(instance: Any, genome: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Evaluate one audit candidate against a clean instance clone."""
    instance_copy = load_benchmark_from_instance(instance)
    return evaluate_sfjssp_genome_detailed(instance_copy, genome)


def _sequence_priority_by_operation(
    instance: Any,
    genome: Dict[str, np.ndarray],
    schedule: Any,
) -> Dict[Tuple[int, int], Tuple[Any, ...]]:
    """Map each operation occurrence to the urgency key used for neighborhood ordering."""
    return {
        (int(record["job_id"]), int(record["op_idx"])): tuple(record["priority"])
        for record in _sequence_urgency_records(instance, genome, schedule)
    }


def _ordered_op_indices_by_urgency(
    instance: Any,
    genome: Dict[str, np.ndarray],
    schedule: Any,
) -> List[int]:
    """Return op-list indices ordered by due-date urgency."""
    priority_by_occurrence = _sequence_priority_by_operation(instance, genome, schedule)
    return sorted(
        range(len(genome["op_list"])),
        key=lambda idx: priority_by_occurrence.get(
            tuple(genome["op_list"][idx]),
            (float("inf"), float("inf"), float("inf"), float("inf"), idx),
        ),
    )


def _iter_push_back_variants(
    instance: Any,
    genome: Dict[str, np.ndarray],
    schedule: Any,
    max_candidates: int = 24,
    max_window: int = 3,
) -> Iterable[Tuple[str, Dict[str, np.ndarray]]]:
    """Yield bounded push-back variants for less urgent early-dispatched slots."""
    records = _sequence_urgency_records(instance, genome, schedule)
    candidates = []

    for position, record in enumerate(records):
        if position >= len(records) - 1:
            continue
        target_position: Optional[int] = None
        window_end = min(len(records), position + max_window + 1)
        for later_pos in range(position + 1, window_end):
            later = records[later_pos]
            if later["job_id"] == record["job_id"]:
                continue
            if later["priority"] < record["priority"]:
                target_position = later_pos
        if target_position is None:
            continue
        candidates.append((record["priority"], position, target_position))

    for _, position, target_position in sorted(candidates)[:max_candidates]:
        variant = _clone_seed_genome(genome)
        moved_job = int(variant["sequence"][position])
        reduced_sequence = np.delete(variant["sequence"], position)
        variant["sequence"] = np.insert(reduced_sequence, target_position, moved_job).astype(int, copy=False)
        yield f"push_back_{position}_to_{target_position}", variant


def _build_sequence_family_candidates(
    instance: Any,
    seed_genome: Dict[str, np.ndarray],
    seed_details: Dict[str, Any],
    max_candidates: int = MAX_SEQUENCE_CANDIDATES,
) -> List[AuditCandidate]:
    """Evaluate bounded sequence-only variants around one best seed."""
    schedule = seed_details.get("schedule")
    raw_candidates: List[Tuple[str, Dict[str, np.ndarray], str]] = []
    raw_candidates.extend(
        (name, genome, "adjacent_swap")
        for name, genome in _iter_adjacent_urgent_swap_variants(
            instance, seed_genome, schedule, max_candidates=24
        )
    )
    raw_candidates.extend(
        (name, genome, "urgent_pull_forward")
        for name, genome in _iter_pull_forward_variants(
            instance, seed_genome, schedule, max_candidates=24, max_window=3
        )
    )
    raw_candidates.extend(
        (name, genome, "urgent_push_back")
        for name, genome in _iter_push_back_variants(
            instance, seed_genome, schedule, max_candidates=24, max_window=3
        )
    )

    seen = {_seed_genome_signature(seed_genome)}
    accepted: List[AuditCandidate] = []
    for name, genome, move_type in raw_candidates:
        signature = _seed_genome_signature(genome)
        if signature in seen:
            continue
        seen.add(signature)
        details = _evaluate_genome(instance, genome)
        accepted.append(
            AuditCandidate(
                name=name,
                family=SEQUENCE_FAMILY_NAME,
                move_type=move_type,
                genome=genome,
                details=details,
            )
        )
        if len(accepted) >= max_candidates:
            break
    return accepted


def _build_assignment_family_candidates(
    instance: Any,
    seed_genome: Dict[str, np.ndarray],
    seed_details: Dict[str, Any],
    max_candidates: int = MAX_ASSIGNMENT_CANDIDATES,
) -> List[AuditCandidate]:
    """Evaluate bounded machine/worker/mode reassignment moves around one best seed."""
    schedule = seed_details.get("schedule")
    ordered_indices = _ordered_op_indices_by_urgency(instance, seed_genome, schedule)
    seen = {_seed_genome_signature(seed_genome)}
    accepted: List[AuditCandidate] = []

    for op_list_index in ordered_indices:
        job_id, op_idx = seed_genome["op_list"][op_list_index]
        operation = instance.get_operation(job_id, op_idx)
        if operation is None:
            continue

        current_machine = int(seed_genome["machines"][op_list_index])
        current_worker = int(seed_genome["workers"][op_list_index])
        current_mode = int(seed_genome["modes"][op_list_index])

        for machine_id in sorted(operation.eligible_machines):
            if machine_id == current_machine:
                continue
            mode_choices = operation.processing_times.get(machine_id, {})
            if not mode_choices:
                continue
            candidate = _clone_seed_genome(seed_genome)
            candidate["machines"][op_list_index] = int(machine_id)
            candidate["modes"][op_list_index] = int(
                current_mode if current_mode in mode_choices else min(mode_choices)
            )
            signature = _seed_genome_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            accepted.append(
                AuditCandidate(
                    name=f"machine_reassign_{job_id}_{op_idx}_to_{machine_id}",
                    family=ASSIGNMENT_FAMILY_NAME,
                    move_type="machine_reassign",
                    genome=candidate,
                    details=_evaluate_genome(instance, candidate),
                )
            )
            if len(accepted) >= max_candidates:
                return accepted

        for worker_id in sorted(operation.eligible_workers):
            if worker_id == current_worker:
                continue
            candidate = _clone_seed_genome(seed_genome)
            candidate["workers"][op_list_index] = int(worker_id)
            signature = _seed_genome_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            accepted.append(
                AuditCandidate(
                    name=f"worker_reassign_{job_id}_{op_idx}_to_{worker_id}",
                    family=ASSIGNMENT_FAMILY_NAME,
                    move_type="worker_reassign",
                    genome=candidate,
                    details=_evaluate_genome(instance, candidate),
                )
            )
            if len(accepted) >= max_candidates:
                return accepted

        for mode_id in sorted(operation.processing_times.get(current_machine, {})):
            if mode_id == current_mode:
                continue
            candidate = _clone_seed_genome(seed_genome)
            candidate["modes"][op_list_index] = int(mode_id)
            signature = _seed_genome_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            accepted.append(
                AuditCandidate(
                    name=f"mode_reassign_{job_id}_{op_idx}_to_{mode_id}",
                    family=ASSIGNMENT_FAMILY_NAME,
                    move_type="mode_reassign",
                    genome=candidate,
                    details=_evaluate_genome(instance, candidate),
                )
            )
            if len(accepted) >= max_candidates:
                return accepted

    return accepted


def _build_offset_family_candidates(
    instance: Any,
    seed_genome: Dict[str, np.ndarray],
    seed_details: Dict[str, Any],
    max_candidates: int = MAX_OFFSET_CANDIDATES,
) -> List[AuditCandidate]:
    """Evaluate bounded offset perturbations around one best seed."""
    schedule = seed_details.get("schedule")
    ordered_indices = _ordered_op_indices_by_urgency(instance, seed_genome, schedule)
    seen = {_seed_genome_signature(seed_genome)}
    accepted: List[AuditCandidate] = []
    deltas = (-2, -1, 1, 2)

    for op_list_index in ordered_indices:
        current_offset = int(seed_genome["offsets"][op_list_index])
        job_id, op_idx = seed_genome["op_list"][op_list_index]
        for delta in deltas:
            candidate_offset = int(min(MAX_OFFSET_VALUE, max(0, current_offset + delta)))
            if candidate_offset == current_offset:
                continue
            candidate = _clone_seed_genome(seed_genome)
            candidate["offsets"][op_list_index] = candidate_offset
            signature = _seed_genome_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            accepted.append(
                AuditCandidate(
                    name=f"offset_adjust_{job_id}_{op_idx}_{current_offset}_to_{candidate_offset}",
                    family=OFFSET_FAMILY_NAME,
                    move_type="offset_adjust",
                    genome=candidate,
                    details=_evaluate_genome(instance, candidate),
                )
            )
            if len(accepted) >= max_candidates:
                return accepted
    return accepted


def _best_candidates_for_mixed(
    candidates: Sequence[AuditCandidate],
    limit: int = 5,
) -> List[AuditCandidate]:
    """Select the best hard-feasible candidates from one pure family for mixed composition."""
    feasible_candidates = [
        candidate
        for candidate in candidates
        if _details_are_hard_feasible(candidate.details)
    ]
    return sorted(feasible_candidates, key=lambda candidate: _candidate_score(candidate.details))[:limit]


def _build_mixed_family_candidates(
    instance: Any,
    seed_genome: Dict[str, np.ndarray],
    sequence_candidates: Sequence[AuditCandidate],
    assignment_candidates: Sequence[AuditCandidate],
    max_candidates: int = MAX_MIXED_CANDIDATES,
) -> List[AuditCandidate]:
    """Compose one bounded sequence move with one bounded assignment move."""
    seen = {_seed_genome_signature(seed_genome)}
    accepted: List[AuditCandidate] = []

    for seq_candidate in _best_candidates_for_mixed(sequence_candidates):
        for assign_candidate in _best_candidates_for_mixed(assignment_candidates):
            mixed = _clone_seed_genome(seed_genome)
            mixed["sequence"] = np.array(seq_candidate.genome["sequence"], copy=True)
            mixed["machines"] = np.array(assign_candidate.genome["machines"], copy=True)
            mixed["workers"] = np.array(assign_candidate.genome["workers"], copy=True)
            mixed["modes"] = np.array(assign_candidate.genome["modes"], copy=True)
            signature = _seed_genome_signature(mixed)
            if signature in seen:
                continue
            seen.add(signature)
            accepted.append(
                AuditCandidate(
                    name=f"mixed::{seq_candidate.name}+{assign_candidate.name}",
                    family=MIXED_FAMILY_NAME,
                    move_type="sequence_plus_assignment",
                    genome=mixed,
                    details=_evaluate_genome(instance, mixed),
                )
            )
            if len(accepted) >= max_candidates:
                return accepted

    return accepted


def _seed_candidates_for_instance(instance: Any) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
    """Collect deterministic accepted warm-start seeds and aligned acceptance diagnostics."""
    accepted, diagnostics = _collect_seed_genome_candidates(
        instance,
        include_tardiness_variants=False,
    )
    accepted_diagnostics = [diagnostic for diagnostic in diagnostics if diagnostic.get("status") == "accepted"]
    if len(accepted) != len(accepted_diagnostics):
        raise ValueError(
            "Accepted seed diagnostics do not align with accepted seed genomes "
            f"for instance {getattr(instance, 'instance_id', '<unknown>')}"
        )
    return accepted, accepted_diagnostics


def _best_generation0_seed(instance: Any) -> Dict[str, Any]:
    """Reconstruct and return the best deterministic generation-0 hard-feasible seed."""
    accepted, diagnostics = _seed_candidates_for_instance(instance)
    evaluated = []
    for genome, diagnostic in zip(accepted, diagnostics):
        details = _evaluate_genome(instance, genome)
        evaluated.append(
            {
                "source_rule": diagnostic.get("source_rule"),
                "genome": genome,
                "details": details,
            }
        )

    hard_feasible = [
        candidate
        for candidate in evaluated
        if _details_are_hard_feasible(candidate["details"])
    ]
    if not hard_feasible:
        raise ValueError(f"No hard-feasible warm-start seeds for instance {getattr(instance, 'instance_id', '<unknown>')}")

    best = min(hard_feasible, key=lambda candidate: _candidate_score(candidate["details"]))
    return {
        "best_source_rule": best["source_rule"],
        "best_genome": best["genome"],
        "best_details": best["details"],
        "accepted_seeds": [
            {
                "source_rule": candidate["source_rule"],
                "genome_signature": _signature_list(_seed_genome_signature(candidate["genome"])),
                **_compact_details(candidate["details"]),
            }
            for candidate in hard_feasible
        ],
    }


def _instance_audit_result(
    benchmark_path: str,
    baseline_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit one benchmark instance around its best generation-0 hard-feasible seed."""
    instance = load_benchmark(benchmark_path)
    seed_info = _best_generation0_seed(instance)
    best_seed_genome = seed_info["best_genome"]
    best_seed_details = seed_info["best_details"]

    sequence_candidates = _build_sequence_family_candidates(instance, best_seed_genome, best_seed_details)
    assignment_candidates = _build_assignment_family_candidates(instance, best_seed_genome, best_seed_details)
    offset_candidates = _build_offset_family_candidates(instance, best_seed_genome, best_seed_details)
    mixed_candidates = _build_mixed_family_candidates(
        instance,
        best_seed_genome,
        sequence_candidates,
        assignment_candidates,
    )

    return {
        "instance": Path(benchmark_path).stem,
        "benchmark_path": benchmark_path,
        "baseline_truth": baseline_truth,
        "best_generation0_seed": {
            "source_rule": seed_info["best_source_rule"],
            "genome_signature": _signature_list(_seed_genome_signature(best_seed_genome)),
            **_compact_details(best_seed_details),
        },
        "accepted_seed_pool": seed_info["accepted_seeds"],
        "neighborhood_families": {
            SEQUENCE_FAMILY_NAME: _summarize_family(
                SEQUENCE_FAMILY_NAME,
                best_seed_genome,
                best_seed_details,
                sequence_candidates,
            ),
            ASSIGNMENT_FAMILY_NAME: _summarize_family(
                ASSIGNMENT_FAMILY_NAME,
                best_seed_genome,
                best_seed_details,
                assignment_candidates,
            ),
            OFFSET_FAMILY_NAME: _summarize_family(
                OFFSET_FAMILY_NAME,
                best_seed_genome,
                best_seed_details,
                offset_candidates,
            ),
            MIXED_FAMILY_NAME: _summarize_family(
                MIXED_FAMILY_NAME,
                best_seed_genome,
                best_seed_details,
                mixed_candidates,
            ),
        },
    }


def _aggregate_family_summaries(instance_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate family summaries across all audited instances."""
    aggregate: Dict[str, Dict[str, Any]] = {}
    family_names = (
        SEQUENCE_FAMILY_NAME,
        ASSIGNMENT_FAMILY_NAME,
        OFFSET_FAMILY_NAME,
        MIXED_FAMILY_NAME,
    )

    for family_name in family_names:
        family_results = [result["neighborhood_families"][family_name] for result in instance_results]
        aggregate[family_name] = {
            "instances": len(family_results),
            "total_candidates_evaluated": sum(item["candidates_evaluated"] for item in family_results),
            "total_hard_feasible_count": sum(item["hard_feasible_count"] for item in family_results),
            "instances_with_improvement_over_best_seed": sum(
                1 for item in family_results if item["improves_over_best_seed_count"] > 0
            ),
            "best_delta_weighted_tardiness": min(
                (
                    item["improves_over_best_seed_best_delta_weighted_tardiness"]
                    for item in family_results
                    if item["improves_over_best_seed_best_delta_weighted_tardiness"] is not None
                ),
                default=None,
            ),
            "best_delta_n_tardy_jobs": min(
                (
                    item["improves_over_best_seed_best_delta_n_tardy_jobs"]
                    for item in family_results
                    if item["improves_over_best_seed_best_delta_n_tardy_jobs"] is not None
                ),
                default=None,
            ),
            "median_collision_rate": _median_or_none(
                [item["collision"]["decoded_collision_rate"] for item in family_results]
            ),
        }

    return aggregate


def _discover_benchmark_files(
    benchmark_dir: str,
    benchmark_path: Optional[str] = None,
) -> List[str]:
    """Return the benchmark files covered by the audit."""
    if benchmark_path:
        return [benchmark_path]
    return sorted(
        os.path.join(benchmark_dir, name)
        for name in os.listdir(benchmark_dir)
        if name.endswith(".json")
    )


def run_representation_audit(
    benchmark_dir: str,
    output_path: str,
    seed: int,
    baseline_artifact: str,
    reference_artifact: Optional[str] = None,
    benchmark_path: Optional[str] = None,
    command: str = "",
) -> Dict[str, Any]:
    """Run the offline representation audit across one benchmark slice."""
    baseline_truth_table = _load_baseline_truth_table(
        baseline_artifact_path=baseline_artifact,
        reference_artifact_path=reference_artifact,
    )
    truth_by_instance = {
        entry["instance"]: entry
        for entry in baseline_truth_table["per_instance"]
    }

    benchmark_files = _discover_benchmark_files(benchmark_dir, benchmark_path=benchmark_path)
    results = [
        _instance_audit_result(
            benchmark_path=filepath,
            baseline_truth=truth_by_instance.get(Path(filepath).stem, {}),
        )
        for filepath in benchmark_files
    ]

    return {
        "provenance": _build_provenance(
            benchmark_dir=benchmark_dir,
            output_path=output_path,
            baseline_artifact=baseline_artifact,
            reference_artifact=reference_artifact,
            seed=seed,
            command=command or "python -m experiments.analyze_nsga_representation",
        ),
        "baseline_truth_table": baseline_truth_table,
        "results": results,
        "aggregate_summary": _aggregate_family_summaries(results),
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the offline representation audit."""
    parser = argparse.ArgumentParser(description="Offline NSGA representation bottleneck audit")
    parser.add_argument("--benchmark-dir", type=str, default="benchmarks/small", help="Directory containing benchmark JSON files")
    parser.add_argument("--benchmark-path", type=str, default=None, help="Optional single benchmark JSON file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic audit seed for provenance")
    parser.add_argument(
        "--baseline-artifact",
        type=str,
        default=str(DEFAULT_BASELINE_ARTIFACT),
        help="Reference comparison artifact used to freeze the baseline truth table",
    )
    parser.add_argument(
        "--reference-artifact",
        type=str,
        default=str(DEFAULT_REFERENCE_ARTIFACT),
        help="Optional secondary artifact kept only for comparison context",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    artifact = run_representation_audit(
        benchmark_dir=args.benchmark_dir,
        output_path=args.output,
        seed=args.seed,
        baseline_artifact=args.baseline_artifact,
        reference_artifact=args.reference_artifact,
        benchmark_path=args.benchmark_path,
        command=" ".join(sys.argv),
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"Saved representation audit to {args.output}")


if __name__ == "__main__":
    main()
