"""
Versioned schedule audit payloads over the canonical schedule oracle.
"""

from __future__ import annotations

from datetime import datetime, timezone
import subprocess
from typing import Any, Dict, List, Optional

from .types import IdentifierMaps

try:
    from ..sfjssp_model.instance import SFJSSPInstance
    from ..sfjssp_model.schedule import ConstraintViolation, Schedule
except ImportError:  # pragma: no cover - supports repo-root imports
    from sfjssp_model.instance import SFJSSPInstance
    from sfjssp_model.schedule import ConstraintViolation, Schedule


SCHEDULE_AUDIT_SCHEMA = "schedule_audit_v1"
MACHINE_CONFLICT_CODES = {
    "machine_overlap",
    "setup_gap",
    "transport_gap",
    "ineligible_machine_assignment",
    "period_violation",
}
WORKER_CONFLICT_CODES = {
    "worker_overlap",
    "rest_violation",
    "ocra_violation",
    "ineligible_worker_assignment",
    "arrival_violation",
}


def build_schedule_audit(
    schedule: Schedule,
    instance: SFJSSPInstance,
    *,
    provenance: Optional[Dict[str, Any]] = None,
    id_maps: Optional[IdentifierMaps] = None,
    input_schema: Optional[str] = None,
    input_source_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one machine-readable audit payload for a schedule."""

    feasible = schedule.check_feasibility(instance)
    metrics = schedule.evaluate(instance)
    hard_violations = [
        _serialize_violation(violation, id_maps)
        for violation in schedule.constraint_violation_details
    ]

    return {
        "audit_schema": SCHEDULE_AUDIT_SCHEMA,
        "instance_id": instance.instance_id,
        "instance_name": instance.instance_name,
        "feasible": feasible,
        "hard_violation_count": len(hard_violations),
        "hard_violation_counts": _count_violation_codes(hard_violations),
        "hard_violations": hard_violations,
        "soft_summary": {
            "metrics": dict(metrics),
            "energy_breakdown": dict(schedule.energy_breakdown),
            "ergonomic_metrics": dict(schedule.ergonomic_metrics),
            "fatigue_metrics": dict(schedule.fatigue_metrics),
            "robustness_metrics": dict(schedule.robustness_metrics),
            "tardiness": {
                "total_tardiness": metrics.get("total_tardiness", 0.0),
                "weighted_tardiness": metrics.get("weighted_tardiness", 0.0),
                "n_tardy_jobs": metrics.get("n_tardy_jobs", 0),
            },
        },
        "job_tardiness": _build_job_tardiness_summary(schedule, instance, id_maps),
        "resource_conflicts": _build_resource_conflict_summary(hard_violations, id_maps),
        "provenance": _build_audit_provenance(
            schedule,
            instance,
            provenance=provenance,
            input_schema=input_schema,
            input_source_id=input_source_id,
        ),
    }


def _build_audit_provenance(
    schedule: Schedule,
    instance: SFJSSPInstance,
    *,
    provenance: Optional[Dict[str, Any]],
    input_schema: Optional[str],
    input_source_id: Optional[str],
) -> Dict[str, Any]:
    provenance = dict(provenance or {})
    schedule_metadata = dict(schedule.metadata)
    git_status_short = _get_git_status_short()
    return {
        "audit_schema": SCHEDULE_AUDIT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "solver": provenance.get("solver") or schedule_metadata.get("solver") or "unknown",
        "objective": provenance.get("objective") or schedule_metadata.get("solver_objective"),
        "seed": provenance.get("seed", schedule_metadata.get("seed")),
        "runtime_seconds": provenance.get(
            "runtime_seconds",
            provenance.get(
                "time_seconds",
                schedule_metadata.get("runtime_seconds", schedule_metadata.get("time_seconds")),
            ),
        ),
        "input_source_id": provenance.get("input_source_id") or input_source_id or instance.instance_id,
        "input_schema": provenance.get("input_schema") or input_schema,
        "git_commit": _get_git_commit(),
        "git_dirty": bool(git_status_short),
        "git_status_short": git_status_short,
        "schedule_metadata": schedule_metadata,
    }


def _serialize_violation(
    violation: ConstraintViolation,
    id_maps: Optional[IdentifierMaps],
) -> Dict[str, Any]:
    payload = violation.to_dict()
    payload["job_external_id"] = (
        id_maps.reverse_jobs.get(violation.job_id)
        if id_maps is not None and violation.job_id is not None
        else None
    )
    payload["operation_external_id"] = (
        id_maps.reverse_operations.get((violation.job_id, violation.op_id))
        if id_maps is not None
        and violation.job_id is not None
        and violation.op_id is not None
        else None
    )
    payload["machine_external_id"] = (
        id_maps.reverse_machines.get(violation.machine_id)
        if id_maps is not None and violation.machine_id is not None
        else None
    )
    payload["worker_external_id"] = (
        id_maps.reverse_workers.get(violation.worker_id)
        if id_maps is not None and violation.worker_id is not None
        else None
    )
    payload["details"] = _json_ready(payload.get("details", {}))
    return payload


def _build_job_tardiness_summary(
    schedule: Schedule,
    instance: SFJSSPInstance,
    id_maps: Optional[IdentifierMaps],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for job in instance.jobs:
        completion_time = schedule.get_job_completion_time(job.job_id, instance)
        tardiness = schedule.get_job_tardiness(job.job_id, instance)
        rows.append(
            {
                "job_id": job.job_id,
                "job_external_id": id_maps.reverse_jobs.get(job.job_id) if id_maps is not None else None,
                "arrival_time": job.arrival_time,
                "due_date": job.due_date,
                "completion_time": completion_time,
                "tardiness": tardiness,
                "weighted_tardiness": tardiness * job.weight,
                "is_tardy": tardiness > 0.0,
            }
        )
    return rows


def _build_resource_conflict_summary(
    hard_violations: List[Dict[str, Any]],
    id_maps: Optional[IdentifierMaps],
) -> Dict[str, List[Dict[str, Any]]]:
    machine_buckets: Dict[int, Dict[str, Any]] = {}
    worker_buckets: Dict[int, Dict[str, Any]] = {}

    for violation in hard_violations:
        code = violation["code"]
        if violation.get("machine_id") is not None and code in MACHINE_CONFLICT_CODES:
            machine_id = int(violation["machine_id"])
            bucket = machine_buckets.setdefault(
                machine_id,
                {
                    "machine_id": machine_id,
                    "machine_external_id": id_maps.reverse_machines.get(machine_id) if id_maps is not None else None,
                    "hard_violation_count": 0,
                    "violation_counts": {},
                    "operations": [],
                },
            )
            bucket["hard_violation_count"] += 1
            bucket["violation_counts"][code] = bucket["violation_counts"].get(code, 0) + 1
            _append_unique_operation(bucket["operations"], violation)

        if violation.get("worker_id") is not None and code in WORKER_CONFLICT_CODES:
            worker_id = int(violation["worker_id"])
            bucket = worker_buckets.setdefault(
                worker_id,
                {
                    "worker_id": worker_id,
                    "worker_external_id": id_maps.reverse_workers.get(worker_id) if id_maps is not None else None,
                    "hard_violation_count": 0,
                    "violation_counts": {},
                    "operations": [],
                },
            )
            bucket["hard_violation_count"] += 1
            bucket["violation_counts"][code] = bucket["violation_counts"].get(code, 0) + 1
            _append_unique_operation(bucket["operations"], violation)

    return {
        "machines": [machine_buckets[key] for key in sorted(machine_buckets)],
        "workers": [worker_buckets[key] for key in sorted(worker_buckets)],
    }


def _append_unique_operation(target: List[Dict[str, Any]], violation: Dict[str, Any]) -> None:
    if violation.get("job_id") is None or violation.get("op_id") is None:
        return
    row = {
        "job_id": violation["job_id"],
        "job_external_id": violation.get("job_external_id"),
        "op_id": violation["op_id"],
        "operation_external_id": violation.get("operation_external_id"),
    }
    if row not in target:
        target.append(row)


def _count_violation_codes(hard_violations: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for violation in hard_violations:
        code = str(violation["code"])
        counts[code] = counts.get(code, 0) + 1
    return counts


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _get_git_commit() -> Optional[str]:
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
