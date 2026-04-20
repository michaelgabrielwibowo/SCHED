"""
Versioned schedule audit payloads over the canonical schedule oracle.
"""

from __future__ import annotations

from datetime import datetime, timezone
import subprocess
from typing import Any, Dict, List, Optional

from .types import IdentifierMaps

try:
    from ..sfjssp_model.instance import SFJSSPInstance, build_public_calibration_record
    from ..sfjssp_model.schedule import ConstraintViolation, Schedule
except ImportError:  # pragma: no cover - supports repo-root imports
    from sfjssp_model.instance import SFJSSPInstance, build_public_calibration_record
    from sfjssp_model.schedule import ConstraintViolation, Schedule


SCHEDULE_AUDIT_SCHEMA = "schedule_audit_v2"
MACHINE_CONFLICT_CODES = {
    "machine_overlap",
    "setup_gap",
    "transport_gap",
    "ineligible_machine_assignment",
    "period_violation",
    "machine_unavailable",
    "machine_maintenance_violation",
    "machine_outage_violation",
    "machine_breakdown_violation",
}
WORKER_CONFLICT_CODES = {
    "worker_overlap",
    "rest_violation",
    "ocra_violation",
    "ineligible_worker_assignment",
    "arrival_violation",
    "worker_unavailable",
    "worker_off_shift_violation",
    "worker_absence_violation",
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
    calibration = instance.build_calibration_record()
    hard_violations = [
        _serialize_violation(violation, id_maps)
        for violation in schedule.constraint_violation_details
    ]

    return {
        "audit_schema": SCHEDULE_AUDIT_SCHEMA,
        "instance_id": instance.instance_id,
        "instance_name": instance.instance_name,
        "calibration": calibration,
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
        "resource_calendars": _build_resource_calendar_summary(instance, id_maps),
        "canonical_events": _build_canonical_event_summary(instance, id_maps),
        "provenance": _build_audit_provenance(
            schedule,
            instance,
            calibration=calibration,
            provenance=provenance,
            input_schema=input_schema,
            input_source_id=input_source_id,
        ),
    }


def _build_audit_provenance(
    schedule: Schedule,
    instance: SFJSSPInstance,
    *,
    calibration: Dict[str, Any],
    provenance: Optional[Dict[str, Any]],
    input_schema: Optional[str],
    input_source_id: Optional[str],
) -> Dict[str, Any]:
    provenance = dict(provenance or {})
    schedule_metadata = dict(schedule.metadata)
    git_status_short = _get_git_status_short()
    audit_provenance = {
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
        "calibration": calibration,
        "git_commit": _get_git_commit(),
        "git_dirty": bool(git_status_short),
        "git_status_short": git_status_short,
        "schedule_metadata": schedule_metadata,
    }
    reserved_keys = {
        "solver",
        "objective",
        "seed",
        "runtime_seconds",
        "time_seconds",
        "input_source_id",
        "input_schema",
        "calibration",
    }
    for key, value in provenance.items():
        if key not in reserved_keys:
            audit_provenance[key] = _json_ready(value)
    return audit_provenance


def validate_schedule_audit_payload(audit_payload: Dict[str, Any]) -> None:
    """Reject audit payloads that omit the enforced calibration provenance contract."""

    if audit_payload.get("audit_schema") != SCHEDULE_AUDIT_SCHEMA:
        raise ValueError(
            f"Expected audit_schema {SCHEDULE_AUDIT_SCHEMA!r}, got "
            f"{audit_payload.get('audit_schema')!r}."
        )
    calibration = audit_payload.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Audit payload must include a top-level calibration record.")
    canonical_calibration = build_public_calibration_record(
        calibration.get("status"),
        calibration.get("justification", ""),
        calibration.get("sources", []),
    )
    if dict(calibration) != canonical_calibration:
        raise ValueError(
            "Audit payload calibration record does not match the canonical calibration contract."
        )

    provenance = audit_payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Audit payload must include provenance.")
    provenance_calibration = provenance.get("calibration")
    if not isinstance(provenance_calibration, dict):
        raise ValueError("Audit provenance must include calibration.")
    if dict(provenance_calibration) != canonical_calibration:
        raise ValueError(
            "Audit provenance calibration record does not match the top-level calibration record."
        )


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


def _build_resource_calendar_summary(
    instance: SFJSSPInstance,
    id_maps: Optional[IdentifierMaps],
) -> Dict[str, List[Dict[str, Any]]]:
    machines: List[Dict[str, Any]] = []
    for machine in sorted(instance.machines, key=lambda item: item.machine_id):
        machines.append(
            {
                "machine_id": machine.machine_id,
                "machine_external_id": id_maps.reverse_machines.get(machine.machine_id) if id_maps is not None else None,
                "unavailability_windows": [
                    _serialize_window(window)
                    for window in instance.get_machine_unavailability(machine.machine_id)
                ],
            }
        )

    workers: List[Dict[str, Any]] = []
    for worker in sorted(instance.workers, key=lambda item: item.worker_id):
        workers.append(
            {
                "worker_id": worker.worker_id,
                "worker_external_id": id_maps.reverse_workers.get(worker.worker_id) if id_maps is not None else None,
                "shift_windows": [
                    {
                        "start_time": window.start_time,
                        "end_time": window.end_time,
                        "shift_label": window.shift_label,
                        "details": _json_ready(window.details),
                    }
                    for window in worker.shift_windows
                ],
                "unavailability_windows": [
                    _serialize_window(window)
                    for window in instance.get_worker_unavailability(worker.worker_id)
                ],
            }
        )

    return {"machines": machines, "workers": workers}


def _build_canonical_event_summary(
    instance: SFJSSPInstance,
    id_maps: Optional[IdentifierMaps],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in instance.iter_canonical_events():
        resource_type = "machine" if event["event_type"] == "machine_breakdown" else "worker"
        resource_id = int(event["resource_id"])
        if resource_type == "machine":
            resource_external_id = (
                id_maps.reverse_machines.get(resource_id) if id_maps is not None else None
            )
        else:
            resource_external_id = (
                id_maps.reverse_workers.get(resource_id) if id_maps is not None else None
            )
        rows.append(
            {
                "event_type": event["event_type"],
                "event_id": event.get("event_id"),
                "resource_type": resource_type,
                "resource_id": resource_id,
                "resource_external_id": resource_external_id,
                "start_time": event["start_time"],
                "end_time": event["end_time"],
                "payload": _json_ready(event.get("payload", {})),
            }
        )
    return rows


def _serialize_window(window: Any) -> Dict[str, Any]:
    return {
        "start_time": window.start_time,
        "end_time": window.end_time,
        "reason": window.reason,
        "source": window.source,
        "event_id": getattr(window, "event_id", ""),
        "details": _json_ready(window.details),
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
