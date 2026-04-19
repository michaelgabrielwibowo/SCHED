"""
Stable JSON and CSV schedule exports built from the canonical audit payload.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .audit import SCHEDULE_AUDIT_SCHEMA, build_schedule_audit
from .types import IdentifierMaps

try:
    from ..sfjssp_model.instance import SFJSSPInstance
    from ..sfjssp_model.schedule import Schedule
except ImportError:  # pragma: no cover - supports repo-root imports
    from sfjssp_model.instance import SFJSSPInstance
    from sfjssp_model.schedule import Schedule


RUN_MANIFEST_SCHEMA = "schedule_run_manifest_v1"
SCHEDULE_EXPORT_SCHEMA = "schedule_export_bundle_v1"

OPERATION_FIELDNAMES = [
    "job_id",
    "job_external_id",
    "op_id",
    "operation_external_id",
    "machine_id",
    "machine_external_id",
    "worker_id",
    "worker_external_id",
    "mode_id",
    "mode_external_id",
    "start_time",
    "completion_time",
    "processing_time",
    "setup_time",
    "transport_time",
    "job_arrival_time",
    "job_due_date",
    "job_weight",
    "job_completion_time",
    "job_tardiness",
]

TIMELINE_FIELDNAMES = [
    "resource_type",
    "resource_id",
    "resource_external_id",
    "job_id",
    "job_external_id",
    "op_id",
    "operation_external_id",
    "machine_id",
    "machine_external_id",
    "worker_id",
    "worker_external_id",
    "mode_id",
    "mode_external_id",
    "start_time",
    "completion_time",
    "processing_time",
    "setup_time",
    "transport_time",
]

VIOLATION_FIELDNAMES = [
    "code",
    "message",
    "job_id",
    "job_external_id",
    "op_id",
    "operation_external_id",
    "machine_id",
    "machine_external_id",
    "worker_id",
    "worker_external_id",
    "details_json",
]


def export_schedule_artifacts(
    output_dir: Any,
    schedule: Schedule,
    instance: SFJSSPInstance,
    *,
    audit_payload: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    id_maps: Optional[IdentifierMaps] = None,
    input_schema: Optional[str] = None,
    input_source_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Write the stable export bundle for one schedule run."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    audit_payload = audit_payload or build_schedule_audit(
        schedule,
        instance,
        provenance=provenance,
        id_maps=id_maps,
        input_schema=input_schema,
        input_source_id=input_source_id,
    )
    schedule_bundle = build_schedule_export_bundle(
        schedule,
        instance,
        audit_payload=audit_payload,
        id_maps=id_maps,
    )

    operations_rows = build_operation_rows(schedule, instance, id_maps=id_maps)
    machine_timeline_rows = build_machine_timeline_rows(schedule, id_maps=id_maps)
    worker_timeline_rows = build_worker_timeline_rows(schedule, id_maps=id_maps)
    violation_rows = build_violation_rows(audit_payload)

    schedule_path = target_dir / "schedule.json"
    manifest_path = target_dir / "run_manifest.json"
    operations_path = target_dir / "operations.csv"
    machine_timeline_path = target_dir / "machine_timeline.csv"
    worker_timeline_path = target_dir / "worker_timeline.csv"
    violations_json_path = target_dir / "violations.json"
    violations_csv_path = target_dir / "violations.csv"

    _write_json(schedule_path, schedule_bundle)
    _write_json(violations_json_path, audit_payload)
    _write_csv(operations_path, OPERATION_FIELDNAMES, operations_rows)
    _write_csv(machine_timeline_path, TIMELINE_FIELDNAMES, machine_timeline_rows)
    _write_csv(worker_timeline_path, TIMELINE_FIELDNAMES, worker_timeline_rows)
    _write_csv(violations_csv_path, VIOLATION_FIELDNAMES, violation_rows)

    manifest = {
        "manifest_schema": RUN_MANIFEST_SCHEMA,
        "schedule_schema": SCHEDULE_EXPORT_SCHEMA,
        "audit_schema": audit_payload.get("audit_schema", SCHEDULE_AUDIT_SCHEMA),
        "instance_id": instance.instance_id,
        "instance_name": instance.instance_name,
        "feasible": audit_payload["feasible"],
        "hard_violation_count": audit_payload["hard_violation_count"],
        "scheduled_operation_count": len(schedule.scheduled_ops),
        "job_count": instance.n_jobs,
        "machine_count": instance.n_machines,
        "worker_count": instance.n_workers,
        "artifacts": {
            "schedule_json": schedule_path.name,
            "operations_csv": operations_path.name,
            "machine_timeline_csv": machine_timeline_path.name,
            "worker_timeline_csv": worker_timeline_path.name,
            "violations_json": violations_json_path.name,
            "violations_csv": violations_csv_path.name,
        },
        "provenance": dict(audit_payload["provenance"]),
    }
    _write_json(manifest_path, manifest)
    return manifest


def build_schedule_export_bundle(
    schedule: Schedule,
    instance: SFJSSPInstance,
    *,
    audit_payload: Dict[str, Any],
    id_maps: Optional[IdentifierMaps] = None,
) -> Dict[str, Any]:
    """Build the stable JSON schedule bundle."""

    operations = build_operation_rows(schedule, instance, id_maps=id_maps)
    return {
        "schedule_schema": SCHEDULE_EXPORT_SCHEMA,
        "instance_id": instance.instance_id,
        "instance_name": instance.instance_name,
        "feasible": audit_payload["feasible"],
        "hard_violation_count": audit_payload["hard_violation_count"],
        "metrics": dict(audit_payload["soft_summary"]["metrics"]),
        "energy_breakdown": dict(audit_payload["soft_summary"]["energy_breakdown"]),
        "ergonomic_metrics": dict(audit_payload["soft_summary"]["ergonomic_metrics"]),
        "fatigue_metrics": dict(audit_payload["soft_summary"]["fatigue_metrics"]),
        "robustness_metrics": dict(audit_payload["soft_summary"]["robustness_metrics"]),
        "job_tardiness": list(audit_payload["job_tardiness"]),
        "operations": operations,
        "machine_timelines": _group_rows_by_resource(
            build_machine_timeline_rows(schedule, id_maps=id_maps),
            resource_key="machine_id",
            external_key="machine_external_id",
        ),
        "worker_timelines": _group_rows_by_resource(
            build_worker_timeline_rows(schedule, id_maps=id_maps),
            resource_key="worker_id",
            external_key="worker_external_id",
        ),
        "provenance": dict(audit_payload["provenance"]),
    }


def build_operation_rows(
    schedule: Schedule,
    instance: SFJSSPInstance,
    *,
    id_maps: Optional[IdentifierMaps] = None,
) -> List[Dict[str, Any]]:
    """Return one deterministic flat row per scheduled operation."""

    rows: List[Dict[str, Any]] = []
    for (job_id, op_id), sched_op in sorted(schedule.scheduled_ops.items()):
        job = instance.get_job(job_id)
        if job is None:
            continue
        rows.append(
            {
                "job_id": job_id,
                "job_external_id": _external_job_id(job_id, id_maps),
                "op_id": op_id,
                "operation_external_id": _external_operation_id(job_id, op_id, id_maps),
                "machine_id": sched_op.machine_id,
                "machine_external_id": _external_machine_id(sched_op.machine_id, id_maps),
                "worker_id": sched_op.worker_id,
                "worker_external_id": _external_worker_id(sched_op.worker_id, id_maps),
                "mode_id": sched_op.mode_id,
                "mode_external_id": _external_mode_id(sched_op.machine_id, sched_op.mode_id, id_maps),
                "start_time": sched_op.start_time,
                "completion_time": sched_op.completion_time,
                "processing_time": sched_op.processing_time,
                "setup_time": sched_op.setup_time,
                "transport_time": sched_op.transport_time,
                "job_arrival_time": job.arrival_time,
                "job_due_date": job.due_date,
                "job_weight": job.weight,
                "job_completion_time": schedule.get_job_completion_time(job_id, instance),
                "job_tardiness": schedule.get_job_tardiness(job_id, instance),
            }
        )
    return rows


def build_machine_timeline_rows(
    schedule: Schedule,
    *,
    id_maps: Optional[IdentifierMaps] = None,
) -> List[Dict[str, Any]]:
    """Return deterministic machine timeline rows."""

    rows: List[Dict[str, Any]] = []
    for machine_id in sorted(schedule.machine_schedules):
        machine_sched = schedule.machine_schedules[machine_id]
        for sched_op in sorted(
            machine_sched.operations,
            key=lambda op: (op.start_time, op.completion_time, op.job_id, op.op_id),
        ):
            rows.append(
                _timeline_row(
                    "machine",
                    machine_id,
                    _external_machine_id(machine_id, id_maps),
                    sched_op,
                    id_maps=id_maps,
                )
            )
    return rows


def build_worker_timeline_rows(
    schedule: Schedule,
    *,
    id_maps: Optional[IdentifierMaps] = None,
) -> List[Dict[str, Any]]:
    """Return deterministic worker timeline rows."""

    rows: List[Dict[str, Any]] = []
    for worker_id in sorted(schedule.worker_schedules):
        worker_sched = schedule.worker_schedules[worker_id]
        for sched_op in sorted(
            worker_sched.operations,
            key=lambda op: (op.start_time, op.completion_time, op.job_id, op.op_id),
        ):
            rows.append(
                _timeline_row(
                    "worker",
                    worker_id,
                    _external_worker_id(worker_id, id_maps),
                    sched_op,
                    id_maps=id_maps,
                )
            )
    return rows


def build_violation_rows(audit_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return deterministic flat violation rows."""

    rows: List[Dict[str, Any]] = []
    hard_violations = sorted(
        audit_payload.get("hard_violations", []),
        key=lambda violation: (
            str(violation.get("code", "")),
            -1 if violation.get("job_id") is None else int(violation["job_id"]),
            -1 if violation.get("op_id") is None else int(violation["op_id"]),
            -1 if violation.get("machine_id") is None else int(violation["machine_id"]),
            -1 if violation.get("worker_id") is None else int(violation["worker_id"]),
            str(violation.get("message", "")),
        ),
    )
    for violation in hard_violations:
        rows.append(
            {
                "code": violation.get("code"),
                "message": violation.get("message"),
                "job_id": violation.get("job_id"),
                "job_external_id": violation.get("job_external_id"),
                "op_id": violation.get("op_id"),
                "operation_external_id": violation.get("operation_external_id"),
                "machine_id": violation.get("machine_id"),
                "machine_external_id": violation.get("machine_external_id"),
                "worker_id": violation.get("worker_id"),
                "worker_external_id": violation.get("worker_external_id"),
                "details_json": json.dumps(
                    violation.get("details", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return rows


def _group_rows_by_resource(
    rows: Iterable[Dict[str, Any]],
    *,
    resource_key: str,
    external_key: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        resource_id = int(row[resource_key])
        bucket = grouped.setdefault(
            resource_id,
            {
                resource_key: resource_id,
                external_key: row[external_key],
                "operations": [],
            },
        )
        bucket["operations"].append(
            {
                "job_id": row["job_id"],
                "job_external_id": row["job_external_id"],
                "op_id": row["op_id"],
                "operation_external_id": row["operation_external_id"],
                "machine_id": row["machine_id"],
                "machine_external_id": row["machine_external_id"],
                "worker_id": row["worker_id"],
                "worker_external_id": row["worker_external_id"],
                "mode_id": row["mode_id"],
                "mode_external_id": row["mode_external_id"],
                "start_time": row["start_time"],
                "completion_time": row["completion_time"],
                "processing_time": row["processing_time"],
                "setup_time": row["setup_time"],
                "transport_time": row["transport_time"],
            }
        )
    return [grouped[key] for key in sorted(grouped)]


def _timeline_row(
    resource_type: str,
    resource_id: int,
    resource_external_id: Optional[Any],
    sched_op,
    *,
    id_maps: Optional[IdentifierMaps],
) -> Dict[str, Any]:
    return {
        "resource_type": resource_type,
        "resource_id": resource_id,
        "resource_external_id": resource_external_id,
        "job_id": sched_op.job_id,
        "job_external_id": _external_job_id(sched_op.job_id, id_maps),
        "op_id": sched_op.op_id,
        "operation_external_id": _external_operation_id(sched_op.job_id, sched_op.op_id, id_maps),
        "machine_id": sched_op.machine_id,
        "machine_external_id": _external_machine_id(sched_op.machine_id, id_maps),
        "worker_id": sched_op.worker_id,
        "worker_external_id": _external_worker_id(sched_op.worker_id, id_maps),
        "mode_id": sched_op.mode_id,
        "mode_external_id": _external_mode_id(sched_op.machine_id, sched_op.mode_id, id_maps),
        "start_time": sched_op.start_time,
        "completion_time": sched_op.completion_time,
        "processing_time": sched_op.processing_time,
        "setup_time": sched_op.setup_time,
        "transport_time": sched_op.transport_time,
    }


def _external_job_id(job_id: int, id_maps: Optional[IdentifierMaps]) -> Optional[Any]:
    return id_maps.reverse_jobs.get(job_id) if id_maps is not None else None


def _external_operation_id(
    job_id: int,
    op_id: int,
    id_maps: Optional[IdentifierMaps],
) -> Optional[Any]:
    return id_maps.reverse_operations.get((job_id, op_id)) if id_maps is not None else None


def _external_machine_id(machine_id: int, id_maps: Optional[IdentifierMaps]) -> Optional[Any]:
    return id_maps.reverse_machines.get(machine_id) if id_maps is not None else None


def _external_worker_id(worker_id: int, id_maps: Optional[IdentifierMaps]) -> Optional[Any]:
    return id_maps.reverse_workers.get(worker_id) if id_maps is not None else None


def _external_mode_id(
    machine_id: int,
    mode_id: int,
    id_maps: Optional[IdentifierMaps],
) -> Optional[Any]:
    if id_maps is None:
        return None
    return id_maps.reverse_machine_modes.get((machine_id, mode_id))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
