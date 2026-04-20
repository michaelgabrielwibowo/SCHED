import csv
import json
from pathlib import Path

import pytest

try:
    from ..interfaces import (
        build_schedule_audit,
        export_schedule_artifacts,
        import_instance_from_dict,
    )
    from ..sfjssp_model import Schedule
except ImportError:  # pragma: no cover - supports repo-root imports
    from interfaces import (
        build_schedule_audit,
        export_schedule_artifacts,
        import_instance_from_dict,
    )
    from sfjssp_model import Schedule


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "interfaces"
ADAPTER_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "interfaces_adapters"


def _load_fixture(name: str) -> dict:
    with (FIXTURE_ROOT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_adapter_fixture(name: str) -> dict:
    with (ADAPTER_FIXTURE_ROOT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_feasible_multi_job_schedule():
    imported = import_instance_from_dict(_load_fixture("valid_multi_job.json"))
    schedule = Schedule(instance_id=imported.instance.instance_id)
    schedule.add_operation(
        job_id=0,
        op_id=0,
        machine_id=0,
        worker_id=1,
        mode_id=0,
        start_time=3.0,
        completion_time=21.0,
        processing_time=18.0,
        setup_time=3.0,
    )
    schedule.add_operation(
        job_id=0,
        op_id=1,
        machine_id=1,
        worker_id=0,
        mode_id=0,
        start_time=28.0,
        completion_time=44.0,
        processing_time=16.0,
        setup_time=4.0,
        transport_time=4.0,
    )
    schedule.add_operation(
        job_id=1,
        op_id=0,
        machine_id=1,
        worker_id=0,
        mode_id=1,
        start_time=48.0,
        completion_time=68.0,
        processing_time=20.0,
        setup_time=4.0,
        transport_time=3.0,
    )
    return imported, schedule


def test_export_schedule_artifacts_writes_expected_files_and_manifest(tmp_path):
    imported, schedule = _build_feasible_multi_job_schedule()
    audit_payload = build_schedule_audit(
        schedule,
        imported.instance,
        provenance={
            "solver": "greedy:spt",
            "objective": "makespan",
            "seed": 11,
            "runtime_seconds": 0.25,
            "input_source_id": "fixture:multi",
            "input_schema": imported.schema,
        },
        id_maps=imported.id_maps,
        input_schema=imported.schema,
        input_source_id="fixture:multi",
    )
    audit_payload["provenance"]["generated_at"] = "2026-04-19T00:00:00+00:00"

    output_dir = tmp_path / "exports"
    manifest = export_schedule_artifacts(
        output_dir,
        schedule,
        imported.instance,
        audit_payload=audit_payload,
        id_maps=imported.id_maps,
    )

    expected_files = {
        "run_manifest.json",
        "schedule.json",
        "operations.csv",
        "machine_timeline.csv",
        "worker_timeline.csv",
        "violations.json",
        "violations.csv",
        "machine_calendar.csv",
        "worker_calendar.csv",
        "events.csv",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}

    manifest_json = _read_json(output_dir / "run_manifest.json")
    schedule_json = _read_json(output_dir / "schedule.json")
    violations_json = _read_json(output_dir / "violations.json")

    assert manifest == manifest_json
    assert manifest_json["manifest_schema"] == "schedule_run_manifest_v2"
    assert manifest_json["schedule_schema"] == "schedule_export_bundle_v2"
    assert manifest_json["audit_schema"] == "schedule_audit_v2"
    assert manifest_json["calibration"]["status"] == "fully_synthetic"
    assert manifest_json["feasible"] is True
    assert manifest_json["scheduled_operation_count"] == 3
    assert manifest_json["provenance"]["solver"] == "greedy:spt"
    assert manifest_json["artifacts"]["schedule_json"] == "schedule.json"

    assert schedule_json["schedule_schema"] == "schedule_export_bundle_v2"
    assert schedule_json["calibration"]["status"] == "fully_synthetic"
    assert schedule_json["feasible"] is True
    assert schedule_json["metrics"]["makespan"] == pytest.approx(68.0)
    assert len(schedule_json["operations"]) == 3
    assert len(schedule_json["machine_timelines"]) == 2
    assert len(schedule_json["worker_timelines"]) == 2
    assert len(schedule_json["resource_calendars"]["machines"]) == 2
    assert len(schedule_json["resource_calendars"]["workers"]) == 2
    assert all(
        machine["unavailability_windows"] == []
        for machine in schedule_json["resource_calendars"]["machines"]
    )
    assert all(
        worker["shift_windows"] == [] and worker["unavailability_windows"] == []
        for worker in schedule_json["resource_calendars"]["workers"]
    )
    assert schedule_json["canonical_events"] == []

    assert violations_json["audit_schema"] == "schedule_audit_v2"
    assert violations_json["hard_violation_count"] == 0


def test_export_schedule_artifacts_are_deterministic_for_fixed_audit_payload(tmp_path):
    imported, schedule = _build_feasible_multi_job_schedule()
    audit_payload = build_schedule_audit(
        schedule,
        imported.instance,
        provenance={"solver": "greedy:spt", "seed": 5},
        id_maps=imported.id_maps,
        input_schema=imported.schema,
    )
    audit_payload["provenance"]["generated_at"] = "2026-04-19T00:00:00+00:00"
    audit_payload["provenance"]["git_commit"] = "fixed-commit"
    audit_payload["provenance"]["git_dirty"] = False
    audit_payload["provenance"]["git_status_short"] = []

    dir_one = tmp_path / "one"
    dir_two = tmp_path / "two"
    export_schedule_artifacts(
        dir_one,
        schedule,
        imported.instance,
        audit_payload=audit_payload,
        id_maps=imported.id_maps,
    )
    export_schedule_artifacts(
        dir_two,
        schedule,
        imported.instance,
        audit_payload=audit_payload,
        id_maps=imported.id_maps,
    )

    for name in [
        "run_manifest.json",
        "schedule.json",
        "operations.csv",
        "machine_timeline.csv",
        "worker_timeline.csv",
        "violations.json",
        "violations.csv",
        "machine_calendar.csv",
        "worker_calendar.csv",
        "events.csv",
    ]:
        assert (dir_one / name).read_text(encoding="utf-8") == (dir_two / name).read_text(encoding="utf-8")


def test_export_schedule_csvs_match_schedule_json_and_use_stable_order(tmp_path):
    imported, schedule = _build_feasible_multi_job_schedule()
    audit_payload = build_schedule_audit(
        schedule,
        imported.instance,
        provenance={"solver": "greedy:spt"},
        id_maps=imported.id_maps,
        input_schema=imported.schema,
    )
    audit_payload["provenance"]["generated_at"] = "2026-04-19T00:00:00+00:00"

    export_schedule_artifacts(
        tmp_path,
        schedule,
        imported.instance,
        audit_payload=audit_payload,
        id_maps=imported.id_maps,
    )

    schedule_json = _read_json(tmp_path / "schedule.json")
    operations_csv = _read_csv(tmp_path / "operations.csv")
    machine_csv = _read_csv(tmp_path / "machine_timeline.csv")
    worker_csv = _read_csv(tmp_path / "worker_timeline.csv")
    machine_calendar_csv = _read_csv(tmp_path / "machine_calendar.csv")
    worker_calendar_csv = _read_csv(tmp_path / "worker_calendar.csv")
    events_csv = _read_csv(tmp_path / "events.csv")

    assert [row["job_external_id"] for row in operations_csv] == ["J1", "J1", "J2"]
    assert [row["operation_external_id"] for row in operations_csv] == ["cut", "finish", "paint"]

    schedule_ops = schedule_json["operations"]
    assert len(schedule_ops) == len(operations_csv)
    for csv_row, json_row in zip(operations_csv, schedule_ops):
        assert csv_row["job_external_id"] == json_row["job_external_id"]
        assert csv_row["operation_external_id"] == json_row["operation_external_id"]
        assert float(csv_row["start_time"]) == pytest.approx(json_row["start_time"])
        assert float(csv_row["completion_time"]) == pytest.approx(json_row["completion_time"])

    assert [row["resource_type"] for row in machine_csv] == ["machine", "machine", "machine"]
    assert [row["resource_id"] for row in machine_csv] == ["0", "1", "1"]
    assert [row["resource_type"] for row in worker_csv] == ["worker", "worker", "worker"]
    assert [row["resource_id"] for row in worker_csv] == ["0", "0", "1"]
    assert machine_calendar_csv == []
    assert worker_calendar_csv == []
    assert events_csv == []


def test_export_schedule_artifacts_write_violation_exports_for_infeasible_schedule(tmp_path):
    imported = import_instance_from_dict(_load_fixture("valid_minimal.json"))
    imported.instance.workers[0].min_rest_fraction = 0.0
    imported.instance.workers[0].ocra_max_per_shift = 999.0
    schedule = Schedule(instance_id=imported.instance.instance_id)
    schedule.add_operation(
        job_id=0,
        op_id=0,
        machine_id=0,
        worker_id=0,
        mode_id=0,
        start_time=1.0,
        completion_time=13.0,
        processing_time=12.0,
        setup_time=2.0,
    )

    audit_payload = build_schedule_audit(
        schedule,
        imported.instance,
        provenance={"solver": "manual"},
        id_maps=imported.id_maps,
        input_schema=imported.schema,
    )
    audit_payload["provenance"]["generated_at"] = "2026-04-19T00:00:00+00:00"

    manifest = export_schedule_artifacts(
        tmp_path,
        schedule,
        imported.instance,
        audit_payload=audit_payload,
        id_maps=imported.id_maps,
    )

    violations_json = _read_json(tmp_path / "violations.json")
    violations_csv = _read_csv(tmp_path / "violations.csv")

    assert manifest["feasible"] is False
    assert violations_json["hard_violation_count"] == 1
    assert violations_json["hard_violations"][0]["code"] == "setup_gap"
    assert len(violations_csv) == 1
    assert violations_csv[0]["code"] == "setup_gap"
    assert json.loads(violations_csv[0]["details_json"])["required_setup_time"] == pytest.approx(2.0)


def test_export_schedule_artifacts_include_calendar_and_event_exports_for_v2_inputs(tmp_path):
    imported = import_instance_from_dict(_load_fixture("valid_with_calendar_events_v2.json"))
    imported.instance.workers[0].min_rest_fraction = 0.0
    imported.instance.workers[0].ocra_max_per_shift = 999.0
    schedule = Schedule(instance_id=imported.instance.instance_id)
    schedule.add_operation(
        job_id=0,
        op_id=0,
        machine_id=0,
        worker_id=0,
        mode_id=0,
        start_time=2.0,
        completion_time=14.0,
        processing_time=12.0,
        setup_time=2.0,
    )

    audit_payload = build_schedule_audit(
        schedule,
        imported.instance,
        provenance={"solver": "greedy:spt"},
        id_maps=imported.id_maps,
        input_schema=imported.schema,
    )
    audit_payload["provenance"]["generated_at"] = "2026-04-19T00:00:00+00:00"

    export_schedule_artifacts(
        tmp_path,
        schedule,
        imported.instance,
        audit_payload=audit_payload,
        id_maps=imported.id_maps,
    )

    schedule_json = _read_json(tmp_path / "schedule.json")
    machine_calendar_csv = _read_csv(tmp_path / "machine_calendar.csv")
    worker_calendar_csv = _read_csv(tmp_path / "worker_calendar.csv")
    events_csv = _read_csv(tmp_path / "events.csv")

    assert schedule_json["resource_calendars"]["machines"][0]["unavailability_windows"][0]["reason"] == "maintenance"
    assert schedule_json["resource_calendars"]["workers"][0]["shift_windows"][0]["shift_label"] == "day"
    assert [row["reason"] for row in machine_calendar_csv] == ["maintenance", "breakdown"]
    assert [row["entry_type"] for row in worker_calendar_csv] == [
        "shift_window",
        "unavailability_window",
        "unavailability_window",
        "unavailability_window",
    ]
    assert [row["event_id"] for row in events_csv] == [
        "worker-absence-000002",
        "machine-breakdown-000001",
    ]


def test_export_schedule_artifacts_preserve_adapter_and_site_profile_provenance(tmp_path):
    imported = import_instance_from_dict(
        _load_adapter_fixture("valid_plant_tables_v1.json"),
        adapter_name="plant_tables_v1",
        site_profile_name="light_assembly_demo_v1",
    )
    schedule = Schedule(instance_id=imported.instance.instance_id)
    schedule.add_operation(
        job_id=0,
        op_id=0,
        machine_id=0,
        worker_id=0,
        mode_id=0,
        start_time=4.0,
        completion_time=14.0,
        processing_time=10.0,
        setup_time=4.0,
        transport_time=2.0,
    )
    audit_payload = build_schedule_audit(
        schedule,
        imported.instance,
        provenance={"solver": "greedy:spt", **imported.provenance},
        id_maps=imported.id_maps,
        input_schema=imported.schema,
    )
    audit_payload["provenance"]["generated_at"] = "2026-04-19T00:00:00+00:00"

    manifest = export_schedule_artifacts(
        tmp_path,
        schedule,
        imported.instance,
        audit_payload=audit_payload,
        id_maps=imported.id_maps,
    )

    assert manifest["provenance"]["adapter"]["adapter_name"] == "plant_tables_v1"
    assert manifest["calibration"]["status"] == "calibrated_synthetic"
    assert manifest["provenance"]["site_profile"]["profile_id"] == "light_assembly_demo_v1"
    assert manifest["provenance"]["parameter_sources"]["site_profile_applied"] is True


def test_export_schedule_artifacts_rejects_audit_payload_without_calibration_provenance(tmp_path):
    imported, schedule = _build_feasible_multi_job_schedule()
    audit_payload = build_schedule_audit(
        schedule,
        imported.instance,
        provenance={"solver": "greedy:spt"},
        id_maps=imported.id_maps,
        input_schema=imported.schema,
    )
    audit_payload["provenance"].pop("calibration")

    with pytest.raises(ValueError, match="Audit provenance must include calibration"):
        export_schedule_artifacts(
            tmp_path,
            schedule,
            imported.instance,
            audit_payload=audit_payload,
            id_maps=imported.id_maps,
        )
