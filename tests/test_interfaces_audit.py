import json
from pathlib import Path

import pytest

try:
    from ..interfaces import build_schedule_audit, import_instance_from_dict
    from ..sfjssp_model import Job, Machine, MachineMode, Operation, Schedule, SFJSSPInstance, Worker
    from ..sfjssp_model.instance import InstanceLabel
except ImportError:  # pragma: no cover - supports repo-root imports
    from interfaces import build_schedule_audit, import_instance_from_dict
    from sfjssp_model import Job, Machine, MachineMode, Operation, Schedule, SFJSSPInstance, Worker
    from sfjssp_model.instance import InstanceLabel


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "interfaces"
ADAPTER_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "interfaces_adapters"


def _load_fixture(name: str) -> dict:
    with (FIXTURE_ROOT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_adapter_fixture(name: str) -> dict:
    with (ADAPTER_FIXTURE_ROOT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_schedule_audit_feasible_imported_schedule_includes_external_ids_and_provenance():
    imported = import_instance_from_dict(_load_fixture("valid_minimal.json"))
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

    audit = build_schedule_audit(
        schedule,
        imported.instance,
        provenance={
            "solver": "greedy:spt",
            "objective": "makespan",
            "seed": 7,
            "runtime_seconds": 0.12,
            "input_source_id": "fixture:valid_minimal",
            "input_schema": imported.schema,
        },
        id_maps=imported.id_maps,
        input_schema=imported.schema,
    )

    assert audit["audit_schema"] == "schedule_audit_v2"
    assert audit["calibration"]["status"] == "fully_synthetic"
    assert audit["feasible"] is True
    assert audit["hard_violation_count"] == 0
    assert audit["hard_violations"] == []
    assert audit["soft_summary"]["metrics"]["total_energy"] > 0.0
    assert audit["job_tardiness"][0]["job_external_id"] == "J0"
    assert audit["job_tardiness"][0]["weighted_tardiness"] == 0.0
    assert audit["resource_conflicts"] == {"machines": [], "workers": []}
    assert audit["resource_calendars"]["machines"] == [
        {
            "machine_id": 0,
            "machine_external_id": "M0",
            "unavailability_windows": [],
        }
    ]
    assert audit["resource_calendars"]["workers"] == [
        {
            "worker_id": 0,
            "worker_external_id": "W0",
            "shift_windows": [],
            "unavailability_windows": [],
        }
    ]
    assert audit["canonical_events"] == []
    assert audit["provenance"]["solver"] == "greedy:spt"
    assert audit["provenance"]["objective"] == "makespan"
    assert audit["provenance"]["seed"] == 7
    assert audit["provenance"]["runtime_seconds"] == pytest.approx(0.12)
    assert audit["provenance"]["input_schema"] == imported.schema
    assert audit["provenance"]["calibration"]["status"] == "fully_synthetic"
    assert "git_dirty" in audit["provenance"]
    assert "git_status_short" in audit["provenance"]


def test_schedule_audit_surfaces_v2_calendar_and_event_provenance():
    imported = import_instance_from_dict(_load_fixture("valid_with_calendar_events_v2.json"))
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

    audit = build_schedule_audit(
        schedule,
        imported.instance,
        provenance={"solver": "greedy:spt"},
        id_maps=imported.id_maps,
        input_schema=imported.schema,
    )

    assert audit["feasible"] is True
    assert audit["resource_calendars"]["machines"][0]["unavailability_windows"][0]["reason"] == "maintenance"
    assert audit["resource_calendars"]["workers"][0]["shift_windows"][0]["shift_label"] == "day"
    assert [event["event_id"] for event in audit["canonical_events"]] == [
        "worker-absence-000002",
        "machine-breakdown-000001",
    ]
    assert audit["canonical_events"][0]["resource_external_id"] == "W0"
    assert audit["canonical_events"][1]["resource_external_id"] == "M0"


def test_schedule_audit_preserves_import_adapter_and_site_profile_provenance():
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

    audit = build_schedule_audit(
        schedule,
        imported.instance,
        provenance={"solver": "greedy:spt", **imported.provenance},
        id_maps=imported.id_maps,
        input_schema=imported.schema,
    )

    assert audit["provenance"]["external_schema"] == imported.schema
    assert audit["provenance"]["calibration"]["status"] == "calibrated_synthetic"
    assert audit["provenance"]["adapter"]["adapter_name"] == "plant_tables_v1"
    assert audit["provenance"]["site_profile"]["profile_id"] == "light_assembly_demo_v1"


def test_schedule_audit_rejects_calibration_sensitive_claim_without_sources():
    instance = SFJSSPInstance(instance_id="AUDIT_BAD_CALIBRATION")
    instance.label = InstanceLabel.CALIBRATED_SYNTHETIC
    instance.label_justification = "Claimed calibration"
    instance.calibration_sources = []
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(Worker(worker_id=0, min_rest_fraction=0.0, ocra_max_per_shift=999.0))
    operation = Operation(
        job_id=0,
        op_id=0,
        processing_times={0: {0: 10.0}},
        eligible_machines={0},
        eligible_workers={0},
    )
    instance.add_job(Job(job_id=0, operations=[operation]))
    instance.ergonomic_risk_map[(0, 0)] = 0.0

    schedule = Schedule(instance_id=instance.instance_id)
    schedule.add_operation(
        job_id=0,
        op_id=0,
        machine_id=0,
        worker_id=0,
        mode_id=0,
        start_time=0.0,
        completion_time=10.0,
        processing_time=10.0,
    )

    with pytest.raises(ValueError, match="requires at least one calibration source"):
        build_schedule_audit(schedule, instance)


def test_schedule_audit_classifies_transport_gap():
    instance = SFJSSPInstance(instance_id="AUDIT_TRANSPORT")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_machine(Machine(machine_id=1, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(Worker(worker_id=0, min_rest_fraction=0.0, ocra_max_per_shift=999.0))
    first = Operation(
        job_id=0,
        op_id=0,
        processing_times={0: {0: 10.0}},
        transport_time=5.0,
        eligible_machines={0},
        eligible_workers={0},
    )
    second = Operation(
        job_id=0,
        op_id=1,
        processing_times={1: {0: 10.0}},
        eligible_machines={1},
        eligible_workers={0},
    )
    instance.add_job(Job(job_id=0, operations=[first, second]))
    instance.ergonomic_risk_map[(0, 0)] = 0.0
    instance.ergonomic_risk_map[(0, 1)] = 0.0

    schedule = Schedule(instance_id=instance.instance_id)
    schedule.add_operation(
        job_id=0,
        op_id=0,
        machine_id=0,
        worker_id=0,
        mode_id=0,
        start_time=0.0,
        completion_time=10.0,
        processing_time=10.0,
        transport_time=5.0,
    )
    schedule.add_operation(
        job_id=0,
        op_id=1,
        machine_id=1,
        worker_id=0,
        mode_id=0,
        start_time=14.0,
        completion_time=24.0,
        processing_time=10.0,
    )

    audit = build_schedule_audit(schedule, instance, provenance={"solver": "manual"})

    assert audit["feasible"] is False
    assert audit["hard_violation_counts"]["transport_gap"] == 1
    violation = audit["hard_violations"][0]
    assert violation["code"] == "transport_gap"
    assert violation["message"].startswith("Precedence violation")
    assert violation["details"]["transport_time"] == pytest.approx(5.0)
    assert violation["details"]["required_ready_time"] == pytest.approx(15.0)


def test_schedule_audit_summarizes_setup_rest_and_ocra_violations_by_resource():
    instance = SFJSSPInstance(instance_id="AUDIT_HUMAN")
    instance.add_machine(Machine(machine_id=0, setup_time=5.0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(
        Worker(
            worker_id=0,
            min_rest_fraction=0.25,
            ocra_max_per_shift=0.2,
        )
    )
    operation = Operation(
        job_id=0,
        op_id=0,
        processing_times={0: {0: 10.0}},
        eligible_machines={0},
        eligible_workers={0},
    )
    instance.add_job(Job(job_id=0, operations=[operation]))
    instance.ergonomic_risk_map[(0, 0)] = 0.05

    schedule = Schedule(instance_id=instance.instance_id)
    schedule.add_operation(
        job_id=0,
        op_id=0,
        machine_id=0,
        worker_id=0,
        mode_id=0,
        start_time=3.0,
        completion_time=13.0,
        processing_time=10.0,
        setup_time=5.0,
    )

    audit = build_schedule_audit(schedule, instance)

    assert audit["feasible"] is False
    assert audit["hard_violation_counts"]["setup_gap"] == 1
    assert audit["hard_violation_counts"]["rest_violation"] == 1
    assert audit["hard_violation_counts"]["ocra_violation"] == 1

    machine_conflict = audit["resource_conflicts"]["machines"][0]
    assert machine_conflict["machine_id"] == 0
    assert machine_conflict["violation_counts"]["setup_gap"] == 1
    assert machine_conflict["operations"] == [
        {
            "job_id": 0,
            "job_external_id": None,
            "op_id": 0,
            "operation_external_id": None,
        }
    ]

    worker_conflict = audit["resource_conflicts"]["workers"][0]
    assert worker_conflict["worker_id"] == 0
    assert worker_conflict["violation_counts"]["rest_violation"] == 1
    assert worker_conflict["violation_counts"]["ocra_violation"] == 1


def test_schedule_audit_keeps_tardy_schedule_feasible_and_reports_soft_metrics():
    instance = SFJSSPInstance(instance_id="AUDIT_TARDY")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(Worker(worker_id=0, min_rest_fraction=0.0, ocra_max_per_shift=999.0))
    operation = Operation(
        job_id=0,
        op_id=0,
        processing_times={0: {0: 20.0}},
        eligible_machines={0},
        eligible_workers={0},
    )
    instance.add_job(Job(job_id=0, due_date=10.0, weight=3.0, operations=[operation]))
    instance.ergonomic_risk_map[(0, 0)] = 0.0

    schedule = Schedule(instance_id=instance.instance_id)
    schedule.add_operation(
        job_id=0,
        op_id=0,
        machine_id=0,
        worker_id=0,
        mode_id=0,
        start_time=20.0,
        completion_time=40.0,
        processing_time=20.0,
    )

    audit = build_schedule_audit(schedule, instance)

    assert audit["feasible"] is True
    assert audit["hard_violation_count"] == 0
    assert audit["soft_summary"]["metrics"]["total_tardiness"] == pytest.approx(30.0)
    assert audit["soft_summary"]["metrics"]["weighted_tardiness"] == pytest.approx(90.0)
    assert audit["soft_summary"]["metrics"]["n_tardy_jobs"] == 1
    assert audit["job_tardiness"][0]["is_tardy"] is True
    assert audit["job_tardiness"][0]["weighted_tardiness"] == pytest.approx(90.0)
