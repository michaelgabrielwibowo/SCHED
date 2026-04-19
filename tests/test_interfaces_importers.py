import json
from copy import deepcopy
from pathlib import Path

import pytest

try:
    from ..interfaces import (
        EXTERNAL_INPUT_SCHEMA,
        InterfaceValidationError,
        import_instance_from_dict,
        load_instance_from_json,
    )
except ImportError:  # pragma: no cover - supports repo-root imports
    from interfaces import (
        EXTERNAL_INPUT_SCHEMA,
        InterfaceValidationError,
        import_instance_from_dict,
        load_instance_from_json,
    )


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "interfaces"


def _load_fixture(name: str) -> dict:
    with (FIXTURE_ROOT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert_issue(exc: InterfaceValidationError, code: str, path_fragment: str) -> None:
    assert any(
        issue.code == code and path_fragment in issue.path for issue in exc.issues
    ), exc.issues


def test_load_instance_from_json_builds_minimal_instance():
    imported = load_instance_from_json(FIXTURE_ROOT / "valid_minimal.json")

    assert imported.schema == EXTERNAL_INPUT_SCHEMA
    assert imported.instance.instance_id == "EXT_MINIMAL"
    assert imported.instance.n_jobs == 1
    assert imported.instance.n_machines == 1
    assert imported.instance.n_workers == 1
    assert imported.instance.period_clock.period_duration == 480.0
    assert imported.id_maps.reverse_machines[0] == "M0"
    assert imported.id_maps.reverse_workers[0] == "W0"
    assert imported.id_maps.reverse_jobs[0] == "J0"
    assert imported.id_maps.reverse_machine_modes[(0, 0)] == "standard"
    assert imported.instance.jobs[0].operations[0].processing_times == {0: {0: 12.0}}
    assert imported.instance.workers[0].eligible_operations == {(0, 0)}
    assert imported.instance.ergonomic_risk_map[(0, 0)] == pytest.approx(0.002)


def test_import_multi_job_fixture_normalizes_ids_and_defaults():
    imported = import_instance_from_dict(_load_fixture("valid_multi_job.json"))

    assert [job["id"] for job in imported.normalized_payload["jobs"]] == ["J1", "J2"]
    assert [machine["id"] for machine in imported.normalized_payload["machines"]] == ["M1", "M2"]
    assert [worker["id"] for worker in imported.normalized_payload["workers"]] == ["W1", "W2"]

    assert imported.instance.instance_type.value == "dynamic"
    assert imported.instance.electricity_prices == {0: 0.11, 60: 0.17}
    assert imported.instance.default_electricity_price == pytest.approx(0.13)
    assert imported.instance.carbon_emission_factor == pytest.approx(0.45)
    assert imported.instance.auxiliary_power_total == pytest.approx(18.0)
    assert imported.instance.ergonomic_risk_map[(0, 1)] == pytest.approx(0.0015)
    assert imported.id_maps.jobs["str:J1"] == 0
    assert imported.id_maps.jobs["str:J2"] == 1
    assert imported.id_maps.machine_modes[("str:M2", "str:eco")] == 0
    assert imported.id_maps.machine_modes[("str:M2", "str:fast")] == 1
    assert imported.instance.workers[0].eligible_operations == {(0, 1), (1, 0)}
    assert imported.instance.workers[1].eligible_operations == {(0, 0), (0, 1), (1, 0)}


def test_importer_rejects_unknown_top_level_field():
    payload = _load_fixture("valid_minimal.json")
    payload["mystery"] = {}

    with pytest.raises(InterfaceValidationError) as excinfo:
        import_instance_from_dict(payload)

    _assert_issue(excinfo.value, "unknown_field", "$.mystery")


def test_importer_rejects_reserved_unsupported_section():
    payload = _load_fixture("valid_minimal.json")
    payload["events"] = {"absence_probability": 0.1}

    with pytest.raises(InterfaceValidationError) as excinfo:
        import_instance_from_dict(payload)

    _assert_issue(excinfo.value, "unsupported_section", "$.events")


def test_importer_rejects_unknown_worker_reference():
    payload = _load_fixture("valid_minimal.json")
    payload["jobs"][0]["operations"][0]["eligible_workers"] = ["UNKNOWN"]

    with pytest.raises(InterfaceValidationError) as excinfo:
        import_instance_from_dict(payload)

    _assert_issue(excinfo.value, "invalid_reference", "$.jobs[0].operations[0].eligible_workers")


def test_importer_normalizes_equivalent_orderings_deterministically():
    unsorted_payload = _load_fixture("valid_multi_job.json")
    sorted_payload = deepcopy(unsorted_payload)
    sorted_payload["machines"] = list(reversed(sorted_payload["machines"]))
    sorted_payload["workers"] = list(reversed(sorted_payload["workers"]))
    sorted_payload["jobs"] = list(reversed(sorted_payload["jobs"]))

    imported_unsorted = import_instance_from_dict(unsorted_payload)
    imported_sorted = import_instance_from_dict(sorted_payload)

    assert imported_unsorted.normalized_payload == imported_sorted.normalized_payload
    assert imported_unsorted.instance.to_dict() == imported_sorted.instance.to_dict()
