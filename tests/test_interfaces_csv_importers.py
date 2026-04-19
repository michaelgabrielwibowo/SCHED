import shutil
from pathlib import Path

import pytest

try:
    from ..interfaces import (
        EXTERNAL_INPUT_SCHEMA,
        InterfaceValidationError,
        load_instance_from_csv_bundle,
        load_instance_from_json,
    )
except ImportError:  # pragma: no cover - supports repo-root imports
    from interfaces import (
        EXTERNAL_INPUT_SCHEMA,
        InterfaceValidationError,
        load_instance_from_csv_bundle,
        load_instance_from_json,
    )


JSON_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "interfaces"
CSV_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "interfaces_csv"


def _assert_issue(exc: InterfaceValidationError, code: str, path_fragment: str) -> None:
    assert any(
        issue.code == code and path_fragment in issue.path for issue in exc.issues
    ), exc.issues


def test_load_instance_from_csv_bundle_builds_minimal_instance():
    imported = load_instance_from_csv_bundle(CSV_FIXTURE_ROOT / "valid_minimal")

    assert imported.schema == EXTERNAL_INPUT_SCHEMA
    assert imported.instance.instance_id == "EXT_MINIMAL"
    assert imported.instance.n_jobs == 1
    assert imported.instance.n_machines == 1
    assert imported.instance.n_workers == 1
    assert imported.id_maps.reverse_machines[0] == "M0"
    assert imported.id_maps.reverse_workers[0] == "W0"
    assert imported.id_maps.reverse_jobs[0] == "J0"
    assert imported.instance.jobs[0].operations[0].processing_times == {0: {0: 12.0}}


def test_csv_bundle_matches_equivalent_json_fixture():
    imported_csv = load_instance_from_csv_bundle(CSV_FIXTURE_ROOT / "valid_multi_job")
    imported_json = load_instance_from_json(JSON_FIXTURE_ROOT / "valid_multi_job.json")

    assert imported_csv.normalized_payload == imported_json.normalized_payload
    assert imported_csv.instance.to_dict() == imported_json.instance.to_dict()
    assert imported_csv.id_maps.reverse_operations == imported_json.id_maps.reverse_operations


def test_csv_importer_rejects_missing_required_table(tmp_path):
    fixture_dir = tmp_path / "missing_table"
    shutil.copytree(CSV_FIXTURE_ROOT / "valid_minimal", fixture_dir)
    (fixture_dir / "operations.csv").unlink()

    with pytest.raises(InterfaceValidationError) as excinfo:
        load_instance_from_csv_bundle(fixture_dir)

    _assert_issue(excinfo.value, "missing_required", "$.csv_bundle.operations")


def test_csv_importer_rejects_unknown_column_in_strict_mode(tmp_path):
    fixture_dir = tmp_path / "unknown_column"
    shutil.copytree(CSV_FIXTURE_ROOT / "valid_minimal", fixture_dir)
    workers_path = fixture_dir / "workers.csv"
    workers_path.write_text(
        "id,name,mystery\nW0,Operator 0,boom\n",
        encoding="utf-8",
    )

    with pytest.raises(InterfaceValidationError) as excinfo:
        load_instance_from_csv_bundle(fixture_dir)

    _assert_issue(excinfo.value, "unknown_field", "$.csv_bundle.workers.mystery")
