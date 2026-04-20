import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "interfaces"
ADAPTER_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "interfaces_adapters"
CSV_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "interfaces_csv"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "interfaces.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_validate_input_happy_path():
    result = _run_cli(
        "validate-input",
        "--input",
        str(FIXTURE_ROOT / "valid_minimal.json"),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["command"] == "validate-input"
    assert payload["schema"] == "sfjssp_external_v1"
    assert payload["input_format"] == "json"
    assert payload["instance_id"] == "EXT_MINIMAL"
    assert payload["provenance"]["calibration"]["status"] == "fully_synthetic"
    assert payload["counts"]["operations"] == 1


def test_cli_validate_input_accepts_v2_json():
    result = _run_cli(
        "validate-input",
        "--input",
        str(FIXTURE_ROOT / "valid_with_calendar_events_v2.json"),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["command"] == "validate-input"
    assert payload["schema"] == "sfjssp_external_v2"
    assert payload["input_format"] == "json"
    assert payload["instance_id"] == "EXT_V2"


def test_cli_validate_input_accepts_adapter_and_site_profile():
    result = _run_cli(
        "validate-input",
        "--input",
        str(ADAPTER_FIXTURE_ROOT / "valid_plant_tables_v1.json"),
        "--adapter",
        "plant_tables_v1",
        "--site-profile",
        "light_assembly_demo_v1",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["schema"] == "sfjssp_external_v2"
    assert payload["provenance"]["adapter"]["adapter_name"] == "plant_tables_v1"
    assert payload["provenance"]["site_profile"]["profile_id"] == "light_assembly_demo_v1"


def test_cli_validate_input_accepts_csv_bundle():
    result = _run_cli(
        "validate-input",
        "--input",
        str(CSV_FIXTURE_ROOT / "valid_minimal"),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["input_format"] == "csv_bundle"
    assert payload["instance_id"] == "EXT_MINIMAL"


def test_cli_validate_input_accepts_v2_csv_bundle():
    result = _run_cli(
        "validate-input",
        "--input",
        str(CSV_FIXTURE_ROOT / "valid_with_calendar_events_v2"),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["schema"] == "sfjssp_external_v2"
    assert payload["input_format"] == "csv_bundle"
    assert payload["instance_id"] == "EXT_V2"


def test_cli_validate_input_returns_structured_validation_error(tmp_path):
    bad_input = tmp_path / "bad.json"
    bad_input.write_text(
        json.dumps(
            {
                "schema": "sfjssp_external_v1",
                "metadata": {"instance_id": "BAD"},
                "machines": [],
                "workers": [],
                "jobs": [],
                "mystery": {},
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli("validate-input", "--input", str(bad_input))

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 2
    assert payload["error_class"] == "validation_error"
    assert payload["code"] == "input_validation_failed"
    assert any(issue["code"] == "unknown_field" for issue in payload["details"])


def test_cli_run_happy_path_exports_documented_bundle(tmp_path):
    output_dir = tmp_path / "out"
    result = _run_cli(
        "run",
        "--input",
        str(FIXTURE_ROOT / "valid_minimal.json"),
        "--solver",
        "greedy:spt",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["command"] == "run"
    assert payload["solver"] == "greedy:spt"
    assert payload["input_format"] == "json"
    assert Path(payload["manifest_path"]).exists()

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    schedule = json.loads((output_dir / "schedule.json").read_text(encoding="utf-8"))
    violations = json.loads((output_dir / "violations.json").read_text(encoding="utf-8"))

    assert manifest["manifest_schema"] == "schedule_run_manifest_v2"
    assert schedule["schedule_schema"] == "schedule_export_bundle_v2"
    assert violations["audit_schema"] == "schedule_audit_v2"
    assert manifest["calibration"]["status"] == "fully_synthetic"
    assert {
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
    } == {path.name for path in output_dir.iterdir()}


def test_cli_solve_uses_documented_default_output_convention(tmp_path):
    output_root = tmp_path / "runs"
    result = _run_cli(
        "solve",
        "--input",
        str(FIXTURE_ROOT / "valid_minimal.json"),
        "--solver",
        "greedy:spt",
        "--output-root",
        str(output_root),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "solve"
    assert payload["manifest_complete"] is True

    run_dir = output_root / "EXT_MINIMAL" / "greedy-spt"
    assert Path(payload["output_dir"]) == run_dir
    assert (run_dir / "run_manifest.json").exists()


def test_cli_run_accepts_csv_bundle(tmp_path):
    output_dir = tmp_path / "csv-out"
    result = _run_cli(
        "run",
        "--input",
        str(CSV_FIXTURE_ROOT / "valid_minimal"),
        "--solver",
        "greedy:spt",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["input_format"] == "csv_bundle"
    assert Path(payload["manifest_path"]).exists()
    assert (output_dir / "schedule.json").exists()


def test_cli_run_accepts_v2_csv_bundle(tmp_path):
    output_dir = tmp_path / "csv-v2-out"
    result = _run_cli(
        "run",
        "--input",
        str(CSV_FIXTURE_ROOT / "valid_with_calendar_events_v2"),
        "--solver",
        "greedy:spt",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["input_format"] == "csv_bundle"
    assert payload["instance_id"] == "EXT_V2"
    assert Path(payload["manifest_path"]).exists()


def test_cli_run_accepts_v2_json(tmp_path):
    output_dir = tmp_path / "json-v2-out"
    result = _run_cli(
        "run",
        "--input",
        str(FIXTURE_ROOT / "valid_with_calendar_events_v2.json"),
        "--solver",
        "greedy:spt",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["input_format"] == "json"
    assert payload["instance_id"] == "EXT_V2"
    assert Path(payload["manifest_path"]).exists()


def test_cli_run_accepts_adapter_and_site_profile(tmp_path):
    output_dir = tmp_path / "adapter-out"
    result = _run_cli(
        "run",
        "--input",
        str(ADAPTER_FIXTURE_ROOT / "valid_plant_tables_v1.json"),
        "--adapter",
        "plant_tables_v1",
        "--site-profile",
        "light_assembly_demo_v1",
        "--solver",
        "greedy:spt",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["provenance"]["adapter"]["adapter_name"] == "plant_tables_v1"
    assert payload["provenance"]["calibration"]["status"] == "calibrated_synthetic"
    assert payload["provenance"]["site_profile"]["profile_id"] == "light_assembly_demo_v1"

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["adapter"]["adapter_name"] == "plant_tables_v1"
    assert manifest["calibration"]["status"] == "calibrated_synthetic"
    assert manifest["provenance"]["site_profile"]["profile_id"] == "light_assembly_demo_v1"
    assert manifest["provenance"]["external_schema"] == "sfjssp_external_v2"


def test_cli_audit_summarizes_existing_run_directory(tmp_path):
    output_dir = tmp_path / "audit-run"
    run_result = _run_cli(
        "solve",
        "--input",
        str(FIXTURE_ROOT / "valid_minimal.json"),
        "--solver",
        "greedy:spt",
        "--output-dir",
        str(output_dir),
    )
    assert run_result.returncode == 0, run_result.stderr

    audit_result = _run_cli(
        "audit",
        "--run-dir",
        str(output_dir),
        "--max-violations",
        "2",
    )

    assert audit_result.returncode == 0, audit_result.stderr
    payload = json.loads(audit_result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "audit"
    assert payload["manifest_complete"] is True
    assert payload["artifacts"]["schedule_json"] == "schedule.json"
    assert len(payload["top_hard_violations"]) <= 2


def test_cli_export_copies_spreadsheet_facing_bundle(tmp_path):
    output_dir = tmp_path / "source-run"
    target_dir = tmp_path / "handoff"
    run_result = _run_cli(
        "solve",
        "--input",
        str(FIXTURE_ROOT / "valid_minimal.json"),
        "--solver",
        "greedy:spt",
        "--output-dir",
        str(output_dir),
    )
    assert run_result.returncode == 0, run_result.stderr

    export_result = _run_cli(
        "export",
        "--run-dir",
        str(output_dir),
        "--target-dir",
        str(target_dir),
    )

    assert export_result.returncode == 0, export_result.stderr
    payload = json.loads(export_result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "export"
    assert payload["manifest_complete"] is True
    assert "operations.csv" in payload["copied_files"]
    assert (target_dir / "run_manifest.json").exists()
    assert (target_dir / "violations.csv").exists()
    assert (target_dir / "events.csv").exists()


def test_cli_run_rejects_unsupported_solver(tmp_path):
    result = _run_cli(
        "run",
        "--input",
        str(FIXTURE_ROOT / "valid_minimal.json"),
        "--solver",
        "nsga:default",
        "--output-dir",
        str(tmp_path / "out"),
    )

    assert result.returncode == 3
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 3
    assert payload["error_class"] == "unsupported_request"
    assert payload["code"] == "unsupported_solver"
    assert "supported_solvers" in payload["details"]


def test_cli_rejects_adapter_for_csv_bundle():
    result = _run_cli(
        "validate-input",
        "--input",
        str(CSV_FIXTURE_ROOT / "valid_minimal"),
        "--adapter",
        "plant_tables_v1",
    )

    assert result.returncode == 3
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 3
    assert payload["error_class"] == "unsupported_request"
    assert payload["code"] == "unsupported_adapter_input"


def test_cli_audit_rejects_incomplete_run_directory(tmp_path):
    output_dir = tmp_path / "broken-run"
    run_result = _run_cli(
        "solve",
        "--input",
        str(FIXTURE_ROOT / "valid_minimal.json"),
        "--solver",
        "greedy:spt",
        "--output-dir",
        str(output_dir),
    )
    assert run_result.returncode == 0, run_result.stderr

    (output_dir / "events.csv").unlink()

    audit_result = _run_cli("audit", "--run-dir", str(output_dir))

    assert audit_result.returncode == 6
    payload = json.loads(audit_result.stderr)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 6
    assert payload["error_class"] == "artifact_error"
    assert payload["code"] == "manifest_incomplete"


def test_cli_audit_rejects_missing_run_directory(tmp_path):
    missing_dir = tmp_path / "does-not-exist"

    audit_result = _run_cli("audit", "--run-dir", str(missing_dir))

    assert audit_result.returncode == 6
    payload = json.loads(audit_result.stderr)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 6
    assert payload["error_class"] == "artifact_error"
    assert payload["code"] == "run_directory_not_found"
