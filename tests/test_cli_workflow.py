import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "interfaces"
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
    assert payload["command"] == "validate-input"
    assert payload["schema"] == "sfjssp_external_v1"
    assert payload["input_format"] == "json"
    assert payload["instance_id"] == "EXT_MINIMAL"
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
    assert payload["command"] == "validate-input"
    assert payload["schema"] == "sfjssp_external_v2"
    assert payload["input_format"] == "json"
    assert payload["instance_id"] == "EXT_V2"


def test_cli_validate_input_accepts_csv_bundle():
    result = _run_cli(
        "validate-input",
        "--input",
        str(CSV_FIXTURE_ROOT / "valid_minimal"),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
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
    assert payload["command"] == "run"
    assert payload["solver"] == "greedy:spt"
    assert payload["input_format"] == "json"
    assert Path(payload["manifest_path"]).exists()

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    schedule = json.loads((output_dir / "schedule.json").read_text(encoding="utf-8"))
    violations = json.loads((output_dir / "violations.json").read_text(encoding="utf-8"))

    assert manifest["manifest_schema"] == "schedule_run_manifest_v1"
    assert schedule["schedule_schema"] == "schedule_export_bundle_v1"
    assert violations["audit_schema"] == "schedule_audit_v1"
    assert {
        "run_manifest.json",
        "schedule.json",
        "operations.csv",
        "machine_timeline.csv",
        "worker_timeline.csv",
        "violations.json",
        "violations.csv",
    } == {path.name for path in output_dir.iterdir()}


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
    assert payload["input_format"] == "csv_bundle"
    assert payload["instance_id"] == "EXT_V2"
    assert Path(payload["manifest_path"]).exists()


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
    assert payload["code"] == "unsupported_solver"
    assert "supported_solvers" in payload["details"]
