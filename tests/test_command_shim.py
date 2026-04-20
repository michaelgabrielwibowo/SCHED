import json
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows command shim only",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "interfaces"


def _run_sched(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cmd", "/c", str(REPO_ROOT / "SCHED.cmd"), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_sched_cmd_without_arguments_shows_top_level_help():
    result = _run_sched()

    assert result.returncode == 0, result.stderr
    assert "usage: SCHED" in result.stdout
    assert "validate-input" in result.stdout


def test_sched_cmd_help_uses_sched_prog():
    result = _run_sched("--help")

    assert result.returncode == 0, result.stderr
    assert "usage: SCHED" in result.stdout
    assert "validate-input" in result.stdout


def test_sched_cmd_validate_input_happy_path():
    result = _run_sched(
        "validate-input",
        "--input",
        str(FIXTURE_ROOT / "valid_minimal.json"),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "validate-input"
    assert payload["instance_id"] == "EXT_MINIMAL"


def test_sched_cmd_solve_happy_path(tmp_path):
    output_dir = tmp_path / "sched-out"
    result = _run_sched(
        "solve",
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
    assert payload["command"] == "solve"
    assert (output_dir / "run_manifest.json").exists()


def test_sched_cmd_propagates_validation_error(tmp_path):
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

    result = _run_sched("validate-input", "--input", str(bad_input))

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert payload["exit_code"] == 2
    assert payload["code"] == "input_validation_failed"
