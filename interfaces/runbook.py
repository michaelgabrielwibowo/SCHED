"""
Operator-facing workflow contract for the external CLI surface.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .audit import SCHEDULE_AUDIT_SCHEMA, validate_schedule_audit_payload
from .exporters import RUN_MANIFEST_SCHEMA, SCHEDULE_EXPORT_SCHEMA


EXIT_SUCCESS = 0
EXIT_VALIDATION_ERROR = 2
EXIT_UNSUPPORTED_REQUEST = 3
EXIT_SOLVER_FAILURE = 4
EXIT_RUNTIME_ERROR = 5
EXIT_ARTIFACT_ERROR = 6

DEFAULT_RUN_OUTPUT_ROOT = "runs"
DEFAULT_SPREADSHEET_EXPORT_DIRNAME = "spreadsheet_export"

REQUIRED_MANIFEST_ARTIFACTS: Mapping[str, str] = {
    "schedule_json": "schedule.json",
    "operations_csv": "operations.csv",
    "machine_timeline_csv": "machine_timeline.csv",
    "worker_timeline_csv": "worker_timeline.csv",
    "violations_json": "violations.json",
    "violations_csv": "violations.csv",
    "machine_calendar_csv": "machine_calendar.csv",
    "worker_calendar_csv": "worker_calendar.csv",
    "events_csv": "events.csv",
}

CLI_EXIT_CODES: Mapping[int, Dict[str, str]] = {
    EXIT_SUCCESS: {
        "error_class": "success",
        "label": "success",
        "description": "The command completed and emitted a valid machine-readable payload.",
    },
    EXIT_VALIDATION_ERROR: {
        "error_class": "validation_error",
        "label": "validation_error",
        "description": "The external input or CLI arguments failed documented validation checks.",
    },
    EXIT_UNSUPPORTED_REQUEST: {
        "error_class": "unsupported_request",
        "label": "unsupported_request",
        "description": "The requested solver, adapter, or workflow shape is not supported.",
    },
    EXIT_SOLVER_FAILURE: {
        "error_class": "solver_error",
        "label": "solver_error",
        "description": "The requested solver could not produce a supported schedule result.",
    },
    EXIT_RUNTIME_ERROR: {
        "error_class": "runtime_error",
        "label": "runtime_error",
        "description": "An unexpected runtime failure occurred outside documented validation or solver failures.",
    },
    EXIT_ARTIFACT_ERROR: {
        "error_class": "artifact_error",
        "label": "artifact_error",
        "description": "A run directory or exported artifact bundle was incomplete or contract-invalid.",
    },
}

CLI_ERROR_CATALOG: Mapping[str, Dict[str, Any]] = {
    "input_validation_failed": {
        "exit_code": EXIT_VALIDATION_ERROR,
        "error_class": "validation_error",
        "default_message": "External input validation failed.",
    },
    "unsupported_solver": {
        "exit_code": EXIT_UNSUPPORTED_REQUEST,
        "error_class": "unsupported_request",
        "default_message": "The requested solver is not supported by the CLI.",
    },
    "unsupported_adapter_input": {
        "exit_code": EXIT_UNSUPPORTED_REQUEST,
        "error_class": "unsupported_request",
        "default_message": "Adapters are only supported for JSON source files, not CSV bundles.",
    },
    "solver_no_solution": {
        "exit_code": EXIT_SOLVER_FAILURE,
        "error_class": "solver_error",
        "default_message": "The solver returned no schedule.",
    },
    "missing_dependency": {
        "exit_code": EXIT_SOLVER_FAILURE,
        "error_class": "solver_error",
        "default_message": "The requested solver dependency is unavailable.",
    },
    "solver_unavailable": {
        "exit_code": EXIT_SOLVER_FAILURE,
        "error_class": "solver_error",
        "default_message": "The requested solver mode is unavailable.",
    },
    "run_directory_not_found": {
        "exit_code": EXIT_ARTIFACT_ERROR,
        "error_class": "artifact_error",
        "default_message": "The requested run directory does not exist.",
    },
    "invalid_run_manifest": {
        "exit_code": EXIT_ARTIFACT_ERROR,
        "error_class": "artifact_error",
        "default_message": "The run manifest is missing required fields or schema markers.",
    },
    "manifest_incomplete": {
        "exit_code": EXIT_ARTIFACT_ERROR,
        "error_class": "artifact_error",
        "default_message": "The run directory is missing required exported artifacts.",
    },
    "artifact_schema_mismatch": {
        "exit_code": EXIT_ARTIFACT_ERROR,
        "error_class": "artifact_error",
        "default_message": "The run directory contains artifacts that do not match the documented schemas.",
    },
    "unknown_command": {
        "exit_code": EXIT_RUNTIME_ERROR,
        "error_class": "runtime_error",
        "default_message": "The CLI received an unknown command.",
    },
    "unhandled_exception": {
        "exit_code": EXIT_RUNTIME_ERROR,
        "error_class": "runtime_error",
        "default_message": "The CLI hit an unexpected unhandled exception.",
    },
}

OPERATOR_RUNBOOKS: Mapping[str, Dict[str, Any]] = {
    "validate_input": {
        "description": "Validate a JSON payload or CSV bundle against the supported external schema.",
        "example": (
            "python -m interfaces.cli validate-input --input "
            "tests/fixtures/interfaces/valid_minimal.json"
        ),
    },
    "solve": {
        "description": "Import, solve, audit, and export a deterministic run directory.",
        "example": (
            "python -m interfaces.cli solve --input tests/fixtures/interfaces/valid_minimal.json "
            "--solver greedy:spt --output-root runs"
        ),
    },
    "audit": {
        "description": "Inspect a previously exported run directory and summarize violations and provenance.",
        "example": "python -m interfaces.cli audit --run-dir runs/EXT_MINIMAL/greedy-spt",
    },
    "export": {
        "description": "Copy the stable spreadsheet-facing artifact set from a run directory into a handoff directory.",
        "example": (
            "python -m interfaces.cli export --run-dir runs/EXT_MINIMAL/greedy-spt "
            "--target-dir handoff/EXT_MINIMAL-greedy-spt"
        ),
    },
}


@dataclass(frozen=True)
class RunDirectoryBundle:
    """Validated run-directory payloads and resolved file paths."""

    run_dir: Path
    manifest_path: Path
    manifest: Dict[str, Any]
    schedule_path: Path
    schedule_payload: Dict[str, Any]
    violations_path: Path
    audit_payload: Dict[str, Any]


class RunDirectoryContractError(ValueError):
    """Structured run-directory contract failure with a stable error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def build_default_run_output_dir(output_root: Any, instance_id: str, solver_spec: str) -> Path:
    """Return the documented deterministic run-directory path."""

    return Path(output_root) / _slugify(instance_id) / _slugify(solver_spec)


def build_default_spreadsheet_export_dir(run_dir: Any) -> Path:
    """Return the documented deterministic spreadsheet handoff directory."""

    return Path(run_dir) / DEFAULT_SPREADSHEET_EXPORT_DIRNAME


def load_run_directory_bundle(run_dir: Any) -> RunDirectoryBundle:
    """Load and validate one exported run directory against the Phase 12 contract."""

    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        raise RunDirectoryContractError(
            "run_directory_not_found",
            CLI_ERROR_CATALOG["run_directory_not_found"]["default_message"],
        )

    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.exists():
        raise RunDirectoryContractError(
            "manifest_incomplete",
            f"{CLI_ERROR_CATALOG['manifest_incomplete']['default_message']} Missing run_manifest.json."
        )
    manifest = _read_json(manifest_path)
    validate_run_manifest_payload(manifest)

    missing_files: List[str] = []
    resolved_paths: Dict[str, Path] = {}
    for artifact_key, expected_name in REQUIRED_MANIFEST_ARTIFACTS.items():
        filename = manifest["artifacts"].get(artifact_key)
        if filename != expected_name:
            raise RunDirectoryContractError(
                "invalid_run_manifest",
                "The run manifest does not use the documented deterministic artifact names."
            )
        artifact_path = run_path / filename
        if not artifact_path.exists():
            missing_files.append(filename)
        resolved_paths[artifact_key] = artifact_path

    if missing_files:
        raise RunDirectoryContractError(
            "manifest_incomplete",
            f"{CLI_ERROR_CATALOG['manifest_incomplete']['default_message']} Missing: "
            + ", ".join(sorted(missing_files))
        )

    schedule_payload = _read_json(resolved_paths["schedule_json"])
    if schedule_payload.get("schedule_schema") != manifest["schedule_schema"]:
        raise RunDirectoryContractError(
            "artifact_schema_mismatch",
            "The run directory schedule bundle does not match the schema declared in run_manifest.json."
        )

    audit_payload = _read_json(resolved_paths["violations_json"])
    validate_schedule_audit_payload(audit_payload)
    if audit_payload.get("audit_schema") != manifest["audit_schema"]:
        raise RunDirectoryContractError(
            "artifact_schema_mismatch",
            "The run directory audit bundle does not match the schema declared in run_manifest.json."
        )

    return RunDirectoryBundle(
        run_dir=run_path,
        manifest_path=manifest_path,
        manifest=manifest,
        schedule_path=resolved_paths["schedule_json"],
        schedule_payload=schedule_payload,
        violations_path=resolved_paths["violations_json"],
        audit_payload=audit_payload,
    )


def validate_run_manifest_payload(manifest: Mapping[str, Any]) -> None:
    """Reject run manifests that do not satisfy the documented operator contract."""

    if manifest.get("manifest_schema") != RUN_MANIFEST_SCHEMA:
        raise RunDirectoryContractError(
            "invalid_run_manifest",
            f"Expected manifest_schema {RUN_MANIFEST_SCHEMA!r}, got "
            f"{manifest.get('manifest_schema')!r}."
        )
    if manifest.get("schedule_schema") != SCHEDULE_EXPORT_SCHEMA:
        raise RunDirectoryContractError(
            "invalid_run_manifest",
            f"Expected schedule_schema {SCHEDULE_EXPORT_SCHEMA!r}, got "
            f"{manifest.get('schedule_schema')!r}."
        )
    if manifest.get("audit_schema") != SCHEDULE_AUDIT_SCHEMA:
        raise RunDirectoryContractError(
            "invalid_run_manifest",
            f"Expected audit_schema {SCHEDULE_AUDIT_SCHEMA!r}, got "
            f"{manifest.get('audit_schema')!r}."
        )
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        raise RunDirectoryContractError(
            "invalid_run_manifest",
            "Run manifest must include a calibration record.",
        )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise RunDirectoryContractError(
            "invalid_run_manifest",
            "Run manifest must include provenance.",
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RunDirectoryContractError(
            "invalid_run_manifest",
            "Run manifest must include an artifacts mapping.",
        )
    missing_artifact_keys = sorted(
        key for key in REQUIRED_MANIFEST_ARTIFACTS if key not in artifacts
    )
    if missing_artifact_keys:
        raise RunDirectoryContractError(
            "invalid_run_manifest",
            "Run manifest is missing required artifact keys: "
            + ", ".join(missing_artifact_keys)
        )


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RunDirectoryContractError(
            "artifact_schema_mismatch",
            f"{path.name} must contain a JSON object.",
        )
    return payload


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    slug = slug.strip("-._")
    return slug or "unknown"
