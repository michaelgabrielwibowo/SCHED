"""
CSV bundle importer that feeds the canonical external payload validator.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .errors import (
    DUPLICATE_ID,
    EMPTY_COLLECTION,
    INVALID_REFERENCE,
    INVALID_TYPE,
    INVALID_VALUE,
    InterfaceValidationError,
    MISSING_REQUIRED,
    UNKNOWN_FIELD,
    ValidationIssue,
    issue,
)
from .importers import import_instance_from_dict
from .schema import (
    EXTERNAL_INPUT_SCHEMA,
    EXTERNAL_INPUT_SCHEMA_V2,
    REQUIRED_ELECTRICITY_PRICE_FIELDS,
    REQUIRED_JOB_FIELDS,
    REQUIRED_MACHINE_BREAKDOWN_FIELDS,
    REQUIRED_MACHINE_FIELDS,
    REQUIRED_MACHINE_MODE_FIELDS,
    REQUIRED_MACHINE_UNAVAILABILITY_FIELDS,
    REQUIRED_METADATA_FIELDS,
    REQUIRED_OPERATION_FIELDS,
    REQUIRED_PROCESSING_OPTION_FIELDS,
    REQUIRED_WORKER_ABSENCE_FIELDS,
    REQUIRED_WORKER_UNAVAILABILITY_FIELDS,
    REQUIRED_WORKER_FIELDS,
    SUPPORTED_ELECTRICITY_PRICE_FIELDS,
    SUPPORTED_MACHINE_BREAKDOWN_FIELDS,
    SUPPORTED_JOB_FIELDS,
    SUPPORTED_CALENDAR_FIELDS,
    SUPPORTED_MACHINE_FIELDS,
    SUPPORTED_MACHINE_MODE_FIELDS,
    SUPPORTED_MACHINE_UNAVAILABILITY_FIELDS,
    SUPPORTED_METADATA_FIELDS,
    SUPPORTED_OPERATION_FIELDS,
    SUPPORTED_PROCESSING_OPTION_FIELDS,
    SUPPORTED_WORKER_ABSENCE_FIELDS,
    SUPPORTED_WORKER_UNAVAILABILITY_FIELDS,
    SUPPORTED_WORKER_FIELDS,
)
from .types import ImportedInstance


LIST_SEPARATOR = "|"


@dataclass(frozen=True)
class _CsvTableSpec:
    name: str
    filename: str
    required_columns: frozenset[str]
    optional_columns: frozenset[str]
    required: bool = True
    singleton: bool = False

    @property
    def allowed_columns(self) -> frozenset[str]:
        return self.required_columns | self.optional_columns


CSV_TABLE_SPECS: Dict[str, _CsvTableSpec] = {
    "metadata": _CsvTableSpec(
        name="metadata",
        filename="metadata.csv",
        required_columns=REQUIRED_METADATA_FIELDS,
        optional_columns=SUPPORTED_METADATA_FIELDS - REQUIRED_METADATA_FIELDS,
        singleton=True,
    ),
    "defaults": _CsvTableSpec(
        name="defaults",
        filename="defaults.csv",
        required_columns=frozenset(),
        optional_columns=frozenset(
            {
                "default_ergonomic_risk",
                "default_electricity_price",
                "carbon_emission_factor",
                "auxiliary_power_total",
            }
        ),
        required=False,
        singleton=True,
    ),
    "electricity_prices": _CsvTableSpec(
        name="electricity_prices",
        filename="electricity_prices.csv",
        required_columns=REQUIRED_ELECTRICITY_PRICE_FIELDS,
        optional_columns=SUPPORTED_ELECTRICITY_PRICE_FIELDS
        - REQUIRED_ELECTRICITY_PRICE_FIELDS,
        required=False,
    ),
    "machines": _CsvTableSpec(
        name="machines",
        filename="machines.csv",
        required_columns=REQUIRED_MACHINE_FIELDS - frozenset({"modes"}),
        optional_columns=(SUPPORTED_MACHINE_FIELDS - REQUIRED_MACHINE_FIELDS)
        - frozenset({"modes"}),
    ),
    "machine_modes": _CsvTableSpec(
        name="machine_modes",
        filename="machine_modes.csv",
        required_columns=REQUIRED_MACHINE_MODE_FIELDS | frozenset({"machine_id"}),
        optional_columns=SUPPORTED_MACHINE_MODE_FIELDS - REQUIRED_MACHINE_MODE_FIELDS,
    ),
    "workers": _CsvTableSpec(
        name="workers",
        filename="workers.csv",
        required_columns=REQUIRED_WORKER_FIELDS,
        optional_columns=SUPPORTED_WORKER_FIELDS - REQUIRED_WORKER_FIELDS,
    ),
    "jobs": _CsvTableSpec(
        name="jobs",
        filename="jobs.csv",
        required_columns=REQUIRED_JOB_FIELDS - frozenset({"operations"}),
        optional_columns=(SUPPORTED_JOB_FIELDS - REQUIRED_JOB_FIELDS)
        - frozenset({"operations"}),
    ),
    "operations": _CsvTableSpec(
        name="operations",
        filename="operations.csv",
        required_columns=(REQUIRED_OPERATION_FIELDS - frozenset({"processing_options"}))
        | frozenset({"job_id"}),
        optional_columns=(SUPPORTED_OPERATION_FIELDS - REQUIRED_OPERATION_FIELDS)
        - frozenset({"processing_options"}),
    ),
    "operation_modes": _CsvTableSpec(
        name="operation_modes",
        filename="operation_modes.csv",
        required_columns=REQUIRED_PROCESSING_OPTION_FIELDS
        | frozenset({"job_id", "operation_id"}),
        optional_columns=SUPPORTED_PROCESSING_OPTION_FIELDS
        - REQUIRED_PROCESSING_OPTION_FIELDS,
    ),
    "machine_unavailability": _CsvTableSpec(
        name="machine_unavailability",
        filename="machine_unavailability.csv",
        required_columns=REQUIRED_MACHINE_UNAVAILABILITY_FIELDS,
        optional_columns=(SUPPORTED_MACHINE_UNAVAILABILITY_FIELDS
        - REQUIRED_MACHINE_UNAVAILABILITY_FIELDS)
        | frozenset({"details_json"}),
        required=False,
    ),
    "worker_unavailability": _CsvTableSpec(
        name="worker_unavailability",
        filename="worker_unavailability.csv",
        required_columns=REQUIRED_WORKER_UNAVAILABILITY_FIELDS,
        optional_columns=(SUPPORTED_WORKER_UNAVAILABILITY_FIELDS
        - REQUIRED_WORKER_UNAVAILABILITY_FIELDS)
        | frozenset({"details_json"}),
        required=False,
    ),
    "machine_breakdowns": _CsvTableSpec(
        name="machine_breakdowns",
        filename="machine_breakdowns.csv",
        required_columns=REQUIRED_MACHINE_BREAKDOWN_FIELDS,
        optional_columns=(SUPPORTED_MACHINE_BREAKDOWN_FIELDS
        - REQUIRED_MACHINE_BREAKDOWN_FIELDS)
        | frozenset({"details_json"}),
        required=False,
    ),
    "worker_absences": _CsvTableSpec(
        name="worker_absences",
        filename="worker_absences.csv",
        required_columns=REQUIRED_WORKER_ABSENCE_FIELDS,
        optional_columns=(SUPPORTED_WORKER_ABSENCE_FIELDS
        - REQUIRED_WORKER_ABSENCE_FIELDS)
        | frozenset({"details_json"}),
        required=False,
    ),
}

CSV_V2_TABLE_NAMES = frozenset(
    {
        "machine_unavailability",
        "worker_unavailability",
        "machine_breakdowns",
        "worker_absences",
    }
)


def load_instance_from_csv_bundle(path: Any, strict: bool = True) -> ImportedInstance:
    """Load a CSV bundle into the canonical external payload importer."""

    importer = _CsvBundleImporter(bundle_path=Path(path), strict=strict)
    payload = importer.build_external_payload()
    importer.raise_if_needed()
    return import_instance_from_dict(payload, strict=strict)


class _CsvBundleImporter:
    def __init__(self, bundle_path: Path, strict: bool):
        self.bundle_path = bundle_path
        self.strict = strict
        self.issues: List[ValidationIssue] = []
        self.present_tables: set[str] = set()

    def build_external_payload(self) -> Dict[str, Any]:
        if not self.bundle_path.exists() or not self.bundle_path.is_dir():
            self.issues.append(
                issue(
                    INVALID_VALUE,
                    "csv_bundle",
                    message=f"CSV bundle path {str(self.bundle_path)!r} must be an existing directory.",
                )
            )
            self.raise_if_needed()

        tables = {name: self._load_table(spec) for name, spec in CSV_TABLE_SPECS.items()}
        self.raise_if_needed()

        schema_name = self._detect_schema_name()

        payload: Dict[str, Any] = {
            "schema": schema_name,
            "metadata": self._build_metadata(tables["metadata"]),
            "machines": self._build_machines(tables["machines"], tables["machine_modes"]),
            "workers": self._build_workers(tables["workers"]),
            "jobs": self._build_jobs(
                tables["jobs"],
                tables["operations"],
                tables["operation_modes"],
            ),
        }

        defaults = self._build_defaults(tables["defaults"], tables["electricity_prices"])
        if defaults:
            payload["defaults"] = defaults
        if schema_name == EXTERNAL_INPUT_SCHEMA_V2:
            payload["calendar"] = self._build_calendar(
                tables["machine_unavailability"],
                tables["worker_unavailability"],
            )
            payload["events"] = self._build_events(
                tables["machine_breakdowns"],
                tables["worker_absences"],
            )
        return payload

    def raise_if_needed(self) -> None:
        if self.issues:
            raise InterfaceValidationError(self.issues)

    def _load_table(self, spec: _CsvTableSpec) -> List[Dict[str, str]]:
        path = self.bundle_path / spec.filename
        location = ("csv_bundle", spec.name)
        if not path.exists():
            if spec.required:
                self.issues.append(
                    issue(
                        MISSING_REQUIRED,
                        *location,
                        message=f"Missing required table {spec.filename!r}.",
                    )
                )
            return []
        self.present_tables.add(spec.name)

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                self.issues.append(
                    issue(
                        INVALID_TYPE,
                        *location,
                        message=f"Table {spec.filename!r} must include a header row.",
                    )
                )
                return []

            headers = [field.strip() for field in reader.fieldnames if field is not None]
            if spec.required_columns - set(headers):
                missing = sorted(spec.required_columns - set(headers))
                self.issues.append(
                    issue(
                        MISSING_REQUIRED,
                        *location,
                        message=f"Table {spec.filename!r} is missing required columns: {', '.join(missing)}.",
                    )
                )
            if self.strict:
                unknown = sorted(set(headers) - spec.allowed_columns)
                for column in unknown:
                    self.issues.append(
                        issue(
                            UNKNOWN_FIELD,
                            *location,
                            column,
                            message=f"Unknown column {column!r} in table {spec.filename!r}.",
                        )
                    )

            rows: List[Dict[str, str]] = []
            for index, raw_row in enumerate(reader):
                if None in raw_row:
                    self.issues.append(
                        issue(
                            INVALID_VALUE,
                            *location,
                            index,
                            message=f"Row {index + 1} in {spec.filename!r} has more values than headers.",
                        )
                    )
                    continue
                row = {
                    str(key).strip(): (value.strip() if isinstance(value, str) else "")
                    for key, value in raw_row.items()
                }
                if all(value == "" for value in row.values()):
                    continue
                rows.append(row)

        if spec.required and not rows:
            self.issues.append(
                issue(
                    EMPTY_COLLECTION,
                    *location,
                    message=f"Table {spec.filename!r} must contain at least one data row.",
                )
            )
        if spec.singleton and len(rows) > 1:
            self.issues.append(
                issue(
                    INVALID_VALUE,
                    *location,
                    message=f"Table {spec.filename!r} must contain exactly one data row.",
                )
            )
        return rows

    def _detect_schema_name(self) -> str:
        if self.present_tables & CSV_V2_TABLE_NAMES:
            return EXTERNAL_INPUT_SCHEMA_V2
        return EXTERNAL_INPUT_SCHEMA

    def _build_metadata(self, rows: List[Dict[str, str]]) -> Dict[str, Any]:
        row = rows[0] if rows else {}
        path = ("csv_bundle", "metadata", 0)
        metadata: Dict[str, Any] = {
            "instance_id": self._required_text(row, "instance_id", path),
        }
        self._set_if_present(metadata, "instance_name", self._optional_text(row, "instance_name"))
        self._set_if_present(metadata, "instance_type", self._optional_text(row, "instance_type"))
        self._set_if_present(
            metadata,
            "planning_horizon",
            self._optional_float(row, "planning_horizon", path),
        )
        self._set_if_present(
            metadata,
            "period_duration",
            self._optional_float(row, "period_duration", path),
        )
        self._set_if_present(
            metadata,
            "horizon_start",
            self._optional_float(row, "horizon_start", path),
        )
        self._set_if_present(metadata, "source", self._optional_text(row, "source"))
        self._set_if_present(metadata, "label", self._optional_text(row, "label"))
        self._set_if_present(
            metadata,
            "label_justification",
            self._optional_text(row, "label_justification"),
        )
        self._set_if_present(
            metadata,
            "calibration_sources",
            self._optional_list(row, "calibration_sources"),
        )
        self._set_if_present(
            metadata,
            "known_limitations",
            self._optional_list(row, "known_limitations"),
        )
        return metadata

    def _build_defaults(
        self,
        default_rows: List[Dict[str, str]],
        electricity_rows: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        row = default_rows[0] if default_rows else {}
        path = ("csv_bundle", "defaults", 0)
        self._set_if_present(
            defaults,
            "default_ergonomic_risk",
            self._optional_float(row, "default_ergonomic_risk", path),
        )
        self._set_if_present(
            defaults,
            "default_electricity_price",
            self._optional_float(row, "default_electricity_price", path),
        )
        self._set_if_present(
            defaults,
            "carbon_emission_factor",
            self._optional_float(row, "carbon_emission_factor", path),
        )
        self._set_if_present(
            defaults,
            "auxiliary_power_total",
            self._optional_float(row, "auxiliary_power_total", path),
        )

        electricity_prices: List[Dict[str, Any]] = []
        for index, price_row in enumerate(electricity_rows):
            price_path = ("csv_bundle", "electricity_prices", index)
            electricity_prices.append(
                {
                    "period": self._required_int(price_row, "period", price_path),
                    "price": self._required_float(price_row, "price", price_path),
                }
            )
        if electricity_prices:
            defaults["electricity_prices"] = electricity_prices
        return defaults

    def _build_machines(
        self,
        machine_rows: List[Dict[str, str]],
        machine_mode_rows: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        known_machine_ids = {
            self._required_text(row, "id", ("csv_bundle", "machines", index))
            for index, row in enumerate(machine_rows)
            if self._optional_text(row, "id") is not None
        }
        modes_by_machine: Dict[str, List[Dict[str, Any]]] = {}
        for index, mode_row in enumerate(machine_mode_rows):
            path = ("csv_bundle", "machine_modes", index)
            machine_id = self._required_text(mode_row, "machine_id", path)
            if machine_id not in known_machine_ids:
                self.issues.append(
                    issue(
                        INVALID_REFERENCE,
                        *path,
                        "machine_id",
                        message=f"Unknown machine reference {machine_id!r} in machine_modes.csv.",
                    )
                )
                continue

            mode_payload: Dict[str, Any] = {
                "id": self._required_text(mode_row, "id", path),
            }
            self._set_if_present(mode_payload, "name", self._optional_text(mode_row, "name"))
            self._set_if_present(
                mode_payload,
                "speed_factor",
                self._optional_float(mode_row, "speed_factor", path),
            )
            self._set_if_present(
                mode_payload,
                "power_multiplier",
                self._optional_float(mode_row, "power_multiplier", path),
            )
            self._set_if_present(
                mode_payload,
                "tool_wear_rate",
                self._optional_float(mode_row, "tool_wear_rate", path),
            )
            modes_by_machine.setdefault(machine_id, []).append(mode_payload)

        machines: List[Dict[str, Any]] = []
        for index, machine_row in enumerate(machine_rows):
            path = ("csv_bundle", "machines", index)
            machine_id = self._required_text(machine_row, "id", path)
            machine_payload: Dict[str, Any] = {
                "id": machine_id,
                "modes": modes_by_machine.get(machine_id, []),
            }
            self._set_if_present(machine_payload, "name", self._optional_text(machine_row, "name"))
            self._set_if_present(
                machine_payload,
                "default_mode_id",
                self._optional_text(machine_row, "default_mode_id"),
            )
            self._set_if_present(
                machine_payload,
                "power_processing",
                self._optional_float(machine_row, "power_processing", path),
            )
            self._set_if_present(
                machine_payload,
                "power_idle",
                self._optional_float(machine_row, "power_idle", path),
            )
            self._set_if_present(
                machine_payload,
                "power_setup",
                self._optional_float(machine_row, "power_setup", path),
            )
            self._set_if_present(
                machine_payload,
                "power_transport",
                self._optional_float(machine_row, "power_transport", path),
            )
            self._set_if_present(
                machine_payload,
                "startup_energy",
                self._optional_float(machine_row, "startup_energy", path),
            )
            self._set_if_present(
                machine_payload,
                "setup_time",
                self._optional_float(machine_row, "setup_time", path),
            )
            self._set_if_present(
                machine_payload,
                "auxiliary_power_share",
                self._optional_float(machine_row, "auxiliary_power_share", path),
            )
            machines.append(machine_payload)
        return machines

    def _build_workers(self, worker_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        workers: List[Dict[str, Any]] = []
        numeric_fields = [
            "labor_cost_per_hour",
            "base_efficiency",
            "fatigue_rate",
            "recovery_rate",
            "fatigue_max",
            "ocra_max_per_shift",
            "ergonomic_tolerance",
            "min_rest_fraction",
            "max_consecutive_work_time",
            "learning_coefficient",
        ]
        for index, worker_row in enumerate(worker_rows):
            path = ("csv_bundle", "workers", index)
            worker_payload: Dict[str, Any] = {
                "id": self._required_text(worker_row, "id", path),
            }
            self._set_if_present(worker_payload, "name", self._optional_text(worker_row, "name"))
            for field in numeric_fields:
                self._set_if_present(
                    worker_payload,
                    field,
                    self._optional_float(worker_row, field, path),
                )
            workers.append(worker_payload)
        return workers

    def _build_jobs(
        self,
        job_rows: List[Dict[str, str]],
        operation_rows: List[Dict[str, str]],
        option_rows: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        known_job_ids = {
            self._required_text(row, "id", ("csv_bundle", "jobs", index))
            for index, row in enumerate(job_rows)
            if self._optional_text(row, "id") is not None
        }
        operations_by_job: Dict[str, List[Dict[str, Any]]] = {}
        operation_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
        duplicate_operation_keys: set[tuple[str, str]] = set()

        for index, operation_row in enumerate(operation_rows):
            path = ("csv_bundle", "operations", index)
            job_id = self._required_text(operation_row, "job_id", path)
            if job_id not in known_job_ids:
                self.issues.append(
                    issue(
                        INVALID_REFERENCE,
                        *path,
                        "job_id",
                        message=f"Unknown job reference {job_id!r} in operations.csv.",
                    )
                )
                continue

            operation_id = self._required_text(operation_row, "id", path)
            operation_key = (job_id, operation_id)
            if operation_key in operation_lookup:
                duplicate_operation_keys.add(operation_key)
                self.issues.append(
                    issue(
                        DUPLICATE_ID,
                        *path,
                        "id",
                        message=f"Duplicate operation identifier {operation_id!r} within job {job_id!r}.",
                    )
                )

            operation_payload: Dict[str, Any] = {
                "id": operation_id,
                "eligible_workers": self._required_list(operation_row, "eligible_workers", path),
                "processing_options": [],
            }
            self._set_if_present(
                operation_payload,
                "ergonomic_risk_rate",
                self._optional_float(operation_row, "ergonomic_risk_rate", path),
            )
            self._set_if_present(
                operation_payload,
                "transport_time",
                self._optional_float(operation_row, "transport_time", path),
            )
            self._set_if_present(
                operation_payload,
                "waiting_time",
                self._optional_float(operation_row, "waiting_time", path),
            )
            operations_by_job.setdefault(job_id, []).append(operation_payload)
            operation_lookup.setdefault(operation_key, operation_payload)

        for index, option_row in enumerate(option_rows):
            path = ("csv_bundle", "operation_modes", index)
            job_id = self._required_text(option_row, "job_id", path)
            operation_id = self._required_text(option_row, "operation_id", path)
            operation_key = (job_id, operation_id)
            if job_id not in known_job_ids:
                self.issues.append(
                    issue(
                        INVALID_REFERENCE,
                        *path,
                        "job_id",
                        message=f"Unknown job reference {job_id!r} in operation_modes.csv.",
                    )
                )
                continue
            if operation_key in duplicate_operation_keys:
                self.issues.append(
                    issue(
                        INVALID_REFERENCE,
                        *path,
                        "operation_id",
                        message=(
                            f"Operation reference {operation_id!r} in job {job_id!r} is ambiguous because "
                            "operations.csv contains duplicate operation identifiers."
                        ),
                    )
                )
                continue
            if operation_key not in operation_lookup:
                self.issues.append(
                    issue(
                        INVALID_REFERENCE,
                        *path,
                        "operation_id",
                        message=f"Unknown operation reference {operation_id!r} in job {job_id!r}.",
                    )
                )
                continue

            option_payload = {
                "machine_id": self._required_text(option_row, "machine_id", path),
                "mode_id": self._required_text(option_row, "mode_id", path),
                "duration": self._required_float(option_row, "duration", path),
            }
            operation_lookup[operation_key]["processing_options"].append(option_payload)

        jobs: List[Dict[str, Any]] = []
        for index, job_row in enumerate(job_rows):
            path = ("csv_bundle", "jobs", index)
            job_id = self._required_text(job_row, "id", path)
            job_payload: Dict[str, Any] = {
                "id": job_id,
                "operations": operations_by_job.get(job_id, []),
            }
            self._set_if_present(
                job_payload,
                "arrival_time",
                self._optional_float(job_row, "arrival_time", path),
            )
            self._set_if_present(
                job_payload,
                "due_date",
                self._optional_float(job_row, "due_date", path),
            )
            self._set_if_present(
                job_payload,
                "weight",
                self._optional_float(job_row, "weight", path),
            )
            jobs.append(job_payload)
        return jobs

    def _build_calendar(
        self,
        machine_rows: List[Dict[str, str]],
        worker_rows: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        return {
            "machine_unavailability": [
                self._build_machine_unavailability_row(row, index)
                for index, row in enumerate(machine_rows)
            ],
            "worker_unavailability": [
                self._build_worker_unavailability_row(row, index)
                for index, row in enumerate(worker_rows)
            ],
        }

    def _build_events(
        self,
        machine_rows: List[Dict[str, str]],
        worker_rows: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        return {
            "machine_breakdowns": [
                self._build_machine_breakdown_row(row, index)
                for index, row in enumerate(machine_rows)
            ],
            "worker_absences": [
                self._build_worker_absence_row(row, index)
                for index, row in enumerate(worker_rows)
            ],
        }

    def _build_machine_unavailability_row(
        self,
        row: Dict[str, str],
        index: int,
    ) -> Dict[str, Any]:
        path = ("csv_bundle", "machine_unavailability", index)
        payload: Dict[str, Any] = {
            "machine_id": self._required_text(row, "machine_id", path),
            "start_time": self._required_float(row, "start_time", path),
            "end_time": self._required_float(row, "end_time", path),
        }
        self._set_if_present(payload, "reason", self._optional_text(row, "reason"))
        self._set_if_present(payload, "source", self._optional_text(row, "source"))
        self._set_if_present(payload, "details", self._optional_json_object(row, "details_json", path))
        return payload

    def _build_worker_unavailability_row(
        self,
        row: Dict[str, str],
        index: int,
    ) -> Dict[str, Any]:
        path = ("csv_bundle", "worker_unavailability", index)
        payload: Dict[str, Any] = {
            "worker_id": self._required_text(row, "worker_id", path),
            "start_time": self._required_float(row, "start_time", path),
            "end_time": self._required_float(row, "end_time", path),
        }
        self._set_if_present(payload, "reason", self._optional_text(row, "reason"))
        self._set_if_present(payload, "source", self._optional_text(row, "source"))
        self._set_if_present(payload, "details", self._optional_json_object(row, "details_json", path))
        return payload

    def _build_machine_breakdown_row(
        self,
        row: Dict[str, str],
        index: int,
    ) -> Dict[str, Any]:
        path = ("csv_bundle", "machine_breakdowns", index)
        payload: Dict[str, Any] = {
            "machine_id": self._required_text(row, "machine_id", path),
            "start_time": self._required_float(row, "start_time", path),
            "repair_duration": self._required_float(row, "repair_duration", path),
        }
        self._set_if_present(payload, "source", self._optional_text(row, "source"))
        self._set_if_present(payload, "details", self._optional_json_object(row, "details_json", path))
        return payload

    def _build_worker_absence_row(
        self,
        row: Dict[str, str],
        index: int,
    ) -> Dict[str, Any]:
        path = ("csv_bundle", "worker_absences", index)
        payload: Dict[str, Any] = {
            "worker_id": self._required_text(row, "worker_id", path),
            "start_time": self._required_float(row, "start_time", path),
            "end_time": self._required_float(row, "end_time", path),
        }
        self._set_if_present(payload, "source", self._optional_text(row, "source"))
        self._set_if_present(payload, "details", self._optional_json_object(row, "details_json", path))
        return payload

    def _required_text(
        self,
        row: Dict[str, str],
        column: str,
        path: Iterable[object],
    ) -> str:
        value = self._optional_text(row, column)
        if value is None:
            self.issues.append(
                issue(
                    MISSING_REQUIRED,
                    *path,
                    column,
                    message=f"Column {column!r} is required.",
                )
            )
            return ""
        return value

    def _optional_text(self, row: Dict[str, str], column: str) -> Optional[str]:
        value = row.get(column, "")
        if value is None:
            return None
        text = value.strip()
        return text or None

    def _required_float(
        self,
        row: Dict[str, str],
        column: str,
        path: Iterable[object],
    ) -> float:
        value = self._optional_float(row, column, path)
        if value is None:
            self.issues.append(
                issue(
                    MISSING_REQUIRED,
                    *path,
                    column,
                    message=f"Column {column!r} is required.",
                )
            )
            return 0.0
        return value

    def _optional_float(
        self,
        row: Dict[str, str],
        column: str,
        path: Iterable[object],
    ) -> Optional[float]:
        raw = self._optional_text(row, column)
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            self.issues.append(
                issue(
                    INVALID_VALUE,
                    *path,
                    column,
                    message=f"Column {column!r} must be numeric, got {raw!r}.",
                )
            )
            return None

    def _required_int(
        self,
        row: Dict[str, str],
        column: str,
        path: Iterable[object],
    ) -> int:
        raw = self._optional_text(row, column)
        if raw is None:
            self.issues.append(
                issue(
                    MISSING_REQUIRED,
                    *path,
                    column,
                    message=f"Column {column!r} is required.",
                )
            )
            return 0
        try:
            return int(raw)
        except ValueError:
            self.issues.append(
                issue(
                    INVALID_VALUE,
                    *path,
                    column,
                    message=f"Column {column!r} must be an integer, got {raw!r}.",
                )
            )
            return 0

    def _required_list(
        self,
        row: Dict[str, str],
        column: str,
        path: Iterable[object],
    ) -> List[str]:
        items = self._optional_list(row, column)
        if items is None:
            self.issues.append(
                issue(
                    MISSING_REQUIRED,
                    *path,
                    column,
                    message=f"Column {column!r} is required.",
                    hint=f"Use {LIST_SEPARATOR!r} to separate multiple identifiers in one cell.",
                )
            )
            return []
        return items

    def _optional_list(self, row: Dict[str, str], column: str) -> Optional[List[str]]:
        raw = self._optional_text(row, column)
        if raw is None:
            return None
        return [item.strip() for item in raw.split(LIST_SEPARATOR) if item.strip()]

    def _optional_json_object(
        self,
        row: Dict[str, str],
        column: str,
        path: Iterable[object],
    ) -> Optional[Dict[str, Any]]:
        raw = self._optional_text(row, column)
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.issues.append(
                issue(
                    INVALID_VALUE,
                    *path,
                    column,
                    message=(
                        f"Column {column!r} must contain a valid JSON object string, got {raw!r}."
                    ),
                    hint=str(exc),
                )
            )
            return None
        if not isinstance(parsed, dict):
            self.issues.append(
                issue(
                    INVALID_TYPE,
                    *path,
                    column,
                    message=f"Column {column!r} must decode to a JSON object.",
                )
            )
            return None
        return parsed

    @staticmethod
    def _set_if_present(target: Dict[str, Any], key: str, value: Any) -> None:
        if value is not None:
            target[key] = value
