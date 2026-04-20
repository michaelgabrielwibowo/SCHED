"""
Adapter for a plant-table JSON layout commonly seen in MES/ERP exports.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from ..errors import (
    INVALID_TYPE,
    INVALID_VALUE,
    InterfaceValidationError,
    MISSING_REQUIRED,
    UNSUPPORTED_SECTION,
    ValidationIssue,
    issue,
)
from ..schema import EXTERNAL_INPUT_SCHEMA_V1, EXTERNAL_INPUT_SCHEMA_V2


PLANT_TABLES_ADAPTER_NAME = "plant_tables_v1"

SUPPORTED_TOP_LEVEL_FIELDS = frozenset(
    {
        "header",
        "defaults",
        "price_schedule",
        "workcenters",
        "workcenter_speeds",
        "operators",
        "orders",
        "routing_steps",
        "step_workcenters",
        "workcenter_calendar",
        "operator_calendar",
        "events",
    }
)
REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "header",
        "workcenters",
        "workcenter_speeds",
        "operators",
        "orders",
        "routing_steps",
        "step_workcenters",
    }
)

ADAPTER_MAPPING_SUMMARY = {
    "header": "metadata",
    "defaults": "defaults",
    "price_schedule": "defaults.electricity_prices",
    "workcenters": "machines",
    "workcenter_speeds": "machines[].modes",
    "operators": "workers",
    "orders": "jobs",
    "routing_steps": "jobs[].operations",
    "step_workcenters": "jobs[].operations[].processing_options",
    "workcenter_calendar": "calendar.machine_unavailability",
    "operator_calendar": "calendar.worker_shifts / calendar.worker_unavailability",
    "events": "events.machine_breakdowns / events.worker_absences",
}

HEADER_FIELD_MAP = {
    "plant_instance_id": "instance_id",
    "plant_instance_name": "instance_name",
    "instance_type": "instance_type",
    "planning_horizon_minutes": "planning_horizon",
    "period_minutes": "period_duration",
    "horizon_start_minutes": "horizon_start",
    "source_system": "source",
    "calibration_status": "calibration_status",
    "calibration_status_justification": "calibration_status_justification",
    "label": "label",
    "label_justification": "label_justification",
    "calibration_sources": "calibration_sources",
    "known_limitations": "known_limitations",
}
DEFAULTS_FIELD_MAP = {
    "default_ocra_rate": "default_ergonomic_risk",
    "electricity_price_per_kwh": "default_electricity_price",
    "carbon_kg_per_kwh": "carbon_emission_factor",
    "auxiliary_power_kw": "auxiliary_power_total",
}
WORKCENTER_FIELD_MAP = {
    "workcenter_code": "id",
    "display_name": "name",
    "default_speed_code": "default_mode_id",
    "processing_power_kw": "power_processing",
    "idle_power_kw": "power_idle",
    "setup_power_kw": "power_setup",
    "transport_power_kw": "power_transport",
    "startup_energy_kwh": "startup_energy",
    "setup_minutes": "setup_time",
    "auxiliary_power_share": "auxiliary_power_share",
}
WORKCENTER_MODE_FIELD_MAP = {
    "speed_code": "id",
    "display_name": "name",
    "speed_multiplier": "speed_factor",
    "power_multiplier": "power_multiplier",
    "tool_wear_rate": "tool_wear_rate",
}
OPERATOR_FIELD_MAP = {
    "operator_code": "id",
    "display_name": "name",
    "hourly_cost": "labor_cost_per_hour",
    "base_efficiency": "base_efficiency",
    "fatigue_rate": "fatigue_rate",
    "recovery_rate": "recovery_rate",
    "fatigue_max": "fatigue_max",
    "ocra_limit_per_shift": "ocra_max_per_shift",
    "ergonomic_tolerance": "ergonomic_tolerance",
    "min_rest_fraction": "min_rest_fraction",
    "max_consecutive_work_minutes": "max_consecutive_work_time",
    "learning_coefficient": "learning_coefficient",
}
ORDER_FIELD_MAP = {
    "order_code": "id",
    "release_time_minutes": "arrival_time",
    "due_time_minutes": "due_date",
    "priority_weight": "weight",
}
ROUTING_STEP_FIELD_MAP = {
    "step_code": "id",
    "ocra_rate": "ergonomic_risk_rate",
    "transport_minutes": "transport_time",
    "waiting_minutes": "waiting_time",
}


def adapt_plant_tables_payload(payload: Any, *, strict: bool = True) -> Dict[str, Any]:
    """Adapt a plant-table JSON payload into `sfjssp_external_v1/v2`."""

    adapter = _PlantTablesAdapter(strict=strict)
    return adapter.adapt(payload)


class _PlantTablesAdapter:
    def __init__(self, *, strict: bool):
        self.strict = strict
        self.issues: List[ValidationIssue] = []
        self.dropped_fields: set[str] = set()
        self.unsupported_sections: set[str] = set()

    def adapt(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise InterfaceValidationError(
                [issue(INVALID_TYPE, message="Adapter payload must be a JSON object.")]
            )

        unknown_top_level = sorted(set(payload) - SUPPORTED_TOP_LEVEL_FIELDS)
        self.unsupported_sections.update(unknown_top_level)
        if self.strict:
            for section_name in unknown_top_level:
                self.issues.append(
                    issue(
                        UNSUPPORTED_SECTION,
                        section_name,
                        message=(
                            f"Raw source section {section_name!r} is not supported by "
                            f"{PLANT_TABLES_ADAPTER_NAME}."
                        ),
                    )
                )

        for section_name in REQUIRED_TOP_LEVEL_FIELDS:
            if section_name not in payload:
                self.issues.append(
                    issue(
                        MISSING_REQUIRED,
                        section_name,
                        message=f"Raw source section {section_name!r} is required.",
                    )
                )

        header = self._expect_mapping(payload.get("header"), "header")
        defaults = self._expect_mapping(payload.get("defaults", {}), "defaults")
        price_schedule = self._expect_list(payload.get("price_schedule", []), "price_schedule")
        workcenters = self._expect_list(payload.get("workcenters"), "workcenters")
        workcenter_speeds = self._expect_list(
            payload.get("workcenter_speeds"), "workcenter_speeds"
        )
        operators = self._expect_list(payload.get("operators"), "operators")
        orders = self._expect_list(payload.get("orders"), "orders")
        routing_steps = self._expect_list(payload.get("routing_steps"), "routing_steps")
        step_workcenters = self._expect_list(
            payload.get("step_workcenters"), "step_workcenters"
        )
        workcenter_calendar = self._expect_list(
            payload.get("workcenter_calendar", []), "workcenter_calendar"
        )
        operator_calendar = self._expect_list(
            payload.get("operator_calendar", []), "operator_calendar"
        )
        events = self._expect_list(payload.get("events", []), "events")

        self._raise_if_needed()

        self._record_dropped_fields("header", header, set(HEADER_FIELD_MAP))
        self._record_dropped_fields("defaults", defaults, set(DEFAULTS_FIELD_MAP))
        external_defaults = self._map_fields(
            defaults,
            DEFAULTS_FIELD_MAP,
            section_name="defaults",
        )
        external_prices = self._build_price_schedule(price_schedule)
        if external_prices:
            external_defaults["electricity_prices"] = external_prices

        machines = self._build_machines(workcenters, workcenter_speeds)
        workers = self._build_workers(operators)
        jobs = self._build_jobs(orders, routing_steps, step_workcenters)
        calendar = self._build_calendar(workcenter_calendar, operator_calendar)
        external_events = self._build_events(events)

        schema_name = (
            EXTERNAL_INPUT_SCHEMA_V2
            if (
                calendar["machine_unavailability"]
                or calendar["worker_shifts"]
                or calendar["worker_unavailability"]
                or external_events["machine_breakdowns"]
                or external_events["worker_absences"]
                or "workcenter_calendar" in payload
                or "operator_calendar" in payload
                or "events" in payload
            )
            else EXTERNAL_INPUT_SCHEMA_V1
        )

        external_payload: Dict[str, Any] = {
            "schema": schema_name,
            "metadata": self._map_fields(
                header,
                HEADER_FIELD_MAP,
                section_name="header",
            ),
            "machines": machines,
            "workers": workers,
            "jobs": jobs,
        }
        if external_defaults:
            external_payload["defaults"] = external_defaults
        if schema_name == EXTERNAL_INPUT_SCHEMA_V2:
            external_payload["calendar"] = calendar
            external_payload["events"] = external_events

        self._raise_if_needed()
        return {
            "payload": external_payload,
            "provenance": {
                "adapter_name": PLANT_TABLES_ADAPTER_NAME,
                "source_schema": PLANT_TABLES_ADAPTER_NAME,
                "mapping_summary": dict(ADAPTER_MAPPING_SUMMARY),
                "dropped_fields": sorted(self.dropped_fields),
                "unsupported_sections": sorted(self.unsupported_sections),
            },
        }

    def _build_price_schedule(self, rows: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            row_mapping = self._expect_mapping(row, "price_schedule", index)
            self._record_dropped_fields(
                "price_schedule",
                row_mapping,
                {"period", "price_per_kwh"},
            )
            normalized.append(
                {
                    "period": row_mapping.get("period"),
                    "price": row_mapping.get("price_per_kwh"),
                }
            )
        return normalized

    def _build_machines(
        self,
        workcenters: List[Any],
        speed_rows: List[Any],
    ) -> List[Dict[str, Any]]:
        modes_by_machine: Dict[Any, List[Dict[str, Any]]] = {}
        for index, raw_mode in enumerate(speed_rows):
            mode_row = self._expect_mapping(raw_mode, "workcenter_speeds", index)
            self._record_dropped_fields(
                "workcenter_speeds",
                mode_row,
                set(WORKCENTER_MODE_FIELD_MAP) | {"workcenter_code"},
            )
            machine_id = mode_row.get("workcenter_code")
            modes_by_machine.setdefault(machine_id, []).append(
                self._map_fields(
                    mode_row,
                    WORKCENTER_MODE_FIELD_MAP,
                    section_name="workcenter_speeds",
                )
            )

        machines: List[Dict[str, Any]] = []
        for index, raw_machine in enumerate(workcenters):
            machine_row = self._expect_mapping(raw_machine, "workcenters", index)
            self._record_dropped_fields("workcenters", machine_row, set(WORKCENTER_FIELD_MAP))
            machine_payload = self._map_fields(
                machine_row,
                WORKCENTER_FIELD_MAP,
                section_name="workcenters",
            )
            machine_payload["modes"] = modes_by_machine.get(machine_row.get("workcenter_code"), [])
            machines.append(machine_payload)
        return machines

    def _build_workers(self, operators: List[Any]) -> List[Dict[str, Any]]:
        workers: List[Dict[str, Any]] = []
        for index, raw_worker in enumerate(operators):
            worker_row = self._expect_mapping(raw_worker, "operators", index)
            self._record_dropped_fields("operators", worker_row, set(OPERATOR_FIELD_MAP))
            workers.append(
                self._map_fields(
                    worker_row,
                    OPERATOR_FIELD_MAP,
                    section_name="operators",
                )
            )
        return workers

    def _build_jobs(
        self,
        orders: List[Any],
        routing_steps: List[Any],
        step_workcenters: List[Any],
    ) -> List[Dict[str, Any]]:
        operations_by_job: Dict[Any, List[Dict[str, Any]]] = {}
        operation_lookup: Dict[tuple[Any, Any], Dict[str, Any]] = {}

        for index, raw_step in enumerate(routing_steps):
            step_row = self._expect_mapping(raw_step, "routing_steps", index)
            self._record_dropped_fields(
                "routing_steps",
                step_row,
                set(ROUTING_STEP_FIELD_MAP) | {"order_code", "eligible_operator_codes"},
            )
            order_code = step_row.get("order_code")
            step_payload = self._map_fields(
                step_row,
                ROUTING_STEP_FIELD_MAP,
                section_name="routing_steps",
            )
            step_payload["eligible_workers"] = self._normalize_identifier_list(
                step_row.get("eligible_operator_codes")
            )
            step_payload["processing_options"] = []
            operations_by_job.setdefault(order_code, []).append(step_payload)
            operation_lookup[(order_code, step_row.get("step_code"))] = step_payload

        for index, raw_option in enumerate(step_workcenters):
            option_row = self._expect_mapping(raw_option, "step_workcenters", index)
            self._record_dropped_fields(
                "step_workcenters",
                option_row,
                {"order_code", "step_code", "workcenter_code", "speed_code", "run_minutes"},
            )
            option_payload = {
                "machine_id": option_row.get("workcenter_code"),
                "mode_id": option_row.get("speed_code"),
                "duration": option_row.get("run_minutes"),
            }
            operation = operation_lookup.get(
                (option_row.get("order_code"), option_row.get("step_code"))
            )
            if operation is None:
                self.issues.append(
                    issue(
                        INVALID_VALUE,
                        "step_workcenters",
                        index,
                        message=(
                            "step_workcenters row references an unknown "
                            "(order_code, step_code) pair."
                        ),
                    )
                )
                continue
            operation["processing_options"].append(option_payload)

        jobs: List[Dict[str, Any]] = []
        for index, raw_order in enumerate(orders):
            order_row = self._expect_mapping(raw_order, "orders", index)
            self._record_dropped_fields("orders", order_row, set(ORDER_FIELD_MAP))
            order_code = order_row.get("order_code")
            job_payload = self._map_fields(
                order_row,
                ORDER_FIELD_MAP,
                section_name="orders",
            )
            job_payload["operations"] = operations_by_job.get(order_code, [])
            jobs.append(job_payload)
        return jobs

    def _build_calendar(
        self,
        machine_rows: List[Any],
        worker_rows: List[Any],
    ) -> Dict[str, Any]:
        machine_unavailability: List[Dict[str, Any]] = []
        worker_shifts: List[Dict[str, Any]] = []
        worker_unavailability: List[Dict[str, Any]] = []

        for index, raw_row in enumerate(machine_rows):
            row = self._expect_mapping(raw_row, "workcenter_calendar", index)
            self._record_dropped_fields(
                "workcenter_calendar",
                row,
                {"workcenter_code", "start_time_minutes", "end_time_minutes", "reason", "source", "details"},
            )
            machine_unavailability.append(
                {
                    "machine_id": row.get("workcenter_code"),
                    "start_time": row.get("start_time_minutes"),
                    "end_time": row.get("end_time_minutes"),
                    "reason": row.get("reason"),
                    "source": row.get("source"),
                    "details": row.get("details"),
                }
            )

        for index, raw_row in enumerate(worker_rows):
            row = self._expect_mapping(raw_row, "operator_calendar", index)
            self._record_dropped_fields(
                "operator_calendar",
                row,
                {
                    "operator_code",
                    "entry_type",
                    "start_time_minutes",
                    "end_time_minutes",
                    "shift_label",
                    "reason",
                    "source",
                    "details",
                },
            )
            entry_type = row.get("entry_type")
            payload = {
                "worker_id": row.get("operator_code"),
                "start_time": row.get("start_time_minutes"),
                "end_time": row.get("end_time_minutes"),
            }
            if entry_type == "shift_window":
                payload["shift_label"] = row.get("shift_label")
                payload["details"] = row.get("details")
                worker_shifts.append(payload)
            elif entry_type == "unavailability_window":
                payload["reason"] = row.get("reason")
                payload["source"] = row.get("source")
                payload["details"] = row.get("details")
                worker_unavailability.append(payload)
            else:
                self.issues.append(
                    issue(
                        INVALID_VALUE,
                        "operator_calendar",
                        index,
                        "entry_type",
                        message=(
                            "operator_calendar entry_type must be "
                            "'shift_window' or 'unavailability_window'."
                        ),
                    )
                )

        return {
            "machine_unavailability": machine_unavailability,
            "worker_shifts": worker_shifts,
            "worker_unavailability": worker_unavailability,
        }

    def _build_events(self, rows: List[Any]) -> Dict[str, Any]:
        machine_breakdowns: List[Dict[str, Any]] = []
        worker_absences: List[Dict[str, Any]] = []

        for index, raw_row in enumerate(rows):
            row = self._expect_mapping(raw_row, "events", index)
            self._record_dropped_fields(
                "events",
                row,
                {
                    "event_type",
                    "workcenter_code",
                    "operator_code",
                    "start_time_minutes",
                    "end_time_minutes",
                    "repair_duration_minutes",
                    "source",
                    "event_id",
                    "details",
                },
            )
            event_type = row.get("event_type")
            if event_type == "machine_breakdown":
                machine_breakdowns.append(
                    {
                        "machine_id": row.get("workcenter_code"),
                        "start_time": row.get("start_time_minutes"),
                        "repair_duration": row.get("repair_duration_minutes"),
                        "source": row.get("source"),
                        "event_id": row.get("event_id"),
                        "details": row.get("details"),
                    }
                )
            elif event_type == "worker_absence":
                worker_absences.append(
                    {
                        "worker_id": row.get("operator_code"),
                        "start_time": row.get("start_time_minutes"),
                        "end_time": row.get("end_time_minutes"),
                        "source": row.get("source"),
                        "event_id": row.get("event_id"),
                        "details": row.get("details"),
                    }
                )
            else:
                self.issues.append(
                    issue(
                        INVALID_VALUE,
                        "events",
                        index,
                        "event_type",
                        message=(
                            "events.event_type must be 'machine_breakdown' or "
                            "'worker_absence'."
                        ),
                    )
                )

        return {
            "machine_breakdowns": machine_breakdowns,
            "worker_absences": worker_absences,
        }

    def _map_fields(
        self,
        row: Dict[str, Any],
        field_map: Dict[str, str],
        *,
        section_name: str,
    ) -> Dict[str, Any]:
        _ = section_name
        mapped: Dict[str, Any] = {}
        for source_key, target_key in field_map.items():
            if source_key in row:
                mapped[target_key] = row[source_key]
        return mapped

    def _record_dropped_fields(
        self,
        section_name: str,
        row: Dict[str, Any],
        known_fields: Iterable[str],
    ) -> None:
        allowed = set(known_fields)
        for key in row:
            if key not in allowed:
                self.dropped_fields.add(f"{section_name}.{key}")

    def _expect_mapping(
        self,
        value: Any,
        section_name: str,
        index: Optional[int] = None,
    ) -> Dict[str, Any]:
        path: List[object] = [section_name]
        if index is not None:
            path.append(index)
        if value is None:
            return {}
        if not isinstance(value, dict):
            self.issues.append(
                issue(
                    INVALID_TYPE,
                    *path,
                    message=f"{section_name} entries must be objects.",
                )
            )
            return {}
        return dict(value)

    def _expect_list(self, value: Any, section_name: str) -> List[Any]:
        if value is None:
            return []
        if not isinstance(value, list):
            self.issues.append(
                issue(
                    INVALID_TYPE,
                    section_name,
                    message=f"Raw source section {section_name!r} must be a list.",
                )
            )
            return []
        return list(value)

    def _normalize_identifier_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, str):
            return [item.strip() for item in value.split("|") if item.strip()]
        return [value]

    def _raise_if_needed(self) -> None:
        if self.issues:
            raise InterfaceValidationError(self.issues)
