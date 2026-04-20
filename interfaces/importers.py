"""
Strict external input normalization and JSON import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .adapters import adapt_source_payload
from .errors import (
    DUPLICATE_ID,
    EMPTY_COLLECTION,
    INVALID_REFERENCE,
    INVALID_TYPE,
    INVALID_VALUE,
    InterfaceValidationError,
    MISSING_REQUIRED,
    SCHEMA_MISMATCH,
    UNKNOWN_FIELD,
    UNSUPPORTED_SECTION,
    ValidationIssue,
    issue,
)
from .site_profiles import apply_site_profile
from .schema import (
    DEFAULT_HORIZON_START,
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_PERIOD_DURATION,
    EXTERNAL_INPUT_SCHEMA,
    REQUIRED_MACHINE_BREAKDOWN_FIELDS,
    REQUIRED_ELECTRICITY_PRICE_FIELDS,
    REQUIRED_JOB_FIELDS,
    REQUIRED_MACHINE_FIELDS,
    REQUIRED_MACHINE_MODE_FIELDS,
    REQUIRED_METADATA_FIELDS,
    REQUIRED_OPERATION_FIELDS,
    REQUIRED_PROCESSING_OPTION_FIELDS,
    REQUIRED_TOP_LEVEL_FIELDS,
    REQUIRED_MACHINE_UNAVAILABILITY_FIELDS,
    REQUIRED_WORKER_ABSENCE_FIELDS,
    REQUIRED_WORKER_SHIFT_FIELDS,
    REQUIRED_WORKER_UNAVAILABILITY_FIELDS,
    REQUIRED_WORKER_FIELDS,
    SUPPORTED_DEFAULT_FIELDS,
    SUPPORTED_ELECTRICITY_PRICE_FIELDS,
    SUPPORTED_EVENTS_FIELDS,
    SUPPORTED_JOB_FIELDS,
    SUPPORTED_CALENDAR_FIELDS,
    SUPPORTED_MACHINE_BREAKDOWN_FIELDS,
    SUPPORTED_MACHINE_FIELDS,
    SUPPORTED_MACHINE_MODE_FIELDS,
    SUPPORTED_MACHINE_UNAVAILABILITY_FIELDS,
    SUPPORTED_METADATA_FIELDS,
    SUPPORTED_OPERATION_FIELDS,
    SUPPORTED_PROCESSING_OPTION_FIELDS,
    SUPPORTED_WORKER_ABSENCE_FIELDS,
    SUPPORTED_WORKER_SHIFT_FIELDS,
    SUPPORTED_WORKER_UNAVAILABILITY_FIELDS,
    SUPPORTED_WORKER_FIELDS,
    SUPPORTED_EXTERNAL_INPUT_SCHEMAS,
    get_reserved_unsupported_top_level_fields,
    get_supported_top_level_fields,
    normalize_external_schema_name,
    schema_supports_calendar_events,
)
from .types import ExternalId, IdentifierMaps, ImportedInstance, TypedIdKey

try:
    from ..sfjssp_model.instance import (
        CALIBRATION_EVIDENCE_REQUIRED_STATUSES,
        PUBLIC_CALIBRATION_STATUSES,
        InstanceLabel,
        InstanceType,
        SFJSSPInstance,
        instance_label_from_public_calibration_status,
        normalize_public_calibration_status,
    )
    from ..sfjssp_model.job import Job, Operation
    from ..sfjssp_model.machine import Machine, MachineMode
    from ..sfjssp_model.period_clock import PeriodClock
    from ..sfjssp_model.worker import Worker
except ImportError:  # pragma: no cover - supports repo-root imports
    from sfjssp_model.instance import (
        CALIBRATION_EVIDENCE_REQUIRED_STATUSES,
        PUBLIC_CALIBRATION_STATUSES,
        InstanceLabel,
        InstanceType,
        SFJSSPInstance,
        instance_label_from_public_calibration_status,
        normalize_public_calibration_status,
    )
    from sfjssp_model.job import Job, Operation
    from sfjssp_model.machine import Machine, MachineMode
    from sfjssp_model.period_clock import PeriodClock
    from sfjssp_model.worker import Worker


def load_instance_from_json(
    path: Any,
    strict: bool = True,
    *,
    adapter_name: Optional[str] = None,
    site_profile_name: Optional[str] = None,
) -> ImportedInstance:
    """Load, validate, and normalize a JSON document through the interface stack."""

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return import_instance_from_dict(
        payload,
        strict=strict,
        adapter_name=adapter_name,
        site_profile_name=site_profile_name,
        raw_source_id=str(json_path),
        raw_input_format="json",
    )


def import_instance_from_dict(
    payload: Any,
    strict: bool = True,
    *,
    adapter_name: Optional[str] = None,
    site_profile_name: Optional[str] = None,
    raw_source_id: str = "in_memory",
    raw_input_format: str = "dict",
) -> ImportedInstance:
    """Validate a raw payload and construct a canonical `SFJSSPInstance`."""

    adapter_provenance: Optional[Dict[str, Any]] = None
    external_payload = payload
    if adapter_name is not None:
        adapted = adapt_source_payload(
            external_payload,
            adapter_name=adapter_name,
            strict=strict,
        )
        external_payload = adapted["payload"]
        adapter_provenance = dict(adapted["provenance"])

    payload_defaults_present = bool(
        isinstance(external_payload, dict) and external_payload.get("defaults")
    )
    site_profile_provenance: Optional[Dict[str, Any]] = None
    if site_profile_name is not None:
        profiled = apply_site_profile(
            external_payload,
            profile_name=site_profile_name,
        )
        external_payload = profiled["payload"]
        site_profile_provenance = dict(profiled["provenance"])

    normalizer = _ExternalPayloadNormalizer(strict=strict)
    imported = normalizer.import_payload(external_payload)
    return ImportedInstance(
        schema=imported.schema,
        normalized_payload=imported.normalized_payload,
        instance=imported.instance,
        id_maps=imported.id_maps,
        provenance=_build_import_provenance(
            raw_source_id=raw_source_id,
            raw_input_format=raw_input_format,
            external_schema=imported.schema,
            instance=imported.instance,
            adapter_provenance=adapter_provenance,
            site_profile_provenance=site_profile_provenance,
            payload_defaults_present=payload_defaults_present,
        ),
    )


def _build_import_provenance(
    *,
    raw_source_id: str,
    raw_input_format: str,
    external_schema: str,
    instance: SFJSSPInstance,
    adapter_provenance: Optional[Dict[str, Any]],
    site_profile_provenance: Optional[Dict[str, Any]],
    payload_defaults_present: bool,
) -> Dict[str, Any]:
    raw_source = {
        "source_id": raw_source_id,
        "input_format": raw_input_format,
    }
    if adapter_provenance is not None:
        raw_source["source_schema"] = adapter_provenance.get("source_schema")
    return {
        "raw_source": raw_source,
        "external_schema": external_schema,
        "calibration": instance.build_calibration_record(),
        "adapter": adapter_provenance,
        "site_profile": site_profile_provenance,
        "parameter_sources": {
            "payload_defaults_present": payload_defaults_present,
            "site_profile_applied": site_profile_provenance is not None,
            "calibration_sources_present": bool(instance.calibration_sources),
        },
    }


class _ExternalPayloadNormalizer:
    def __init__(self, strict: bool):
        self.strict = strict
        self.issues: List[ValidationIssue] = []
        self.schema_name = EXTERNAL_INPUT_SCHEMA

    def import_payload(self, payload: Any) -> ImportedInstance:
        normalized = self._normalize_payload(payload)
        self._raise_if_needed()
        id_maps = self._build_identifier_maps(normalized)
        instance = self._build_instance(normalized, id_maps)
        self._raise_if_needed()
        return ImportedInstance(
            schema=self.schema_name,
            normalized_payload=normalized,
            instance=instance,
            id_maps=id_maps,
        )

    def _normalize_payload(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            self.issues.append(
                issue(INVALID_TYPE, message="Top-level payload must be a JSON object.")
            )
            return {}

        declared_schema = payload.get("schema")
        normalized_schema = normalize_external_schema_name(declared_schema)
        if normalized_schema is None:
            self.issues.append(
                issue(
                    SCHEMA_MISMATCH,
                    "schema",
                    message=(
                        "Expected one of "
                        f"{sorted(SUPPORTED_EXTERNAL_INPUT_SCHEMAS)!r}, got {declared_schema!r}."
                    ),
                )
            )
            self.schema_name = EXTERNAL_INPUT_SCHEMA
        else:
            self.schema_name = normalized_schema

        self._ensure_allowed_keys(payload, get_supported_top_level_fields(self.schema_name), ())
        self._require_keys(payload, REQUIRED_TOP_LEVEL_FIELDS, ())

        for section in get_reserved_unsupported_top_level_fields(self.schema_name):
            if section in payload:
                self.issues.append(
                    issue(
                        UNSUPPORTED_SECTION,
                        section,
                        message=(
                            f"Top-level section {section!r} is reserved and unsupported "
                            f"in {self.schema_name}."
                        ),
                        hint=(
                            "Use metadata, defaults, machines, workers, jobs, and the "
                            "documented schema-specific sections only."
                        ),
                    )
                )

        metadata = self._normalize_metadata(payload.get("metadata"), ("metadata",))
        defaults = self._normalize_defaults(payload.get("defaults", {}), ("defaults",))
        machines = self._normalize_machine_list(payload.get("machines"), ("machines",))
        workers = self._normalize_worker_list(payload.get("workers"), ("workers",), metadata)
        jobs = self._normalize_job_list(payload.get("jobs"), ("jobs",), defaults)
        calendar = None
        events = None
        if schema_supports_calendar_events(self.schema_name):
            calendar = self._normalize_calendar(payload.get("calendar", {}), ("calendar",))
            events = self._normalize_events(payload.get("events", {}), ("events",))

        if metadata.get("instance_type") == "static":
            if any(job["arrival_time"] > 0.0 for job in jobs):
                self.issues.append(
                    issue(
                        INVALID_VALUE,
                        "metadata",
                        "instance_type",
                        message="Static instances cannot contain jobs with positive arrival_time.",
                        hint="Set metadata.instance_type to 'dynamic' or use arrival_time = 0 for every job.",
                    )
                )

        normalized_payload = {
            "schema": self.schema_name,
            "metadata": metadata,
            "defaults": defaults,
            "machines": machines,
            "workers": workers,
            "jobs": jobs,
        }
        if calendar is not None:
            normalized_payload["calendar"] = calendar
            normalized_payload["events"] = events
        return normalized_payload

    def _normalize_metadata(self, metadata: Any, path: Tuple[object, ...]) -> Dict[str, Any]:
        if not isinstance(metadata, dict):
            self.issues.append(issue(INVALID_TYPE, *path, message="metadata must be an object."))
            return {}

        self._ensure_allowed_keys(metadata, SUPPORTED_METADATA_FIELDS, path)
        self._require_keys(metadata, REQUIRED_METADATA_FIELDS, path)

        return {
            "instance_id": self._normalize_identifier(metadata.get("instance_id"), path + ("instance_id",)),
            "instance_name": self._optional_string(metadata.get("instance_name"), path + ("instance_name",), ""),
            "instance_type": self._optional_enum(
                metadata.get("instance_type"),
                {member.value for member in InstanceType},
                path + ("instance_type",),
                DEFAULT_INSTANCE_TYPE,
            ),
            "planning_horizon": self._optional_number(
                metadata.get("planning_horizon"),
                path + ("planning_horizon",),
                1000.0,
                minimum=0.0,
            ),
            "period_duration": self._optional_number(
                metadata.get("period_duration"),
                path + ("period_duration",),
                DEFAULT_PERIOD_DURATION,
                minimum=1e-9,
            ),
            "horizon_start": self._optional_number(
                metadata.get("horizon_start"),
                path + ("horizon_start",),
                DEFAULT_HORIZON_START,
            ),
            "source": self._optional_string(metadata.get("source"), path + ("source",), ""),
            **self._normalize_calibration_metadata(metadata, path),
            "known_limitations": self._optional_string_list(
                metadata.get("known_limitations"),
                path + ("known_limitations",),
            ),
        }

    def _normalize_calibration_metadata(
        self,
        metadata: Dict[str, Any],
        path: Tuple[object, ...],
    ) -> Dict[str, Any]:
        raw_status = metadata.get("calibration_status")
        legacy_status = metadata.get("label")
        normalized_status = self._normalize_calibration_status(
            raw_status,
            path + ("calibration_status",),
        )
        normalized_legacy_status = self._normalize_calibration_status(
            legacy_status,
            path + ("label",),
        )
        if (
            normalized_status is not None
            and normalized_legacy_status is not None
            and normalized_status != normalized_legacy_status
        ):
            self.issues.append(
                issue(
                    INVALID_VALUE,
                    *path,
                    "calibration_status",
                    message=(
                        "metadata.calibration_status and compatibility alias "
                        "metadata.label disagree."
                    ),
                )
            )

        calibration_status = (
            normalized_status
            or normalized_legacy_status
            or InstanceLabel.FULLY_SYNTHETIC.value
        )

        raw_justification = metadata.get("calibration_status_justification")
        legacy_justification = metadata.get("label_justification")
        if (
            raw_justification is not None
            and legacy_justification is not None
            and raw_justification != legacy_justification
        ):
            self.issues.append(
                issue(
                    INVALID_VALUE,
                    *path,
                    "calibration_status_justification",
                    message=(
                        "metadata.calibration_status_justification and compatibility "
                        "alias metadata.label_justification disagree."
                    ),
                )
            )
        calibration_justification = self._optional_string(
            raw_justification if raw_justification is not None else legacy_justification,
            path + ("calibration_status_justification",),
            "Imported external document",
        )
        calibration_sources = self._optional_string_list(
            metadata.get("calibration_sources"),
            path + ("calibration_sources",),
        )

        if calibration_status in CALIBRATION_EVIDENCE_REQUIRED_STATUSES:
            if raw_justification is None and legacy_justification is None:
                self.issues.append(
                    issue(
                        MISSING_REQUIRED,
                        *path,
                        "calibration_status_justification",
                        message=(
                            f"Calibration status {calibration_status!r} requires an explicit "
                            "justification."
                        ),
                    )
                )
            if not calibration_sources:
                self.issues.append(
                    issue(
                        MISSING_REQUIRED,
                        *path,
                        "calibration_sources",
                        message=(
                            f"Calibration status {calibration_status!r} requires at least one "
                            "calibration source reference."
                        ),
                    )
                )

        return {
            "calibration_status": calibration_status,
            "calibration_status_justification": calibration_justification,
            "calibration_sources": calibration_sources,
        }

    def _normalize_calibration_status(
        self,
        value: Any,
        path: Tuple[object, ...],
    ) -> Optional[str]:
        if value is None:
            return None
        normalized = normalize_public_calibration_status(value)
        if normalized is None:
            self.issues.append(
                issue(
                    INVALID_VALUE,
                    *path,
                    message=(
                        "Expected one of "
                        f"{sorted(PUBLIC_CALIBRATION_STATUSES)!r}, got {value!r}."
                    ),
                    hint=(
                        "Use metadata.calibration_status for public payloads; "
                        "metadata.label remains a compatibility alias only."
                    ),
                )
            )
        return normalized

    def _normalize_defaults(self, defaults: Any, path: Tuple[object, ...]) -> Dict[str, Any]:
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            self.issues.append(issue(INVALID_TYPE, *path, message="defaults must be an object."))
            return {}

        self._ensure_allowed_keys(defaults, SUPPORTED_DEFAULT_FIELDS, path)

        raw_prices = defaults.get("electricity_prices", [])
        if raw_prices in (None, {}):
            raw_prices = []
        if not isinstance(raw_prices, list):
            self.issues.append(
                issue(
                    INVALID_TYPE,
                    *path,
                    "electricity_prices",
                    message="electricity_prices must be a list of {period, price} objects.",
                )
            )
            raw_prices = []

        normalized_prices: List[Dict[str, float]] = []
        for index, raw_price in enumerate(raw_prices):
            price_path = path + ("electricity_prices", index)
            if not isinstance(raw_price, dict):
                self.issues.append(issue(INVALID_TYPE, *price_path, message="Each electricity_prices entry must be an object."))
                continue
            self._ensure_allowed_keys(raw_price, SUPPORTED_ELECTRICITY_PRICE_FIELDS, price_path)
            self._require_keys(raw_price, REQUIRED_ELECTRICITY_PRICE_FIELDS, price_path)
            normalized_prices.append(
                {
                    "period": self._required_int(raw_price.get("period"), price_path + ("period",), minimum=0),
                    "price": self._required_number(raw_price.get("price"), price_path + ("price",), minimum=0.0),
                }
            )
        normalized_prices.sort(key=lambda item: item["period"])

        return {
            "default_ergonomic_risk": self._optional_number(
                defaults.get("default_ergonomic_risk"),
                path + ("default_ergonomic_risk",),
                0.0,
                minimum=0.0,
            ),
            "default_electricity_price": self._optional_number(
                defaults.get("default_electricity_price"),
                path + ("default_electricity_price",),
                0.10,
                minimum=0.0,
            ),
            "electricity_prices": normalized_prices,
            "carbon_emission_factor": self._optional_number(
                defaults.get("carbon_emission_factor"),
                path + ("carbon_emission_factor",),
                0.5,
                minimum=0.0,
            ),
            "auxiliary_power_total": self._optional_number(
                defaults.get("auxiliary_power_total"),
                path + ("auxiliary_power_total",),
                50.0,
                minimum=0.0,
            ),
        }

    def _normalize_machine_list(self, machines: Any, path: Tuple[object, ...]) -> List[Dict[str, Any]]:
        return self._normalize_resource_list(
            machines,
            path,
            SUPPORTED_MACHINE_FIELDS,
            REQUIRED_MACHINE_FIELDS,
            self._normalize_machine,
        )

    def _normalize_calendar(self, calendar: Any, path: Tuple[object, ...]) -> Dict[str, Any]:
        if calendar is None:
            calendar = {}
        if not isinstance(calendar, dict):
            self.issues.append(issue(INVALID_TYPE, *path, message="calendar must be an object."))
            return {
                "machine_unavailability": [],
                "worker_shifts": [],
                "worker_unavailability": [],
            }

        self._ensure_allowed_keys(calendar, SUPPORTED_CALENDAR_FIELDS, path)

        return {
            "machine_unavailability": self._normalize_unavailability_list(
                calendar.get("machine_unavailability", []),
                path + ("machine_unavailability",),
                resource_field="machine_id",
                supported_fields=SUPPORTED_MACHINE_UNAVAILABILITY_FIELDS,
                required_fields=REQUIRED_MACHINE_UNAVAILABILITY_FIELDS,
                default_reason="calendar_unavailable",
                default_source="calendar",
            ),
            "worker_shifts": self._normalize_worker_shift_list(
                calendar.get("worker_shifts", []),
                path + ("worker_shifts",),
            ),
            "worker_unavailability": self._normalize_unavailability_list(
                calendar.get("worker_unavailability", []),
                path + ("worker_unavailability",),
                resource_field="worker_id",
                supported_fields=SUPPORTED_WORKER_UNAVAILABILITY_FIELDS,
                required_fields=REQUIRED_WORKER_UNAVAILABILITY_FIELDS,
                default_reason="calendar_unavailable",
                default_source="calendar",
            ),
        }

    def _normalize_events(self, events: Any, path: Tuple[object, ...]) -> Dict[str, Any]:
        if events is None:
            events = {}
        if not isinstance(events, dict):
            self.issues.append(issue(INVALID_TYPE, *path, message="events must be an object."))
            return {
                "machine_breakdowns": [],
                "worker_absences": [],
            }

        self._ensure_allowed_keys(events, SUPPORTED_EVENTS_FIELDS, path)

        return {
            "machine_breakdowns": self._normalize_machine_breakdown_list(
                events.get("machine_breakdowns", []),
                path + ("machine_breakdowns",),
            ),
            "worker_absences": self._normalize_worker_absence_list(
                events.get("worker_absences", []),
                path + ("worker_absences",),
            ),
        }

    def _normalize_unavailability_list(
        self,
        value: Any,
        path: Tuple[object, ...],
        *,
        resource_field: str,
        supported_fields: Iterable[str],
        required_fields: Iterable[str],
        default_reason: str,
        default_source: str,
    ) -> List[Dict[str, Any]]:
        if value is None:
            value = []
        if not isinstance(value, list):
            self.issues.append(issue(INVALID_TYPE, *path, message="Expected a list."))
            return []

        normalized: List[Tuple[Tuple[str, float, float, str, str], Dict[str, Any]]] = []
        for index, raw_item in enumerate(value):
            item_path = path + (index,)
            if not isinstance(raw_item, dict):
                self.issues.append(issue(INVALID_TYPE, *item_path, message="Each entry must be an object."))
                continue
            self._ensure_allowed_keys(raw_item, supported_fields, item_path)
            self._require_keys(raw_item, required_fields, item_path)

            resource_id = self._normalize_identifier(raw_item.get(resource_field), item_path + (resource_field,))
            start_time = self._required_number(raw_item.get("start_time"), item_path + ("start_time",), minimum=0.0)
            end_time = self._required_number(raw_item.get("end_time"), item_path + ("end_time",), minimum=0.0)
            if end_time < start_time:
                self.issues.append(
                    issue(
                        INVALID_VALUE,
                        *item_path,
                        "end_time",
                        message="end_time must be greater than or equal to start_time.",
                    )
                )
            reason = self._optional_string(raw_item.get("reason"), item_path + ("reason",), default_reason)
            source = self._optional_string(raw_item.get("source"), item_path + ("source",), default_source)
            details = self._optional_mapping(raw_item.get("details"), item_path + ("details",))
            normalized_item = {
                resource_field: resource_id,
                "start_time": start_time,
                "end_time": end_time,
                "reason": reason,
                "source": source,
                "details": details,
            }
            sort_key = (
                self._identifier_key(resource_id),
                start_time,
                end_time,
                reason,
                source,
            )
            normalized.append((sort_key, normalized_item))

        normalized.sort(key=lambda pair: pair[0])
        return [item for _, item in normalized]

    def _normalize_worker_shift_list(
        self,
        value: Any,
        path: Tuple[object, ...],
    ) -> List[Dict[str, Any]]:
        if value is None:
            value = []
        if not isinstance(value, list):
            self.issues.append(issue(INVALID_TYPE, *path, message="Expected a list."))
            return []

        normalized: List[Tuple[Tuple[str, float, float, str], Dict[str, Any]]] = []
        for index, raw_item in enumerate(value):
            item_path = path + (index,)
            if not isinstance(raw_item, dict):
                self.issues.append(
                    issue(INVALID_TYPE, *item_path, message="Each entry must be an object.")
                )
                continue
            self._ensure_allowed_keys(raw_item, SUPPORTED_WORKER_SHIFT_FIELDS, item_path)
            self._require_keys(raw_item, REQUIRED_WORKER_SHIFT_FIELDS, item_path)

            worker_id = self._normalize_identifier(raw_item.get("worker_id"), item_path + ("worker_id",))
            start_time = self._required_number(raw_item.get("start_time"), item_path + ("start_time",), minimum=0.0)
            end_time = self._required_number(raw_item.get("end_time"), item_path + ("end_time",), minimum=0.0)
            if end_time < start_time:
                self.issues.append(
                    issue(
                        INVALID_VALUE,
                        *item_path,
                        "end_time",
                        message="end_time must be greater than or equal to start_time.",
                    )
                )
            shift_label = self._optional_string(
                raw_item.get("shift_label"),
                item_path + ("shift_label",),
                "shift",
            )
            details = self._optional_mapping(raw_item.get("details"), item_path + ("details",))
            normalized_item = {
                "worker_id": worker_id,
                "start_time": start_time,
                "end_time": end_time,
                "shift_label": shift_label,
                "details": details,
            }
            normalized.append(
                (
                    (self._identifier_key(worker_id), start_time, end_time, shift_label),
                    normalized_item,
                )
            )

        normalized.sort(key=lambda pair: pair[0])
        return [item for _, item in normalized]

    def _normalize_machine_breakdown_list(
        self,
        value: Any,
        path: Tuple[object, ...],
    ) -> List[Dict[str, Any]]:
        if value is None:
            value = []
        if not isinstance(value, list):
            self.issues.append(issue(INVALID_TYPE, *path, message="Expected a list."))
            return []

        normalized: List[Tuple[Tuple[str, float, float, str, str], Dict[str, Any]]] = []
        for index, raw_item in enumerate(value):
            item_path = path + (index,)
            if not isinstance(raw_item, dict):
                self.issues.append(issue(INVALID_TYPE, *item_path, message="Each entry must be an object."))
                continue
            self._ensure_allowed_keys(raw_item, SUPPORTED_MACHINE_BREAKDOWN_FIELDS, item_path)
            self._require_keys(raw_item, REQUIRED_MACHINE_BREAKDOWN_FIELDS, item_path)

            machine_id = self._normalize_identifier(raw_item.get("machine_id"), item_path + ("machine_id",))
            start_time = self._required_number(raw_item.get("start_time"), item_path + ("start_time",), minimum=0.0)
            repair_duration = self._required_number(
                raw_item.get("repair_duration"),
                item_path + ("repair_duration",),
                minimum=1e-9,
            )
            source = self._optional_string(raw_item.get("source"), item_path + ("source",), "event")
            event_id = self._optional_string(raw_item.get("event_id"), item_path + ("event_id",), "")
            details = self._optional_mapping(raw_item.get("details"), item_path + ("details",))
            normalized_item = {
                "machine_id": machine_id,
                "start_time": start_time,
                "repair_duration": repair_duration,
                "source": source,
                "event_id": event_id,
                "details": details,
            }
            normalized.append(
                (
                    (self._identifier_key(machine_id), start_time, repair_duration, source, event_id),
                    normalized_item,
                )
            )

        normalized.sort(key=lambda pair: pair[0])
        return [item for _, item in normalized]

    def _normalize_worker_absence_list(
        self,
        value: Any,
        path: Tuple[object, ...],
    ) -> List[Dict[str, Any]]:
        if value is None:
            value = []
        if not isinstance(value, list):
            self.issues.append(issue(INVALID_TYPE, *path, message="Expected a list."))
            return []

        normalized: List[Tuple[Tuple[str, float, float, str, str], Dict[str, Any]]] = []
        for index, raw_item in enumerate(value):
            item_path = path + (index,)
            if not isinstance(raw_item, dict):
                self.issues.append(issue(INVALID_TYPE, *item_path, message="Each entry must be an object."))
                continue
            self._ensure_allowed_keys(raw_item, SUPPORTED_WORKER_ABSENCE_FIELDS, item_path)
            self._require_keys(raw_item, REQUIRED_WORKER_ABSENCE_FIELDS, item_path)

            worker_id = self._normalize_identifier(raw_item.get("worker_id"), item_path + ("worker_id",))
            start_time = self._required_number(raw_item.get("start_time"), item_path + ("start_time",), minimum=0.0)
            end_time = self._required_number(raw_item.get("end_time"), item_path + ("end_time",), minimum=0.0)
            if end_time < start_time:
                self.issues.append(
                    issue(
                        INVALID_VALUE,
                        *item_path,
                        "end_time",
                        message="end_time must be greater than or equal to start_time.",
                    )
                )
            source = self._optional_string(raw_item.get("source"), item_path + ("source",), "event")
            event_id = self._optional_string(raw_item.get("event_id"), item_path + ("event_id",), "")
            details = self._optional_mapping(raw_item.get("details"), item_path + ("details",))
            normalized_item = {
                "worker_id": worker_id,
                "start_time": start_time,
                "end_time": end_time,
                "source": source,
                "event_id": event_id,
                "details": details,
            }
            normalized.append(
                (
                    (self._identifier_key(worker_id), start_time, end_time, source, event_id),
                    normalized_item,
                )
            )

        normalized.sort(key=lambda pair: pair[0])
        return [item for _, item in normalized]

    def _normalize_worker_list(
        self,
        workers: Any,
        path: Tuple[object, ...],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return self._normalize_resource_list(
            workers,
            path,
            SUPPORTED_WORKER_FIELDS,
            REQUIRED_WORKER_FIELDS,
            lambda item, item_path: self._normalize_worker(item, item_path, metadata),
        )

    def _normalize_job_list(
        self,
        jobs: Any,
        path: Tuple[object, ...],
        defaults: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return self._normalize_resource_list(
            jobs,
            path,
            SUPPORTED_JOB_FIELDS,
            REQUIRED_JOB_FIELDS,
            lambda item, item_path: self._normalize_job(item, item_path, defaults),
        )

    def _normalize_resource_list(
        self,
        items: Any,
        path: Tuple[object, ...],
        supported_fields: Iterable[str],
        required_fields: Iterable[str],
        normalizer,
    ) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            self.issues.append(issue(INVALID_TYPE, *path, message="Expected a list."))
            return []
        if not items:
            self.issues.append(issue(EMPTY_COLLECTION, *path, message="List must not be empty."))
            return []

        normalized: List[Tuple[TypedIdKey, Dict[str, Any]]] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(items):
            item_path = path + (index,)
            if not isinstance(item, dict):
                self.issues.append(issue(INVALID_TYPE, *item_path, message="Each entry must be an object."))
                continue
            self._ensure_allowed_keys(item, supported_fields, item_path)
            self._require_keys(item, required_fields, item_path)
            external_id = self._normalize_identifier(item.get("id"), item_path + ("id",))
            identifier_key = self._identifier_key(external_id)
            if identifier_key in seen_ids:
                self.issues.append(issue(DUPLICATE_ID, *item_path, "id", message=f"Duplicate identifier {external_id!r}."))
            seen_ids.add(identifier_key)
            normalized.append((identifier_key, normalizer(item, item_path)))

        normalized.sort(key=lambda pair: pair[0])
        return [item for _, item in normalized]

    def _normalize_machine(self, machine: Dict[str, Any], path: Tuple[object, ...]) -> Dict[str, Any]:
        raw_modes = machine.get("modes")
        if not isinstance(raw_modes, list):
            self.issues.append(issue(INVALID_TYPE, *path, "modes", message="modes must be a non-empty list."))
            raw_modes = []
        elif not raw_modes:
            self.issues.append(issue(EMPTY_COLLECTION, *path, "modes", message="modes must not be empty."))

        seen_mode_ids: set[str] = set()
        normalized_modes: List[Tuple[str, Dict[str, Any]]] = []
        for index, raw_mode in enumerate(raw_modes):
            mode_path = path + ("modes", index)
            if not isinstance(raw_mode, dict):
                self.issues.append(issue(INVALID_TYPE, *mode_path, message="Each mode must be an object."))
                continue
            self._ensure_allowed_keys(raw_mode, SUPPORTED_MACHINE_MODE_FIELDS, mode_path)
            self._require_keys(raw_mode, REQUIRED_MACHINE_MODE_FIELDS, mode_path)
            mode_id = self._normalize_identifier(raw_mode.get("id"), mode_path + ("id",))
            mode_key = self._identifier_key(mode_id)
            if mode_key in seen_mode_ids:
                self.issues.append(issue(DUPLICATE_ID, *mode_path, "id", message=f"Duplicate machine mode identifier {mode_id!r}."))
            seen_mode_ids.add(mode_key)
            normalized_modes.append(
                (
                    mode_key,
                    {
                        "id": mode_id,
                        "name": self._optional_string(raw_mode.get("name"), mode_path + ("name",), ""),
                        "speed_factor": self._optional_number(raw_mode.get("speed_factor"), mode_path + ("speed_factor",), 1.0, minimum=1e-9),
                        "power_multiplier": self._optional_number(raw_mode.get("power_multiplier"), mode_path + ("power_multiplier",), 1.0, minimum=0.0),
                        "tool_wear_rate": self._optional_number(raw_mode.get("tool_wear_rate"), mode_path + ("tool_wear_rate",), 1.0, minimum=0.0),
                    },
                )
            )
        normalized_modes.sort(key=lambda pair: pair[0])
        normalized_mode_list = [item for _, item in normalized_modes]
        default_mode_id = machine.get("default_mode_id", normalized_mode_list[0]["id"] if normalized_mode_list else 0)
        default_mode_id = self._normalize_identifier(default_mode_id, path + ("default_mode_id",))
        if normalized_mode_list and self._identifier_key(default_mode_id) not in {
            self._identifier_key(mode["id"]) for mode in normalized_mode_list
        }:
            self.issues.append(
                issue(
                    INVALID_REFERENCE,
                    *path,
                    "default_mode_id",
                    message=f"default_mode_id {default_mode_id!r} does not match any declared mode.",
                )
            )

        return {
            "id": self._normalize_identifier(machine.get("id"), path + ("id",)),
            "name": self._optional_string(machine.get("name"), path + ("name",), ""),
            "default_mode_id": default_mode_id,
            "power_processing": self._optional_number(machine.get("power_processing"), path + ("power_processing",), 10.0, minimum=0.0),
            "power_idle": self._optional_number(machine.get("power_idle"), path + ("power_idle",), 2.0, minimum=0.0),
            "power_setup": self._optional_number(machine.get("power_setup"), path + ("power_setup",), 5.0, minimum=0.0),
            "power_transport": self._optional_number(machine.get("power_transport"), path + ("power_transport",), 2.0, minimum=0.0),
            "startup_energy": self._optional_number(machine.get("startup_energy"), path + ("startup_energy",), 50.0, minimum=0.0),
            "setup_time": self._optional_number(machine.get("setup_time"), path + ("setup_time",), 0.0, minimum=0.0),
            "auxiliary_power_share": self._optional_number(machine.get("auxiliary_power_share"), path + ("auxiliary_power_share",), 0.0, minimum=0.0),
            "modes": normalized_mode_list,
        }

    def _normalize_worker(
        self,
        worker: Dict[str, Any],
        path: Tuple[object, ...],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "id": self._normalize_identifier(worker.get("id"), path + ("id",)),
            "name": self._optional_string(worker.get("name"), path + ("name",), ""),
            "labor_cost_per_hour": self._optional_number(worker.get("labor_cost_per_hour"), path + ("labor_cost_per_hour",), 20.0, minimum=0.0),
            "base_efficiency": self._optional_number(worker.get("base_efficiency"), path + ("base_efficiency",), 1.0, minimum=1e-9),
            "fatigue_rate": self._optional_number(worker.get("fatigue_rate"), path + ("fatigue_rate",), 0.03, minimum=0.0),
            "recovery_rate": self._optional_number(worker.get("recovery_rate"), path + ("recovery_rate",), 0.05, minimum=0.0),
            "fatigue_max": self._optional_number(worker.get("fatigue_max"), path + ("fatigue_max",), 1.0, minimum=0.0),
            "ocra_max_per_shift": self._optional_number(worker.get("ocra_max_per_shift"), path + ("ocra_max_per_shift",), 2.2, minimum=0.0),
            "ergonomic_tolerance": self._optional_number(worker.get("ergonomic_tolerance"), path + ("ergonomic_tolerance",), 1.0, minimum=0.0),
            "min_rest_fraction": self._optional_number(worker.get("min_rest_fraction"), path + ("min_rest_fraction",), 0.125, minimum=0.0, maximum=1.0),
            "max_consecutive_work_time": self._optional_number(
                worker.get("max_consecutive_work_time"),
                path + ("max_consecutive_work_time",),
                metadata.get("period_duration", DEFAULT_PERIOD_DURATION),
                minimum=0.0,
            ),
            "learning_coefficient": self._optional_number(worker.get("learning_coefficient"), path + ("learning_coefficient",), 0.1, minimum=0.0),
        }

    def _normalize_job(
        self,
        job: Dict[str, Any],
        path: Tuple[object, ...],
        defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_operations = job.get("operations")
        if not isinstance(raw_operations, list):
            self.issues.append(issue(INVALID_TYPE, *path, "operations", message="operations must be a non-empty list."))
            raw_operations = []
        elif not raw_operations:
            self.issues.append(issue(EMPTY_COLLECTION, *path, "operations", message="operations must not be empty."))

        seen_operation_ids: set[str] = set()
        normalized_operations: List[Dict[str, Any]] = []
        for index, raw_operation in enumerate(raw_operations):
            operation_path = path + ("operations", index)
            if not isinstance(raw_operation, dict):
                self.issues.append(issue(INVALID_TYPE, *operation_path, message="Each operation must be an object."))
                continue
            self._ensure_allowed_keys(raw_operation, SUPPORTED_OPERATION_FIELDS, operation_path)
            self._require_keys(raw_operation, REQUIRED_OPERATION_FIELDS, operation_path)
            operation_id = self._normalize_identifier(raw_operation.get("id"), operation_path + ("id",))
            operation_key = self._identifier_key(operation_id)
            if operation_key in seen_operation_ids:
                self.issues.append(issue(DUPLICATE_ID, *operation_path, "id", message=f"Duplicate operation identifier {operation_id!r} within a job."))
            seen_operation_ids.add(operation_key)
            normalized_operations.append(
                self._normalize_operation(raw_operation, operation_path, defaults)
            )

        return {
            "id": self._normalize_identifier(job.get("id"), path + ("id",)),
            "arrival_time": self._optional_number(job.get("arrival_time"), path + ("arrival_time",), 0.0, minimum=0.0),
            "due_date": self._optional_number(job.get("due_date"), path + ("due_date",), None),
            "weight": self._optional_number(job.get("weight"), path + ("weight",), 1.0, minimum=0.0),
            "operations": normalized_operations,
        }

    def _normalize_operation(
        self,
        operation: Dict[str, Any],
        path: Tuple[object, ...],
        defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_options = operation.get("processing_options")
        if not isinstance(raw_options, list):
            self.issues.append(issue(INVALID_TYPE, *path, "processing_options", message="processing_options must be a non-empty list."))
            raw_options = []
        elif not raw_options:
            self.issues.append(issue(EMPTY_COLLECTION, *path, "processing_options", message="processing_options must not be empty."))

        seen_options: set[Tuple[str, str]] = set()
        normalized_options: List[Tuple[Tuple[str, str], Dict[str, Any]]] = []
        for index, raw_option in enumerate(raw_options):
            option_path = path + ("processing_options", index)
            if not isinstance(raw_option, dict):
                self.issues.append(issue(INVALID_TYPE, *option_path, message="Each processing option must be an object."))
                continue
            self._ensure_allowed_keys(raw_option, SUPPORTED_PROCESSING_OPTION_FIELDS, option_path)
            self._require_keys(raw_option, REQUIRED_PROCESSING_OPTION_FIELDS, option_path)
            machine_id = self._normalize_identifier(raw_option.get("machine_id"), option_path + ("machine_id",))
            mode_id = self._normalize_identifier(raw_option.get("mode_id"), option_path + ("mode_id",))
            option_key = (self._identifier_key(machine_id), self._identifier_key(mode_id))
            if option_key in seen_options:
                self.issues.append(issue(DUPLICATE_ID, *option_path, message=f"Duplicate processing option for machine {machine_id!r} and mode {mode_id!r}."))
            seen_options.add(option_key)
            normalized_options.append(
                (
                    option_key,
                    {
                        "machine_id": machine_id,
                        "mode_id": mode_id,
                        "duration": self._required_number(raw_option.get("duration"), option_path + ("duration",), minimum=1e-9),
                    },
                )
            )
        normalized_options.sort(key=lambda pair: pair[0])

        eligible_workers = self._normalize_identifier_list(
            operation.get("eligible_workers"),
            path + ("eligible_workers",),
            require_non_empty=True,
        )
        eligible_workers.sort(key=self._identifier_key)

        return {
            "id": self._normalize_identifier(operation.get("id"), path + ("id",)),
            "processing_options": [item for _, item in normalized_options],
            "eligible_workers": eligible_workers,
            "ergonomic_risk_rate": self._optional_number(
                operation.get("ergonomic_risk_rate"),
                path + ("ergonomic_risk_rate",),
                defaults["default_ergonomic_risk"],
                minimum=0.0,
            ),
            "transport_time": self._optional_number(operation.get("transport_time"), path + ("transport_time",), 0.0, minimum=0.0),
            "waiting_time": self._optional_number(operation.get("waiting_time"), path + ("waiting_time",), 0.0, minimum=0.0),
        }

    def _build_identifier_maps(self, normalized: Dict[str, Any]) -> IdentifierMaps:
        jobs: Dict[TypedIdKey, int] = {}
        reverse_jobs: Dict[int, ExternalId] = {}
        machines: Dict[TypedIdKey, int] = {}
        reverse_machines: Dict[int, ExternalId] = {}
        workers: Dict[TypedIdKey, int] = {}
        reverse_workers: Dict[int, ExternalId] = {}
        operations: Dict[Tuple[TypedIdKey, TypedIdKey], Tuple[int, int]] = {}
        reverse_operations: Dict[Tuple[int, int], ExternalId] = {}
        machine_modes: Dict[Tuple[TypedIdKey, TypedIdKey], int] = {}
        reverse_machine_modes: Dict[Tuple[int, int], ExternalId] = {}

        for machine_index, machine in enumerate(normalized["machines"]):
            machine_key = self._identifier_key(machine["id"])
            machines[machine_key] = machine_index
            reverse_machines[machine_index] = machine["id"]
            for mode_index, mode in enumerate(machine["modes"]):
                machine_modes[(machine_key, self._identifier_key(mode["id"]))] = mode_index
                reverse_machine_modes[(machine_index, mode_index)] = mode["id"]

        for worker_index, worker in enumerate(normalized["workers"]):
            worker_key = self._identifier_key(worker["id"])
            workers[worker_key] = worker_index
            reverse_workers[worker_index] = worker["id"]

        for job_index, job in enumerate(normalized["jobs"]):
            job_key = self._identifier_key(job["id"])
            jobs[job_key] = job_index
            reverse_jobs[job_index] = job["id"]
            for op_index, operation in enumerate(job["operations"]):
                op_key = self._identifier_key(operation["id"])
                operations[(job_key, op_key)] = (job_index, op_index)
                reverse_operations[(job_index, op_index)] = operation["id"]

        return IdentifierMaps(
            jobs=jobs,
            reverse_jobs=reverse_jobs,
            machines=machines,
            reverse_machines=reverse_machines,
            workers=workers,
            reverse_workers=reverse_workers,
            operations=operations,
            reverse_operations=reverse_operations,
            machine_modes=machine_modes,
            reverse_machine_modes=reverse_machine_modes,
        )

    def _build_instance(self, normalized: Dict[str, Any], id_maps: IdentifierMaps) -> SFJSSPInstance:
        metadata = normalized["metadata"]
        defaults = normalized["defaults"]
        period_clock = PeriodClock(
            period_duration=metadata["period_duration"],
            horizon_start=metadata["horizon_start"],
        )
        instance = SFJSSPInstance(
            instance_id=str(metadata["instance_id"]),
            instance_name=metadata["instance_name"],
            label=instance_label_from_public_calibration_status(
                metadata["calibration_status"]
            ),
            label_justification=metadata["calibration_status_justification"],
            instance_type=InstanceType(metadata["instance_type"]),
            planning_horizon=metadata["planning_horizon"],
            source=metadata["source"],
            calibration_sources=list(metadata["calibration_sources"]),
            known_limitations=list(metadata["known_limitations"]),
            carbon_emission_factor=defaults["carbon_emission_factor"],
            default_electricity_price=defaults["default_electricity_price"],
            auxiliary_power_total=defaults["auxiliary_power_total"],
            default_ergonomic_risk=defaults["default_ergonomic_risk"],
            period_clock=period_clock,
        )
        instance.electricity_prices = {
            entry["period"]: entry["price"] for entry in defaults["electricity_prices"]
        }
        instance.build_calibration_record()

        for machine_index, machine_payload in enumerate(normalized["machines"]):
            machine_key = self._identifier_key(machine_payload["id"])
            modes = []
            for mode in machine_payload["modes"]:
                internal_mode_id = id_maps.machine_modes[(machine_key, self._identifier_key(mode["id"]))]
                modes.append(
                    MachineMode(
                        mode_id=internal_mode_id,
                        mode_name=mode["name"] or str(mode["id"]),
                        speed_factor=mode["speed_factor"],
                        power_multiplier=mode["power_multiplier"],
                        tool_wear_rate=mode["tool_wear_rate"],
                    )
                )
            instance.add_machine(
                Machine(
                    machine_id=machine_index,
                    machine_name=machine_payload["name"] or str(machine_payload["id"]),
                    modes=modes,
                    default_mode_id=id_maps.machine_modes[
                        (machine_key, self._identifier_key(machine_payload["default_mode_id"]))
                    ],
                    power_processing=machine_payload["power_processing"],
                    power_idle=machine_payload["power_idle"],
                    power_setup=machine_payload["power_setup"],
                    power_transport=machine_payload["power_transport"],
                    startup_energy=machine_payload["startup_energy"],
                    setup_time=machine_payload["setup_time"],
                    auxiliary_power_share=machine_payload["auxiliary_power_share"],
                )
            )

        for worker_index, worker_payload in enumerate(normalized["workers"]):
            instance.add_worker(
                Worker(
                    worker_id=worker_index,
                    worker_name=worker_payload["name"] or str(worker_payload["id"]),
                    labor_cost_per_hour=worker_payload["labor_cost_per_hour"],
                    base_efficiency=worker_payload["base_efficiency"],
                    fatigue_rate=worker_payload["fatigue_rate"],
                    recovery_rate=worker_payload["recovery_rate"],
                    fatigue_max=worker_payload["fatigue_max"],
                    ocra_max_per_shift=worker_payload["ocra_max_per_shift"],
                    ergonomic_tolerance=worker_payload["ergonomic_tolerance"],
                    min_rest_fraction=worker_payload["min_rest_fraction"],
                    max_consecutive_work_time=worker_payload["max_consecutive_work_time"],
                    learning_coefficient=worker_payload["learning_coefficient"],
                    SHIFT_DURATION=metadata["period_duration"],
                )
            )

        for job_index, job_payload in enumerate(normalized["jobs"]):
            operations: List[Operation] = []
            for op_index, operation_payload in enumerate(job_payload["operations"]):
                processing_times: Dict[int, Dict[int, float]] = {}
                eligible_machines = set()
                for option in operation_payload["processing_options"]:
                    machine_key = self._identifier_key(option["machine_id"])
                    mode_lookup = (machine_key, self._identifier_key(option["mode_id"]))
                    if machine_key not in id_maps.machines:
                        self.issues.append(
                            issue(
                                INVALID_REFERENCE,
                                "jobs",
                                job_index,
                                "operations",
                                op_index,
                                "processing_options",
                                message=f"Unknown machine reference {option['machine_id']!r}.",
                            )
                        )
                        continue
                    if mode_lookup not in id_maps.machine_modes:
                        self.issues.append(
                            issue(
                                INVALID_REFERENCE,
                                "jobs",
                                job_index,
                                "operations",
                                op_index,
                                "processing_options",
                                message=(
                                    f"Unknown mode reference {option['mode_id']!r} for machine {option['machine_id']!r}."
                                ),
                            )
                        )
                        continue
                    machine_id = id_maps.machines[machine_key]
                    mode_id = id_maps.machine_modes[mode_lookup]
                    processing_times.setdefault(machine_id, {})[mode_id] = option["duration"]
                    eligible_machines.add(machine_id)

                eligible_workers = set()
                for external_worker_id in operation_payload["eligible_workers"]:
                    worker_key = self._identifier_key(external_worker_id)
                    if worker_key not in id_maps.workers:
                        self.issues.append(
                            issue(
                                INVALID_REFERENCE,
                                "jobs",
                                job_index,
                                "operations",
                                op_index,
                                "eligible_workers",
                                message=f"Unknown worker reference {external_worker_id!r}.",
                            )
                        )
                        continue
                    eligible_workers.add(id_maps.workers[worker_key])

                operation = Operation(
                    job_id=job_index,
                    op_id=op_index,
                    processing_times=processing_times,
                    transport_time=operation_payload["transport_time"],
                    waiting_time=operation_payload["waiting_time"],
                    eligible_machines=eligible_machines,
                    eligible_workers=eligible_workers,
                )
                operations.append(operation)
                instance.ergonomic_risk_map[(job_index, op_index)] = operation_payload["ergonomic_risk_rate"]

            instance.add_job(
                Job(
                    job_id=job_index,
                    operations=operations,
                    arrival_time=job_payload["arrival_time"],
                    due_date=job_payload["due_date"],
                    weight=job_payload["weight"],
                )
            )

        for job in instance.jobs:
            for operation in job.operations:
                for worker_id in operation.eligible_workers:
                    instance.workers[worker_id].eligible_operations.add((job.job_id, operation.op_id))

        self._apply_calendar_payload(instance, normalized.get("calendar"), id_maps)
        self._apply_event_payload(instance, normalized.get("events"), id_maps)
        return instance

    def _apply_calendar_payload(
        self,
        instance: SFJSSPInstance,
        calendar_payload: Optional[Dict[str, Any]],
        id_maps: IdentifierMaps,
    ) -> None:
        if not calendar_payload:
            return

        for index, window in enumerate(calendar_payload.get("worker_shifts", [])):
            worker_key = self._identifier_key(window["worker_id"])
            if worker_key not in id_maps.workers:
                self.issues.append(
                    issue(
                        INVALID_REFERENCE,
                        "calendar",
                        "worker_shifts",
                        index,
                        "worker_id",
                        message=f"Unknown worker reference {window['worker_id']!r}.",
                    )
                )
                continue
            instance.add_worker_shift_window(
                id_maps.workers[worker_key],
                window["start_time"],
                window["end_time"],
                shift_label=window["shift_label"],
                details=window["details"],
            )

        for index, window in enumerate(calendar_payload.get("machine_unavailability", [])):
            machine_key = self._identifier_key(window["machine_id"])
            if machine_key not in id_maps.machines:
                self.issues.append(
                    issue(
                        INVALID_REFERENCE,
                        "calendar",
                        "machine_unavailability",
                        index,
                        "machine_id",
                        message=f"Unknown machine reference {window['machine_id']!r}.",
                    )
                )
                continue
            instance.add_machine_unavailability(
                id_maps.machines[machine_key],
                window["start_time"],
                window["end_time"],
                reason=window["reason"],
                source=window["source"],
                details=window["details"],
            )

        for index, window in enumerate(calendar_payload.get("worker_unavailability", [])):
            worker_key = self._identifier_key(window["worker_id"])
            if worker_key not in id_maps.workers:
                self.issues.append(
                    issue(
                        INVALID_REFERENCE,
                        "calendar",
                        "worker_unavailability",
                        index,
                        "worker_id",
                        message=f"Unknown worker reference {window['worker_id']!r}.",
                    )
                )
                continue
            instance.add_worker_unavailability(
                id_maps.workers[worker_key],
                window["start_time"],
                window["end_time"],
                reason=window["reason"],
                source=window["source"],
                details=window["details"],
            )

    def _apply_event_payload(
        self,
        instance: SFJSSPInstance,
        events_payload: Optional[Dict[str, Any]],
        id_maps: IdentifierMaps,
    ) -> None:
        if not events_payload:
            return

        for index, event in enumerate(events_payload.get("machine_breakdowns", [])):
            machine_key = self._identifier_key(event["machine_id"])
            if machine_key not in id_maps.machines:
                self.issues.append(
                    issue(
                        INVALID_REFERENCE,
                        "events",
                        "machine_breakdowns",
                        index,
                        "machine_id",
                        message=f"Unknown machine reference {event['machine_id']!r}.",
                    )
                )
                continue
            instance.add_machine_breakdown_event(
                id_maps.machines[machine_key],
                event["start_time"],
                event["repair_duration"],
                source=event["source"],
                event_id=event["event_id"] or None,
                details=event["details"],
            )

        for index, event in enumerate(events_payload.get("worker_absences", [])):
            worker_key = self._identifier_key(event["worker_id"])
            if worker_key not in id_maps.workers:
                self.issues.append(
                    issue(
                        INVALID_REFERENCE,
                        "events",
                        "worker_absences",
                        index,
                        "worker_id",
                        message=f"Unknown worker reference {event['worker_id']!r}.",
                    )
                )
                continue
            instance.add_worker_absence_event(
                id_maps.workers[worker_key],
                event["start_time"],
                event["end_time"],
                source=event["source"],
                event_id=event["event_id"] or None,
                details=event["details"],
            )

    def _ensure_allowed_keys(
        self,
        payload: Dict[str, Any],
        allowed_fields: Iterable[str],
        path: Tuple[object, ...],
    ) -> None:
        if not self.strict:
            return
        allowed = set(allowed_fields)
        reserved_top_level = (
            get_reserved_unsupported_top_level_fields(self.schema_name) if not path else set()
        )
        for key in payload.keys():
            if key not in allowed and key not in reserved_top_level:
                self.issues.append(
                    issue(
                        UNKNOWN_FIELD,
                        *path,
                        key,
                        message=f"Field {key!r} is not supported by {self.schema_name}.",
                    )
                )

    def _require_keys(
        self,
        payload: Dict[str, Any],
        required_fields: Iterable[str],
        path: Tuple[object, ...],
    ) -> None:
        for field_name in required_fields:
            if field_name not in payload:
                self.issues.append(
                    issue(
                        MISSING_REQUIRED,
                        *path,
                        field_name,
                        message=f"Field {field_name!r} is required.",
                    )
                )

    def _normalize_identifier(self, value: Any, path: Tuple[object, ...]) -> ExternalId:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            self.issues.append(issue(INVALID_TYPE, *path, message="Identifiers must be strings or integers."))
            return "<invalid>"
        if isinstance(value, str) and not value.strip():
            self.issues.append(issue(INVALID_VALUE, *path, message="Identifiers must not be empty strings."))
        return value

    @staticmethod
    def _identifier_key(value: ExternalId) -> TypedIdKey:
        if isinstance(value, int):
            return f"int:{value}"
        return f"str:{value}"

    def _normalize_identifier_list(
        self,
        value: Any,
        path: Tuple[object, ...],
        require_non_empty: bool = False,
    ) -> List[ExternalId]:
        if not isinstance(value, list):
            self.issues.append(issue(INVALID_TYPE, *path, message="Expected a list of identifiers."))
            return []
        if require_non_empty and not value:
            self.issues.append(issue(EMPTY_COLLECTION, *path, message="List must not be empty."))
        items: List[ExternalId] = []
        seen: set[str] = set()
        for index, raw_value in enumerate(value):
            normalized = self._normalize_identifier(raw_value, path + (index,))
            identifier_key = self._identifier_key(normalized)
            if identifier_key in seen:
                self.issues.append(issue(DUPLICATE_ID, *path, index, message=f"Duplicate identifier {normalized!r}."))
            seen.add(identifier_key)
            items.append(normalized)
        return items

    def _optional_string(self, value: Any, path: Tuple[object, ...], default: str) -> str:
        if value is None:
            return default
        if not isinstance(value, str):
            self.issues.append(issue(INVALID_TYPE, *path, message="Expected a string."))
            return default
        return value

    def _optional_mapping(self, value: Any, path: Tuple[object, ...]) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            self.issues.append(issue(INVALID_TYPE, *path, message="Expected an object."))
            return {}
        return dict(value)

    def _optional_string_list(self, value: Any, path: Tuple[object, ...]) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            self.issues.append(issue(INVALID_TYPE, *path, message="Expected a list of strings."))
            return []
        items: List[str] = []
        for index, raw_value in enumerate(value):
            if not isinstance(raw_value, str):
                self.issues.append(issue(INVALID_TYPE, *path, index, message="Expected a string."))
                continue
            items.append(raw_value)
        return items

    def _optional_enum(
        self,
        value: Any,
        allowed: Iterable[str],
        path: Tuple[object, ...],
        default: str,
    ) -> str:
        allowed_values = set(allowed)
        if value is None:
            return default
        if not isinstance(value, str):
            self.issues.append(issue(INVALID_TYPE, *path, message="Expected a string enumeration value."))
            return default
        if value not in allowed_values:
            self.issues.append(
                issue(
                    INVALID_VALUE,
                    *path,
                    message=f"Expected one of {sorted(allowed_values)!r}, got {value!r}.",
                )
            )
            return default
        return value

    def _required_int(
        self,
        value: Any,
        path: Tuple[object, ...],
        minimum: Optional[int] = None,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self.issues.append(issue(INVALID_TYPE, *path, message="Expected an integer."))
            return 0
        if minimum is not None and value < minimum:
            self.issues.append(issue(INVALID_VALUE, *path, message=f"Expected an integer >= {minimum}, got {value}."))
        return value

    def _required_number(
        self,
        value: Any,
        path: Tuple[object, ...],
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.issues.append(issue(INVALID_TYPE, *path, message="Expected a number."))
            return 0.0
        number = float(value)
        if minimum is not None and number < minimum:
            self.issues.append(issue(INVALID_VALUE, *path, message=f"Expected a value >= {minimum}, got {number}."))
        if maximum is not None and number > maximum:
            self.issues.append(issue(INVALID_VALUE, *path, message=f"Expected a value <= {maximum}, got {number}."))
        return number

    def _optional_number(
        self,
        value: Any,
        path: Tuple[object, ...],
        default: Optional[float],
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> Optional[float]:
        if value is None:
            return default
        return self._required_number(value, path, minimum=minimum, maximum=maximum)

    def _raise_if_needed(self) -> None:
        if self.issues:
            raise InterfaceValidationError(self.issues)
