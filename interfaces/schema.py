"""
Versioned schema contract for external SFJSSP inputs.

`sfjssp_external_v1` is intentionally smaller than the internal dataclass
surface. It accepts only the fields required to build a canonical
`SFJSSPInstance` without inventing a second scheduling semantics layer.

`sfjssp_external_v2` extends the JSON and CSV contracts with validated
`calendar` and `events` sections that map directly onto canonical worker shift
windows, explicit resource-unavailability windows, and typed breakdown/absence
events.

Calibration truthfulness:
- `metadata.calibration_status` is the public calibration field
- supported values are `fully_synthetic`, `calibrated_synthetic`, and
  `site_calibrated`
- `metadata.label` and `metadata.label_justification` remain accepted only as
  compatibility aliases for older payloads
- non-synthetic claims must include both a justification and at least one
  calibration source reference

Units:
- all durations are in minutes
- all power values are in kW
- all energy values are in kWh
- labor cost is in currency per hour
- ergonomic risk rates are in OCRA-index per minute
"""

from __future__ import annotations


EXTERNAL_INPUT_SCHEMA_V1 = "sfjssp_external_v1"
EXTERNAL_INPUT_SCHEMA_V2 = "sfjssp_external_v2"

# Backward-compatible alias used by the current CSV bundle path.
EXTERNAL_INPUT_SCHEMA = EXTERNAL_INPUT_SCHEMA_V1
LATEST_EXTERNAL_INPUT_SCHEMA = EXTERNAL_INPUT_SCHEMA_V2
SUPPORTED_EXTERNAL_INPUT_SCHEMAS = frozenset(
    {EXTERNAL_INPUT_SCHEMA_V1, EXTERNAL_INPUT_SCHEMA_V2}
)

SUPPORTED_TOP_LEVEL_FIELDS_V1 = frozenset(
    {
        "schema",
        "metadata",
        "defaults",
        "machines",
        "workers",
        "jobs",
    }
)
SUPPORTED_TOP_LEVEL_FIELDS_V2 = frozenset(
    set(SUPPORTED_TOP_LEVEL_FIELDS_V1) | {"calendar", "events"}
)
SUPPORTED_TOP_LEVEL_FIELDS = SUPPORTED_TOP_LEVEL_FIELDS_V1

RESERVED_UNSUPPORTED_TOP_LEVEL_FIELDS_V1 = frozenset({"transport", "calendar", "events"})
RESERVED_UNSUPPORTED_TOP_LEVEL_FIELDS_V2 = frozenset({"transport"})
RESERVED_UNSUPPORTED_TOP_LEVEL_FIELDS = RESERVED_UNSUPPORTED_TOP_LEVEL_FIELDS_V1

REQUIRED_TOP_LEVEL_FIELDS = frozenset({"schema", "metadata", "machines", "workers", "jobs"})

SUPPORTED_METADATA_FIELDS = frozenset(
    {
        "instance_id",
        "instance_name",
        "instance_type",
        "planning_horizon",
        "period_duration",
        "horizon_start",
        "source",
        "calibration_status",
        "calibration_status_justification",
        "label",
        "label_justification",
        "calibration_sources",
        "known_limitations",
    }
)
REQUIRED_METADATA_FIELDS = frozenset({"instance_id"})

SUPPORTED_DEFAULT_FIELDS = frozenset(
    {
        "default_ergonomic_risk",
        "default_electricity_price",
        "electricity_prices",
        "carbon_emission_factor",
        "auxiliary_power_total",
    }
)

SUPPORTED_ELECTRICITY_PRICE_FIELDS = frozenset({"period", "price"})
REQUIRED_ELECTRICITY_PRICE_FIELDS = frozenset({"period", "price"})

SUPPORTED_MACHINE_FIELDS = frozenset(
    {
        "id",
        "name",
        "default_mode_id",
        "power_processing",
        "power_idle",
        "power_setup",
        "power_transport",
        "startup_energy",
        "setup_time",
        "auxiliary_power_share",
        "modes",
    }
)
REQUIRED_MACHINE_FIELDS = frozenset({"id", "modes"})

SUPPORTED_MACHINE_MODE_FIELDS = frozenset(
    {"id", "name", "speed_factor", "power_multiplier", "tool_wear_rate"}
)
REQUIRED_MACHINE_MODE_FIELDS = frozenset({"id"})

SUPPORTED_WORKER_FIELDS = frozenset(
    {
        "id",
        "name",
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
    }
)
REQUIRED_WORKER_FIELDS = frozenset({"id"})

SUPPORTED_JOB_FIELDS = frozenset({"id", "arrival_time", "due_date", "weight", "operations"})
REQUIRED_JOB_FIELDS = frozenset({"id", "operations"})

SUPPORTED_OPERATION_FIELDS = frozenset(
    {
        "id",
        "processing_options",
        "eligible_workers",
        "ergonomic_risk_rate",
        "transport_time",
        "waiting_time",
    }
)
REQUIRED_OPERATION_FIELDS = frozenset({"id", "processing_options", "eligible_workers"})

SUPPORTED_PROCESSING_OPTION_FIELDS = frozenset({"machine_id", "mode_id", "duration"})
REQUIRED_PROCESSING_OPTION_FIELDS = frozenset({"machine_id", "mode_id", "duration"})

SUPPORTED_CALENDAR_FIELDS = frozenset(
    {"machine_unavailability", "worker_unavailability", "worker_shifts"}
)

SUPPORTED_MACHINE_UNAVAILABILITY_FIELDS = frozenset(
    {"machine_id", "start_time", "end_time", "reason", "source", "details"}
)
REQUIRED_MACHINE_UNAVAILABILITY_FIELDS = frozenset(
    {"machine_id", "start_time", "end_time"}
)

SUPPORTED_WORKER_UNAVAILABILITY_FIELDS = frozenset(
    {"worker_id", "start_time", "end_time", "reason", "source", "details"}
)
REQUIRED_WORKER_UNAVAILABILITY_FIELDS = frozenset(
    {"worker_id", "start_time", "end_time"}
)

SUPPORTED_WORKER_SHIFT_FIELDS = frozenset(
    {"worker_id", "start_time", "end_time", "shift_label", "details"}
)
REQUIRED_WORKER_SHIFT_FIELDS = frozenset({"worker_id", "start_time", "end_time"})

SUPPORTED_EVENTS_FIELDS = frozenset({"machine_breakdowns", "worker_absences"})

SUPPORTED_MACHINE_BREAKDOWN_FIELDS = frozenset(
    {"machine_id", "start_time", "repair_duration", "source", "details", "event_id"}
)
REQUIRED_MACHINE_BREAKDOWN_FIELDS = frozenset(
    {"machine_id", "start_time", "repair_duration"}
)

SUPPORTED_WORKER_ABSENCE_FIELDS = frozenset(
    {"worker_id", "start_time", "end_time", "source", "details", "event_id"}
)
REQUIRED_WORKER_ABSENCE_FIELDS = frozenset({"worker_id", "start_time", "end_time"})

DEFAULT_PERIOD_DURATION = 480.0
DEFAULT_HORIZON_START = 0.0
DEFAULT_INSTANCE_TYPE = "static"


def normalize_external_schema_name(schema_name: object) -> str | None:
    """Return the normalized supported schema name, or `None` if unsupported."""

    if schema_name in SUPPORTED_EXTERNAL_INPUT_SCHEMAS:
        return str(schema_name)
    return None


def get_supported_top_level_fields(schema_name: str) -> frozenset[str]:
    """Return the allowed top-level fields for one schema version."""

    if schema_name == EXTERNAL_INPUT_SCHEMA_V2:
        return SUPPORTED_TOP_LEVEL_FIELDS_V2
    return SUPPORTED_TOP_LEVEL_FIELDS_V1


def get_reserved_unsupported_top_level_fields(schema_name: str) -> frozenset[str]:
    """Return top-level sections that are reserved but unsupported."""

    if schema_name == EXTERNAL_INPUT_SCHEMA_V2:
        return RESERVED_UNSUPPORTED_TOP_LEVEL_FIELDS_V2
    return RESERVED_UNSUPPORTED_TOP_LEVEL_FIELDS_V1


def schema_supports_calendar_events(schema_name: str) -> bool:
    """True when the declared schema supports the public calendar/events contract."""

    return schema_name == EXTERNAL_INPUT_SCHEMA_V2
