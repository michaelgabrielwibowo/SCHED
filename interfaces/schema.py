"""
Versioned schema contract for external SFJSSP inputs.

`sfjssp_external_v1` is intentionally smaller than the internal dataclass
surface. It accepts only the fields required to build a canonical
`SFJSSPInstance` without inventing a second scheduling semantics layer.

Units:
- all durations are in minutes
- all power values are in kW
- all energy values are in kWh
- labor cost is in currency per hour
- ergonomic risk rates are in OCRA-index per minute

Reserved top-level sections `transport`, `calendar`, and `events` are rejected
in v1 because the current external workflow does not yet expose them as a
validated public contract.
"""

from __future__ import annotations


EXTERNAL_INPUT_SCHEMA = "sfjssp_external_v1"

SUPPORTED_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "metadata",
        "defaults",
        "machines",
        "workers",
        "jobs",
    }
)
RESERVED_UNSUPPORTED_TOP_LEVEL_FIELDS = frozenset({"transport", "calendar", "events"})
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

DEFAULT_PERIOD_DURATION = 480.0
DEFAULT_HORIZON_START = 0.0
DEFAULT_INSTANCE_TYPE = "static"
