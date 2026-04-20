import json
from pathlib import Path

import pytest

try:
    from ..interfaces import EXTERNAL_INPUT_SCHEMA_V2, InterfaceValidationError, load_instance_from_json
except ImportError:  # pragma: no cover - supports repo-root imports
    from interfaces import EXTERNAL_INPUT_SCHEMA_V2, InterfaceValidationError, load_instance_from_json


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "interfaces_adapters"


def _load_fixture(name: str) -> dict:
    with (FIXTURE_ROOT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert_issue(exc: InterfaceValidationError, code: str, path_fragment: str) -> None:
    assert any(
        issue.code == code and path_fragment in issue.path for issue in exc.issues
    ), exc.issues


def test_plant_tables_adapter_builds_v2_instance_and_declares_dropped_fields():
    imported = load_instance_from_json(
        FIXTURE_ROOT / "valid_plant_tables_v1.json",
        adapter_name="plant_tables_v1",
    )

    assert imported.schema == EXTERNAL_INPUT_SCHEMA_V2
    assert imported.instance.instance_id == "PLANT_EXT_V2"
    assert imported.instance.instance_name == "Plant Adapter Fixture"
    assert imported.instance.instance_type.value == "dynamic"
    assert imported.instance.default_electricity_price == pytest.approx(0.15)
    assert imported.instance.electricity_prices == {0: 0.15, 60: 0.22}
    assert imported.instance.get_machine_unavailability(0)[0].reason == "maintenance"
    assert imported.instance.workers[0].shift_windows[0].shift_label == "day"
    assert imported.instance.machine_breakdown_events[0].event_id == "machine-breakdown-plant-1"
    assert imported.instance.worker_absence_events[0].event_id == "worker-absence-plant-1"

    provenance = imported.provenance
    assert provenance["raw_source"]["input_format"] == "json"
    assert provenance["external_schema"] == EXTERNAL_INPUT_SCHEMA_V2
    assert provenance["calibration"]["status"] == "calibrated_synthetic"
    assert set(provenance["calibration"]["sources"]) == {
        "time-study:wc-01-2026q1",
        "energy-meter:wc-01-2026q1",
    }
    assert provenance["adapter"]["adapter_name"] == "plant_tables_v1"
    assert provenance["adapter"]["mapping_summary"]["workcenters"] == "machines"
    assert set(provenance["adapter"]["dropped_fields"]) >= {
        "header.erp_batch_id",
        "workcenters.department_code",
        "operators.team_name",
        "orders.customer_code",
        "routing_steps.inspection_code",
    }
    assert provenance["site_profile"] is None
    assert provenance["parameter_sources"]["payload_defaults_present"] is True
    assert provenance["parameter_sources"]["site_profile_applied"] is False


def test_plant_tables_adapter_rejects_unsupported_top_level_sections_in_strict_mode(tmp_path):
    payload = _load_fixture("valid_plant_tables_v1.json")
    payload["inventory_buffers"] = [{"buffer_code": "B1"}]
    source_path = tmp_path / "invalid_plant_tables.json"
    source_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InterfaceValidationError) as excinfo:
        load_instance_from_json(
            source_path,
            adapter_name="plant_tables_v1",
        )

    _assert_issue(excinfo.value, "unsupported_section", "$.inventory_buffers")


def test_site_profile_overlay_is_explicit_and_preserved_in_provenance():
    imported = load_instance_from_json(
        FIXTURE_ROOT / "valid_plant_tables_v1.json",
        adapter_name="plant_tables_v1",
        site_profile_name="light_assembly_demo_v1",
    )

    assert imported.instance.default_electricity_price == pytest.approx(0.17)
    assert imported.instance.auxiliary_power_total == pytest.approx(8.0)
    assert imported.instance.workers[0].min_rest_fraction == pytest.approx(0.10)
    assert imported.instance.workers[0].ocra_max_per_shift == pytest.approx(0.85)
    assert imported.instance.machines[0].setup_time == pytest.approx(4.0)

    site_profile = imported.provenance["site_profile"]
    assert site_profile["profile_id"] == "light_assembly_demo_v1"
    assert site_profile["calibration_status"] == "illustrative_not_calibrated"
    assert set(site_profile["overlay_sections"]) == {"defaults", "machines", "workers"}
    assert imported.provenance["parameter_sources"]["site_profile_applied"] is True
