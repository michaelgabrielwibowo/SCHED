"""
Explicit site-parameter overlays for external interface runs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .errors import INVALID_REFERENCE, InterfaceValidationError, ValidationIssue, issue


@dataclass(frozen=True)
class SiteProfileDefinition:
    profile_id: str
    description: str
    calibration_status: str
    overlay: Dict[str, Any]


SITE_PROFILES: Dict[str, SiteProfileDefinition] = {
    "light_assembly_demo_v1": SiteProfileDefinition(
        profile_id="light_assembly_demo_v1",
        description=(
            "Illustrative light-assembly overlay for labor pacing and energy pricing. "
            "This is not factory calibrated."
        ),
        calibration_status="illustrative_not_calibrated",
        overlay={
            "defaults": {
                "default_electricity_price": 0.17,
                "carbon_emission_factor": 0.31,
                "auxiliary_power_total": 8.0,
            },
            "machines": [
                {"id": "*", "setup_time": 4.0},
            ],
            "workers": [
                {
                    "id": "*",
                    "min_rest_fraction": 0.10,
                    "ocra_max_per_shift": 0.85,
                    "ergonomic_tolerance": 0.90,
                }
            ],
        },
    ),
    "heavy_fabrication_demo_v1": SiteProfileDefinition(
        profile_id="heavy_fabrication_demo_v1",
        description=(
            "Illustrative heavy-fabrication overlay for higher auxiliary load and longer "
            "resource setup windows. This is not factory calibrated."
        ),
        calibration_status="illustrative_not_calibrated",
        overlay={
            "defaults": {
                "default_electricity_price": 0.12,
                "carbon_emission_factor": 0.52,
                "auxiliary_power_total": 30.0,
            },
            "machines": [
                {"id": "*", "setup_time": 8.0, "power_setup": 3.5},
            ],
            "workers": [
                {
                    "id": "*",
                    "min_rest_fraction": 0.125,
                    "ocra_max_per_shift": 1.20,
                    "fatigue_rate": 0.018,
                }
            ],
        },
    ),
}

SUPPORTED_SITE_PROFILES = frozenset(SITE_PROFILES)


def apply_site_profile(
    payload: Dict[str, Any],
    *,
    profile_name: str,
) -> Dict[str, Any]:
    """Apply one named site overlay to a public external-schema payload."""

    if profile_name not in SITE_PROFILES:
        raise InterfaceValidationError.from_single_issue(
            code="invalid_value",
            path="$.site_profile",
            message=(
                f"Unknown site profile {profile_name!r}. Expected one of "
                f"{sorted(SUPPORTED_SITE_PROFILES)!r}."
            ),
        )

    profile = SITE_PROFILES[profile_name]
    normalized = deepcopy(payload)
    issues: List[ValidationIssue] = []
    overridden_fields: Dict[str, List[str]] = {}

    if "defaults" in profile.overlay:
        defaults_payload = normalized.setdefault("defaults", {})
        defaults_payload.update(dict(profile.overlay["defaults"]))
        overridden_fields["defaults"] = sorted(profile.overlay["defaults"])

    for section_name in ("machines", "workers", "jobs"):
        resource_overlays = list(profile.overlay.get(section_name, []))
        if not resource_overlays:
            continue
        overridden_fields[section_name] = _apply_resource_overlays(
            normalized,
            section_name=section_name,
            overlays=resource_overlays,
            issues=issues,
        )

    if issues:
        raise InterfaceValidationError(issues)

    return {
        "payload": normalized,
        "provenance": {
            "profile_id": profile.profile_id,
            "description": profile.description,
            "calibration_status": profile.calibration_status,
            "overlay_sections": sorted(overridden_fields),
            "overridden_fields": {
                section_name: sorted(fields)
                for section_name, fields in overridden_fields.items()
            },
        },
    }


def _apply_resource_overlays(
    payload: Dict[str, Any],
    *,
    section_name: str,
    overlays: Iterable[Dict[str, Any]],
    issues: List[ValidationIssue],
) -> List[str]:
    resources = payload.setdefault(section_name, [])
    if not isinstance(resources, list):
        issues.append(
            issue(
                "invalid_type",
                section_name,
                message=f"Section {section_name!r} must be a list before applying a site profile.",
            )
        )
        return []

    indexed = {resource.get("id"): resource for resource in resources if isinstance(resource, dict)}
    overridden_fields: set[str] = set()

    for index, overlay in enumerate(overlays):
        target_id = overlay.get("id")
        overlay_fields = {key: value for key, value in overlay.items() if key != "id"}
        overridden_fields.update(overlay_fields)
        if target_id == "*":
            for resource in resources:
                if isinstance(resource, dict):
                    resource.update(overlay_fields)
            continue
        if target_id not in indexed:
            issues.append(
                issue(
                    INVALID_REFERENCE,
                    "site_profile",
                    section_name,
                    index,
                    "id",
                    message=(
                        f"Site profile overlay targets unknown {section_name[:-1]} id "
                        f"{target_id!r}."
                    ),
                )
            )
            continue
        indexed[target_id].update(overlay_fields)

    return sorted(overridden_fields)
