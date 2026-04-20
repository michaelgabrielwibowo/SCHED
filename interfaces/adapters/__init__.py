"""
Adapter registry for plant-oriented source layouts.
"""

from __future__ import annotations

from typing import Any, Dict

from ..errors import InterfaceValidationError
from .plant_tables import PLANT_TABLES_ADAPTER_NAME, adapt_plant_tables_payload


SUPPORTED_SOURCE_ADAPTERS = frozenset({PLANT_TABLES_ADAPTER_NAME})


def adapt_source_payload(
    payload: Any,
    *,
    adapter_name: str,
    strict: bool = True,
) -> Dict[str, Any]:
    """Normalize one supported raw source payload into the public external schema."""

    normalized_name = str(adapter_name).strip()
    if normalized_name == PLANT_TABLES_ADAPTER_NAME:
        return adapt_plant_tables_payload(payload, strict=strict)
    raise InterfaceValidationError.from_single_issue(
        code="invalid_value",
        path="$.adapter",
        message=(
            f"Unknown adapter {adapter_name!r}. Expected one of "
            f"{sorted(SUPPORTED_SOURCE_ADAPTERS)!r}."
        ),
    )


__all__ = [
    "SUPPORTED_SOURCE_ADAPTERS",
    "adapt_source_payload",
]
