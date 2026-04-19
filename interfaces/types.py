"""
Typed return values for external interface operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

try:
    from ..sfjssp_model.instance import SFJSSPInstance
except ImportError:  # pragma: no cover - supports repo-root imports
    from sfjssp_model.instance import SFJSSPInstance


ExternalId = Union[str, int]
TypedIdKey = str


@dataclass(frozen=True)
class IdentifierMaps:
    """Forward and reverse mappings between external and internal identifiers."""

    jobs: Dict[TypedIdKey, int]
    reverse_jobs: Dict[int, ExternalId]
    machines: Dict[TypedIdKey, int]
    reverse_machines: Dict[int, ExternalId]
    workers: Dict[TypedIdKey, int]
    reverse_workers: Dict[int, ExternalId]
    operations: Dict[Tuple[TypedIdKey, TypedIdKey], Tuple[int, int]]
    reverse_operations: Dict[Tuple[int, int], ExternalId]
    machine_modes: Dict[Tuple[TypedIdKey, TypedIdKey], int]
    reverse_machine_modes: Dict[Tuple[int, int], ExternalId]


@dataclass(frozen=True)
class ImportedInstance:
    """Result of validating and importing an external payload."""

    schema: str
    normalized_payload: Dict[str, Any]
    instance: SFJSSPInstance
    id_maps: IdentifierMaps
