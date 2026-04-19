"""
Structured validation errors for the external interface layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


SCHEMA_MISMATCH = "schema_mismatch"
MISSING_REQUIRED = "missing_required"
UNKNOWN_FIELD = "unknown_field"
UNSUPPORTED_SECTION = "unsupported_section"
INVALID_TYPE = "invalid_type"
INVALID_VALUE = "invalid_value"
DUPLICATE_ID = "duplicate_id"
INVALID_REFERENCE = "invalid_reference"
EMPTY_COLLECTION = "empty_collection"


@dataclass(frozen=True)
class ValidationIssue:
    """One structured validation issue in the external payload."""

    code: str
    path: str
    message: str
    hint: Optional[str] = None


class InterfaceValidationError(ValueError):
    """Raised when an external payload fails schema validation."""

    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues: List[ValidationIssue] = list(issues)
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if not self.issues:
            return "External input validation failed."
        rendered = []
        for issue in self.issues[:5]:
            line = f"{issue.code} at {issue.path}: {issue.message}"
            if issue.hint:
                line = f"{line} Hint: {issue.hint}"
            rendered.append(line)
        if len(self.issues) > 5:
            rendered.append(f"... {len(self.issues) - 5} more issue(s)")
        return "External input validation failed:\n" + "\n".join(rendered)


def path_to_string(*segments: object) -> str:
    """Render a JSON-style location path."""

    path = "$"
    for segment in segments:
        if isinstance(segment, int):
            path = f"{path}[{segment}]"
        else:
            path = f"{path}.{segment}"
    return path


def issue(
    code: str,
    *segments: object,
    message: str,
    hint: Optional[str] = None,
) -> ValidationIssue:
    """Create a structured validation issue."""

    return ValidationIssue(code=code, path=path_to_string(*segments), message=message, hint=hint)
