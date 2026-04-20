"""
Canonical calendar and event primitives for SFJSSP resources.

These types are shared by the core model so worker, machine, instance, and
schedule semantics can reason over the same serialized time windows/events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, List


def _stable_details_text(details: Dict[str, Any]) -> str:
    """Create a deterministic sort key for arbitrary metadata payloads."""
    return json.dumps(details, sort_keys=True, default=str)


@dataclass(frozen=True)
class AvailabilityWindow:
    """Explicit unavailability interval for one machine or worker."""

    start_time: float
    end_time: float
    reason: str = "unavailable"
    source: str = "calendar"
    details: Dict[str, Any] = field(default_factory=dict)
    event_id: str = ""

    def __post_init__(self):
        if self.start_time < 0.0:
            raise ValueError("AvailabilityWindow.start_time must be non-negative")
        if self.end_time < self.start_time:
            raise ValueError("AvailabilityWindow.end_time must be >= start_time")

    def overlaps(self, start_time: float, end_time: float) -> bool:
        """True when the half-open intervals [start, end) overlap."""
        return start_time < self.end_time and end_time > self.start_time

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "reason": self.reason,
            "source": self.source,
            "details": dict(self.details),
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AvailabilityWindow":
        return cls(
            start_time=data["start_time"],
            end_time=data["end_time"],
            reason=data.get("reason", "unavailable"),
            source=data.get("source", "calendar"),
            details=data.get("details", {}),
            event_id=data.get("event_id", ""),
        )


@dataclass(frozen=True)
class ShiftWindow:
    """Worker shift-availability interval used to derive off-shift windows."""

    start_time: float
    end_time: float
    shift_label: str = "shift"
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.start_time < 0.0:
            raise ValueError("ShiftWindow.start_time must be non-negative")
        if self.end_time < self.start_time:
            raise ValueError("ShiftWindow.end_time must be >= start_time")

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "shift_label": self.shift_label,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShiftWindow":
        return cls(
            start_time=data["start_time"],
            end_time=data["end_time"],
            shift_label=data.get("shift_label", "shift"),
            details=data.get("details", {}),
        )


@dataclass(frozen=True)
class MachineBreakdownEvent:
    """Typed machine breakdown event for canonical replay and serialization."""

    machine_id: int
    start_time: float
    repair_duration: float
    source: str = "generated"
    details: Dict[str, Any] = field(default_factory=dict)
    event_id: str = ""

    @property
    def end_time(self) -> float:
        return self.start_time + self.repair_duration

    def to_availability_window(self) -> AvailabilityWindow:
        details = {
            "machine_id": self.machine_id,
            "repair_duration": self.repair_duration,
        }
        details.update(self.details)
        return AvailabilityWindow(
            start_time=self.start_time,
            end_time=self.end_time,
            reason="breakdown",
            source=self.source,
            details=details,
            event_id=self.event_id,
        )

    def to_dict(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "start_time": self.start_time,
            "repair_duration": self.repair_duration,
            "end_time": self.end_time,
            "source": self.source,
            "details": dict(self.details),
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MachineBreakdownEvent":
        if "repair_duration" in data:
            repair_duration = data["repair_duration"]
        else:
            repair_duration = data["end_time"] - data["start_time"]
        return cls(
            machine_id=data["machine_id"],
            start_time=data["start_time"],
            repair_duration=repair_duration,
            source=data.get("source", "generated"),
            details=data.get("details", {}),
            event_id=data.get("event_id", ""),
        )


@dataclass(frozen=True)
class WorkerAbsenceEvent:
    """Typed worker absence event for canonical replay and serialization."""

    worker_id: int
    start_time: float
    end_time: float
    source: str = "generated"
    details: Dict[str, Any] = field(default_factory=dict)
    event_id: str = ""

    def __post_init__(self):
        if self.start_time < 0.0:
            raise ValueError("WorkerAbsenceEvent.start_time must be non-negative")
        if self.end_time < self.start_time:
            raise ValueError("WorkerAbsenceEvent.end_time must be >= start_time")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_availability_window(self) -> AvailabilityWindow:
        details = {"worker_id": self.worker_id, "duration": self.duration}
        details.update(self.details)
        return AvailabilityWindow(
            start_time=self.start_time,
            end_time=self.end_time,
            reason="absence",
            source=self.source,
            details=details,
            event_id=self.event_id,
        )

    def to_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "source": self.source,
            "details": dict(self.details),
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerAbsenceEvent":
        end_time = data.get("end_time")
        if end_time is None:
            end_time = data["start_time"] + data["duration"]
        return cls(
            worker_id=data["worker_id"],
            start_time=data["start_time"],
            end_time=end_time,
            source=data.get("source", "generated"),
            details=data.get("details", {}),
            event_id=data.get("event_id", ""),
        )


def sort_windows(windows: Iterable[AvailabilityWindow]) -> List[AvailabilityWindow]:
    """Return windows in deterministic canonical order."""
    return sorted(
        windows,
        key=lambda window: (
            window.start_time,
            window.end_time,
            window.reason,
            window.source,
            window.event_id,
            _stable_details_text(window.details),
        ),
    )


def sort_shift_windows(windows: Iterable[ShiftWindow]) -> List[ShiftWindow]:
    """Return shift windows in deterministic canonical order."""
    return sorted(
        windows,
        key=lambda window: (
            window.start_time,
            window.end_time,
            window.shift_label,
            _stable_details_text(window.details),
        ),
    )


def sort_machine_breakdown_events(
    events: Iterable[MachineBreakdownEvent],
) -> List[MachineBreakdownEvent]:
    """Return machine events in deterministic canonical order."""
    return sorted(
        events,
        key=lambda event: (
            event.start_time,
            event.end_time,
            event.machine_id,
            event.event_id,
            _stable_details_text(event.details),
        ),
    )


def sort_worker_absence_events(
    events: Iterable[WorkerAbsenceEvent],
) -> List[WorkerAbsenceEvent]:
    """Return worker events in deterministic canonical order."""
    return sorted(
        events,
        key=lambda event: (
            event.start_time,
            event.end_time,
            event.worker_id,
            event.event_id,
            _stable_details_text(event.details),
        ),
    )
