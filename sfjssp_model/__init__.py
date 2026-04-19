"""
SFJSSP core model package.

The canonical executable semantics are defined by `Schedule.check_feasibility`,
`Schedule.evaluate`, and the project-level `SEMANTICS.md`.
"""

from .job import Job, Operation
from .machine import Machine, MachineMode
from .worker import Worker
from .schedule import Schedule
from .instance import (
    AvailabilityWindow,
    MachineBreakdownEvent,
    SFJSSPInstance,
    WorkerAbsenceEvent,
)

__all__ = [
    'Job',
    'Operation',
    'Machine',
    'MachineMode',
    'Worker',
    'Schedule',
    'AvailabilityWindow',
    'MachineBreakdownEvent',
    'SFJSSPInstance',
    'WorkerAbsenceEvent',
]
