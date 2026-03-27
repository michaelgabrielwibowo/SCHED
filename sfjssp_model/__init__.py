"""
SFJSSP Core Model Package
Sustainable Flexible Job-Shop Scheduling Problem data structures

Evidence Status: PROPOSED implementation based on literature synthesis
- Target survey (2023) conceptual model
- E-DFJSP 2025 energy model
- DyDFJSP 2023 fatigue dynamics
- NSGA-III 2021 ergonomic indices
"""

from .job import Job, Operation
from .machine import Machine, MachineMode
from .worker import Worker
from .schedule import Schedule
from .instance import SFJSSPInstance

__all__ = [
    'Job',
    'Operation',
    'Machine',
    'MachineMode',
    'Worker',
    'Schedule',
    'SFJSSPInstance',
]