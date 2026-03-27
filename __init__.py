"""
SFJSSP Code Package

Sustainable Flexible Job-Shop Scheduling Problem implementation
for Industry 5.0 context.

Evidence Status:
- Core model: PROPOSED synthesis of literature components
- Components confirmed from:
  - Standard FJSSP [CONFIRMED]
  - DRCFJSSP dual resources [CONFIRMED]
  - E-DFJSP 2025 energy modeling [CONFIRMED]
  - DyDFJSP 2023 fatigue dynamics [CONFIRMED]
  - NSGA-III 2021 ergonomic indices [CONFIRMED]

No claim that this exact integration exists in literature.
This is a research implementation based on synthesis.
"""

__version__ = "0.1.0"
__author__ = "SFJSSP Research Team"

from sfjssp_model import (
    Job,
    Operation,
    Machine,
    MachineMode,
    Worker,
    Schedule,
    SFJSSPInstance,
)

from environment import SFJSSPEnv

from baseline_solver import GreedyScheduler, spt_rule, fifo_rule, edt_rule

from utils import BenchmarkGenerator, GeneratorConfig

__all__ = [
    # Core model
    'Job',
    'Operation',
    'Machine',
    'MachineMode',
    'Worker',
    'Schedule',
    'SFJSSPInstance',
    # Environment
    'SFJSSPEnv',
    # Baseline solvers
    'GreedyScheduler',
    'spt_rule',
    'fifo_rule',
    'edt_rule',
    # Utilities
    'BenchmarkGenerator',
    'GeneratorConfig',
]
