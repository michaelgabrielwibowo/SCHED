"""
SFJSSP package entry point.

This repository is a research implementation with a canonical executable
problem definition documented in `SEMANTICS.md`.
"""

__version__ = "0.1.0"
__author__ = "SFJSSP Research Team"

try:
    from .sfjssp_model import (
        Job,
        Machine,
        MachineMode,
        Operation,
        Schedule,
        SFJSSPInstance,
        Worker,
    )
    from .environment import SFJSSPEnv
    from .baseline_solver import GreedyScheduler, edt_rule, fifo_rule, spt_rule
    from .utils import BenchmarkGenerator, GeneratorConfig, InstanceSize
except ImportError:  # pragma: no cover - fallback for direct module import
    from sfjssp_model import (
        Job,
        Machine,
        MachineMode,
        Operation,
        Schedule,
        SFJSSPInstance,
        Worker,
    )
    from environment import SFJSSPEnv
    from baseline_solver import GreedyScheduler, edt_rule, fifo_rule, spt_rule
    from utils import BenchmarkGenerator, GeneratorConfig, InstanceSize

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
    'InstanceSize',
]
