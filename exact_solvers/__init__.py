"""
Exact Solvers Package

Constraint Programming (CP) and Mixed-Integer Programming (MIP) solvers.

Current support status:
- `CPScheduler` is the verified exact-solver path for the `makespan` objective.
- `MIPScheduler` is intentionally quarantined and raises `NotImplementedError`
  until its formulation is revalidated against current schedule semantics.

Evidence Status:
- CP for scheduling: CONFIRMED from literature
- Application to SFJSSP: PROPOSED
"""

from .cp_solver import CPScheduler, MIPScheduler

__all__ = [
    'CPScheduler',
    'MIPScheduler',
]
