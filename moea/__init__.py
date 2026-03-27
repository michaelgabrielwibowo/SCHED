"""
Multi-Objective Optimization Package

NSGA-III and other MOEA solvers for SFJSSP.

Evidence Status:
- NSGA-III: CONFIRMED from literature for multi-objective scheduling
- Application to SFJSSP: PROPOSED synthesis
"""

from .nsga3 import NSGA3, Individual, Population

__all__ = [
    'NSGA3',
    'Individual',
    'Population',
]
