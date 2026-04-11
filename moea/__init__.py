"""
Multi-Objective Optimization Package

NSGA-III and other MOEA solvers for SFJSSP.

Evidence Status:
- NSGA-III: CONFIRMED from literature for multi-objective scheduling
- Application to SFJSSP: PROPOSED synthesis
"""

from .nsga3 import (
    NSGA3,
    Individual,
    Population,
    create_sfjssp_genome,
    evaluate_sfjssp_genome,
)

__all__ = [
    'NSGA3',
    'Individual',
    'Population',
    'create_sfjssp_genome',
    'evaluate_sfjssp_genome',
]
