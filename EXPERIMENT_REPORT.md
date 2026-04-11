# SFJSSP Verification Report

**Date:** 2026-04-11  
**Status:** Current engineering verification summary

## Executive Summary

This repository is a research implementation, not a validated production scheduler. The current codebase now has a root-runnable package boundary, benchmark-backed regression tests, corrected time-unit handling for energy and labor, a repaired Gym observation contract, and a working NSGA-III demo path that no longer collapses to all-penalty solutions.

What is verified now:
- `pytest -q` passes from repo root.
- Small and medium committed benchmarks load as full instances and pass greedy regression tests.
- Due dates are treated as soft constraints through tardiness objectives.
- The flat Gym observation path returns a real `numpy.ndarray` accepted by `observation_space.contains(...)`.
- The NSGA-III genome now includes machine modes and the bundled demo produces non-penalty schedules.
- `torch` and `ortools` are optional dependencies, and both optional paths have now been exercised in this environment.
- OR-Tools CP-SAT runs end to end for the `makespan` objective on stored small benchmarks and returns feasible schedules.
- Torch-backed DRL training runs one episode end to end and writes checkpoints/history.

What is still not verified in this environment:
- Any claim of MIP exact-solver correctness. That path is now quarantined and raises a clear `NotImplementedError`.
- Any claim of real industrial validity or calibrated benchmark realism beyond the synthetic generator assumptions.

## Current Implementation Status

| Component | Status | Notes |
|---|---|---|
| Core model | Verified | Serialization, unit handling, and soft due-date behavior covered by tests |
| Greedy solver | Verified baseline | Small/medium benchmark regression tests pass |
| Gym environment | Verified contract | Flat observation mode and resource masks tested |
| NSGA-III | Verified demo path | Example script returns non-penalty solutions |
| Exact solvers | CP makespan verified, MIP quarantined | CP-SAT smoke passes on stored small benchmarks for makespan; MIP is blocked behind a clear error |
| DRL training | Torch path smoke verified | One-episode PPO run completes and saves artifacts |

## Verification Commands

```bash
pytest -q
python -m experiments.test_nsga3
python verify_all_solvers.py
```

## Important Modeling Decisions

1. Due dates are soft constraints. Late jobs contribute tardiness cost and do not make a schedule infeasible.
2. The former "resilience" metric is treated as schedule robustness: load balance plus slack buffer.
3. Energy is computed in kWh from durations stored in minutes.
4. Labor cost is computed from hourly rates with durations stored in minutes.
5. `torch` and `ortools` are optional. The core model, greedy solvers, and NSGA-III path do not require them.
6. The CP exact solver is currently the only verified exact-solver path, and only for the makespan objective on stored small benchmarks; the MIP formulation is quarantined until revalidated.

## Known Limits

1. The MIP exact-solver formulation remains unvalidated and intentionally unavailable.
2. CP objective variants beyond makespan still need revalidation before being treated as verified.
3. Torch-backed policy/training quality is only smoke-verified, not performance-validated.
4. Dynamic events in the Gym environment have partial support only.
5. Benchmark realism remains synthetic and literature-calibrated rather than factory-measured.

## Recommended Next Checks

1. Re-run comparison scripts and regenerate committed result artifacts with provenance.
2. Revalidate CP objective variants beyond makespan before exposing them as verified exact-solver modes.
3. Rebuild or replace the quarantined MIP formulation before re-exposing it as a supported solver.
