# SFJSSP Verification Report

**Date:** 2026-04-13  
**Status:** Current engineering verification summary

## Executive Summary

This repository is a research implementation, not a validated production
scheduler. Canonical problem semantics now live in
[SEMANTICS.md](/C:/Users/s1233/SCHEDULE/SEMANTICS.md); verification notes in
this file should be read as implementation status, not as a replacement for the
schedule-level oracle.

What is verified now:
- `pytest -q` passes from repo root.
- Small and medium committed benchmarks load as full instances and pass greedy regression tests.
- Due dates are treated as soft constraints through tardiness objectives.
- The flat Gym observation path returns a real `numpy.ndarray` accepted by `observation_space.contains(...)`.
- The NSGA-III genome now includes machine modes and the bundled demo produces non-penalty schedules.
- `torch` and `ortools` are optional dependencies, and both optional paths have now been exercised in this environment.
- OR-Tools CP-SAT runs end to end for the `makespan` objective on narrow canonical fixtures with zero setup-time ambiguity.
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
| Exact solvers | CP makespan parity is narrow, MIP quarantined | CP-SAT is parity-tested on narrow canonical fixtures for makespan only; MIP is blocked behind a clear error |
| DRL training | Torch path smoke verified | One-episode PPO run completes and saves artifacts |

## Verification Commands

```bash
pytest -q
python -m experiments.test_nsga3
python verify_all_solvers.py
```

## Local Comparison Artifact

Comparison outputs under `experiments/results/` are intentionally disposable and
are no longer committed as regression artifacts. The latest reviewed local run
used the following command and provenance:

- Command: `python -m experiments.compare_solvers --benchmark-dir benchmarks/small --output experiments/results/comparison_2026-04-13-report-policy.json --cp --generations 30 --nsga-report-member-policy best_makespan_feasible`
- Commit: `09eb0a5626fb88fe200023f237a2c9dc7d27530d`
- Worktree state at run time: dirty (`git_dirty: true`, with `git_status_short` embedded in the artifact provenance)
- Scope: stored `benchmarks/small`, NSGA-III `30` generations with greedy warm-start enabled, report policy `best_makespan_feasible`, CP enabled, CP objective restricted to `makespan`
- Artifact schema: `comparison_results_v3` with embedded provenance
- Staging review: greedy runs have no recorded constraint violations, NSGA-III report-member makespans are raw schedule metrics rather than penalty-scale sentinel values, CP appears only as `objective="makespan"`, and the artifact now records both the published NSGA report member and the tardiness-best feasible representative.

Observed outcomes from the local comparison run:
- All greedy comparison runs on the stored small benchmarks are now hard-feasible and report no constraint violations.
- NSGA-III now reports raw schedule metrics rather than penalized selection values.
- The comparison artifact now makes the public NSGA contract explicit: the published report member is `best_makespan_feasible`, and the legacy `selected_*` fields are compatibility aliases to that report member.
- Deterministic greedy warm-start seeds are now enabled in the comparison path, and the repaired seed decoder accepted all 10 deterministic greedy-rule seeds on all 10 stored small benchmarks, including `least_slack_rule`, `critical_ratio_rule`, and `tardiness_composite_rule`.
- On this `30`-generation comparison slice, average NSGA-III makespan improved from `5393.9` in the previous tracked artifact to `754.7`, average `min_total_penalty` dropped from `487564.7` to `29226.1`, and NSGA-III now beats the best greedy makespan on `3/10` stored small benchmarks.
- Those local CP numbers were produced before the setup-gap truthfulness tightening in the canonical oracle, so they should be treated as historical diagnostics rather than current parity evidence.
- Even after the NSGA repair, the local comparison slice still carries non-zero soft-constraint penalty mass on every stored small benchmark under the default report-member policy.
- The report-member penalty fields show what remains under that default policy: average report-member `n_tardy_jobs=8.0`, average report-member `weighted_tardiness=2185.5`, average report-member `total_penalty=29854.9`, average report-member `max_ergonomic_exposure=1.077`, zero hard violations, and zero OCRA penalty across all 10 stored small benchmarks.
- The representative-member fields also show where the next improvement should come from: the feasible Pareto front already contains a better tardiness member on `2/10` stored small benchmarks, with average tardiness-best `weighted_tardiness=2148.6` and average tardiness-best `n_tardy_jobs=7.8`.

The comparison script now clones the benchmark instance before each solver run so solver results are no longer contaminated by mutable state left behind by previous experiments.

## NSGA Budget Sweep

Budget sweep outputs under `experiments/results/` are also disposable local
artifacts rather than committed fixtures. The latest reviewed sweep used:

- Command: `python -m experiments.sweep_nsga_budget --benchmark-dir benchmarks/small --output experiments/results/nsga_budget_sweep_2026-04-13.json --generations 30,60,120 --population-sizes 30,60 --seed 42 --nsga-report-member-policy best_makespan_feasible`
- Sweep scope: stored `benchmarks/small`, warm-start enabled, report policy `best_makespan_feasible`, published greedy baseline slice (`SPT`, `FIFO`, `EDD`), fixed RNG seed `42`
- Sweep schema: `nsga_budget_sweep_v2`
- Budget matrix: generations `{30, 60, 120}` x population sizes `{30, 60}`

Observed outcome from the sweep:
- All six configurations produced the same report-member average makespan (`754.7`), average report-member weighted tardiness (`2185.5`), average report-member tardy-job count (`8.0`), average tardiness-best weighted tardiness (`2148.6`), average tardiness-best tardy-job count (`7.8`), greedy-win count (`3/10`), and zero-tardy count (`0/10`).
- Increasing generations or population size only increased runtime. The recommended budget remains the current baseline: `30` generations and population `30`.
- Under the current initialization, crossover, mutation, and report-member policy, a larger NSGA budget alone is not the next useful lever.

## Important Modeling Decisions

1. Due dates are soft constraints. Late jobs contribute tardiness cost and do not make a schedule infeasible.
2. The former "resilience" metric is treated as schedule robustness: load balance plus slack buffer.
3. Energy is computed in kWh from durations stored in minutes.
4. Labor cost is computed from hourly rates with durations stored in minutes.
5. `torch` and `ortools` are optional. The core model, greedy solvers, and NSGA-III path do not require them.
6. The CP exact-solver path is only parity-tested on narrow canonical fixtures for the makespan objective; stored small benchmarks are not yet parity-validated under machine setup semantics, and the MIP formulation remains quarantined until revalidated.

## Known Limits

1. The MIP exact-solver formulation remains unvalidated and intentionally unavailable.
2. CP objective variants beyond makespan still need revalidation before being treated as verified, and even the makespan path should be treated as narrow-scope parity coverage rather than benchmark-wide proof.
3. Torch-backed policy/training quality is only smoke-verified, not performance-validated.
4. Dynamic events in the Gym environment have partial support only.
5. Benchmark realism remains synthetic and literature-calibrated rather than factory-measured.
6. NSGA-III comparison quality improved materially after fixing the warm-start decoder, but it still carries tardiness-driven soft-constraint penalty mass on every stored small benchmark in the reviewed `30`-generation slice.
7. The April 13, 2026 budget sweep showed no benefit from simply increasing generations or population size under the current fixed-seed setup and default report-member policy.

## Recommended Next Checks

1. Pursue algorithmic tardiness reduction rather than more reporting work. The report-member semantics are now explicit, and the April 13, 2026 budget sweep showed that longer runs and larger populations alone do not improve either the default report member or the tardiness-best feasible representative on the stored small benchmark slice under fixed seed `42`.
2. Revalidate CP objective variants beyond makespan before exposing them as verified exact-solver modes.
3. Rebuild or replace the quarantined MIP formulation before re-exposing it as a supported solver.
