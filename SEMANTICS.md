# Canonical Scheduling Semantics

This file is the source-of-truth problem definition for the current codebase.

## Canonical Oracle

The canonical feasibility and objective oracle is:

- `Schedule.check_feasibility(instance)`
- `Schedule.evaluate(instance)`

All solver, environment, and decoder paths should be treated as projections of
that oracle. When a path cannot yet match it exactly, the code or docs must say
so explicitly.

## Hard Feasibility Rules

The current canonical schedule problem enforces:

- Operation coverage: every operation in the instance must be scheduled.
- Job arrival: operation `0` of each job cannot start before `job.arrival_time`.
- Precedence: operation `k` cannot start before operation `k-1` completes, plus:
  - predecessor waiting time: `job.operations[k-1].waiting_time`
  - predecessor transport time when the successor moves to a different machine
- Machine capacity: machine operations cannot overlap, and the gap before an
  operation must contain that operation's setup time.
- Worker capacity: worker operations cannot overlap.
- Resource eligibility: assigned machines and workers must belong to the
  operation's eligible sets.
- Single-period containment: a scheduled operation cannot cross a period
  boundary once placed.
- Rest ratio: for each worker, `rest / (work + rest)` must stay at or above
  `worker.min_rest_fraction`, with rest reconstructed from timeline gaps.
- Ergonomic limit: each worker's maximum per-period exposure must stay at or
  below `worker.ocra_max_per_shift`.

The canonical model does not enforce a "no consecutive periods" rule.

## Timing Conventions

- `setup_time` belongs to the operation being started.
- `transport_time` belongs to the predecessor-to-successor transition and is
  stored on the predecessor scheduled operation once the successor machine is
  known.
- `waiting_time` belongs to the predecessor model operation.
- Period length is `worker.SHIFT_DURATION`.

## Objective Definitions

`Schedule.evaluate(instance)` currently reports:

- `makespan`
- `total_energy`, with breakdown fields for processing, idle, setup, transport,
  startup, and auxiliary energy
- `carbon_emissions`
- ergonomic exposure metrics
- fatigue proxy metrics reconstructed from work/rest timelines
- `total_labor_cost`
- `tool_replacement_cost`
- `total_cost_including_tool_replacement`
- tardiness metrics
- robustness metrics

## External Data Contract Boundary

- The external JSON import contract is `sfjssp_external_v1`.
- That contract is a thin projection onto the canonical instance model, not an
  independent scheduling semantics layer.
- Supported top-level sections in v1 are `schema`, `metadata`, `defaults`,
  `machines`, `workers`, and `jobs`.
- Durations are expressed in minutes, machine power in kW, energy values in
  kWh, labor cost in currency per hour, and ergonomic risk in OCRA-index per
  minute.
- Operation precedence is defined by list order within each job.
- Top-level `transport`, `calendar`, and `events` sections are reserved and are
  rejected by the current importer instead of being silently ignored.

### Energy Accounting

Canonical energy is computed from the schedule timeline:

- processing energy uses the assigned machine mode
- setup energy uses the stored per-operation `setup_time`
- idle energy excludes the part of a gap already consumed by setup
- transport energy uses the predecessor operation's stored `transport_time`
- startup energy is charged once per used machine
- auxiliary energy is charged over each used machine's local active horizon

### Human-State Accounting

- Fatigue is a lightweight dynamic proxy reconstructed from each worker's
  alternating rest gaps and processing spans.
- Learning parameters are stored but are not currently part of canonical
  feasibility or objective evaluation.

## Dynamic Events

Dynamic instances currently support:

- stochastic job arrivals
- stochastic machine breakdowns
- stochastic worker absences at shift boundaries

These event generators are synthetic and not factory-calibrated.

## Explicit Non-Claims

The current codebase does not support the following as canonical claims:

- industrial calibration
- production readiness
- end-to-end learned processing-time reduction from worker learning
- a validated exact MIP formulation
- full semantic parity for every CP objective beyond the documented scope
