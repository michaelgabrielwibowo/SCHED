# SFJSSP Implementation

Sustainable Flexible Job-Shop Scheduling Problem (SFJSSP) implementation for Industry 5.0 research.

## Overview

This codebase implements scheduling optimization for manufacturing systems considering:
- **Economic objectives**: Makespan, tardiness
- **Environmental objectives**: Energy consumption, carbon emissions
- **Human objectives**: Ergonomic risk, worker fatigue, labor cost
- **Schedule robustness**: Load balance and slack buffer against disruptions

## Canonical Semantics

The current source-of-truth scheduling semantics are documented in
[SEMANTICS.md](/C:/Users/s1233/SCHEDULE/SEMANTICS.md).

This repository is a research implementation. Literature informed many
components, but the integrated executable problem here is defined by the code
and `SEMANTICS.md`, not by any single paper or survey.

## External Input Contract

The first external workflow slice is available under `interfaces/` through the
versioned schema `sfjssp_external_v1`, exposed as either a JSON document or a
fixed CSV bundle that normalizes into the same canonical payload.

Current supported top-level sections:
- `schema`
- `metadata`
- `defaults`
- `machines`
- `workers`
- `jobs`

Current v1 behavior:
- durations are in minutes
- machine power is in kW
- energy inputs are in kWh
- labor cost is in currency per hour
- ergonomic risk rates are in OCRA-index per minute
- operation precedence is the list order within each job
- operation transport and waiting delays are provided per operation
- unknown fields are rejected in strict mode
- reserved top-level sections `transport`, `calendar`, and `events` are rejected in v1 instead of being silently ignored

The importer is intentionally narrower than `SFJSSPInstance.to_dict()`. It is a
thin validation layer over the canonical model, not a second semantics engine.

```python
from interfaces import load_instance_from_csv_bundle, load_instance_from_json

imported = load_instance_from_json("tests/fixtures/interfaces/valid_minimal.json")
instance = imported.instance

csv_imported = load_instance_from_csv_bundle(
    "tests/fixtures/interfaces_csv/valid_minimal"
)
```

The matching audit surface is `build_schedule_audit(...)`, which emits the
versioned payload `schedule_audit_v1` from the canonical schedule oracle rather
than from solver-specific diagnostics.

Stable file exports are written by `export_schedule_artifacts(...)`, which
produces `run_manifest.json`, `schedule.json`, `operations.csv`,
`machine_timeline.csv`, `worker_timeline.csv`, `violations.json`, and
`violations.csv`.

CLI happy path:

```bash
python -m interfaces.cli validate-input --input tests/fixtures/interfaces/valid_minimal.json
python -m interfaces.cli run --input tests/fixtures/interfaces/valid_minimal.json --solver greedy:spt --output-dir out
python -m interfaces.cli validate-input --input tests/fixtures/interfaces_csv/valid_minimal
```

## Project Structure

```
sfjssp_code/
├── sfjssp_model/          # Core data structures
│   ├── job.py             # Job, Operation
│   ├── machine.py         # Machine, MachineMode, energy states
│   ├── worker.py          # Worker with fatigue, OCRA, learning
│   ├── instance.py        # SFJSSPInstance, dynamic events
│   └── schedule.py        # Schedule with multi-objective evaluation
├── environment/           # Gym environment for DRL
│   └── sfjssp_env.py      # SFJSSPEnv class
├── baseline_solver/       # Greedy heuristics
│   └── greedy_solvers.py  # FIFO, SPT, EDD, composite rules
├── moea/                  # Multi-objective evolutionary algorithms
│   └── nsga3.py           # NSGA-III implementation
├── agents/                # DRL policy networks
│   └── policy_networks.py # Job/Machine/Worker agent networks
├── training/              # Training pipelines
│   └── train_drl.py       # PPO training loop
├── experiments/           # Experiment scripts
│   ├── generate_benchmarks.py
│   └── test_nsga3.py
├── tests/                 # Unit tests
│   └── test_core_model.py
└── benchmarks/            # Generated benchmark instances
```

## Installation

### Core Install

```bash
pip install -r requirements.txt
```

### Optional DRL Support

```bash
pip install torch
```

### Exact solvers (optional)

```bash
pip install ortools>=9.8.0
```

`Schedule.check_feasibility()` and `Schedule.evaluate()` are the canonical
oracle. Solver paths that cannot yet match that oracle exactly should be
treated as research paths rather than normative references.

### Development

```bash
pytest -q
```

## Quick Start

### 1. Generate Benchmark Instance

```bash
python -m experiments.generate_benchmarks --mode example --output benchmarks
python -m experiments.generate_benchmarks --mode suite --sizes small,medium --output benchmarks
```

### 2. Run Greedy Scheduler

```python
from sfjssp_model import SFJSSPInstance, Machine, Worker, Job, Operation
from baseline_solver import GreedyScheduler, spt_rule

# Create instance
instance = SFJSSPInstance(instance_id="test")
# ... add machines, workers, jobs ...

# Schedule
scheduler = GreedyScheduler(job_rule=spt_rule)
schedule = scheduler.schedule(instance, verbose=True)

# Evaluate
objectives = schedule.evaluate(instance)
print(f"Makespan: {objectives['makespan']}")
print(f"Energy: {objectives['total_energy']}")
```

### 3. Run NSGA-III Optimization

```bash
python -m experiments.test_nsga3
```

### 4. Train DRL Agent (`torch` optional)

```bash
python -m training.train_drl --episodes 100
```

## API Reference

### Core Model

```python
from sfjssp_model import (
    SFJSSPInstance,   # Problem instance
    Job, Operation,   # Job structure
    Machine,          # Machine with energy modes
    Worker,           # Worker with fatigue, ergonomics
    Schedule,         # Solution representation
)
```

### Environment

```python
import numpy as np

from environment import SFJSSPEnv

env = SFJSSPEnv(instance, use_graph_state=False)
obs, info = env.reset(seed=42)

job_idx = int(np.argmax(env._compute_job_mask()))
machine_idx, worker_idx, mode_idx = np.argwhere(
    env.compute_resource_mask(job_idx) > 0
)[0]
action = {
    "job_idx": job_idx,
    "op_idx": 0,
    "machine_idx": int(machine_idx),
    "worker_idx": int(worker_idx),
    "mode_idx": int(mode_idx),
}

obs, reward, terminated, truncated, info = env.step(action)
```

### Baseline Solvers

```python
from baseline_solver import (
    GreedyScheduler,
    spt_rule,      # Shortest Processing Time
    fifo_rule,     # First In First Out
    edt_rule,      # Earliest Due Date
    least_slack_rule,  # Least Slack Time
    critical_ratio_rule,  # Critical Ratio
    composite_rule, # Multi-objective composite
    tardiness_composite_rule, # Due-date biased composite
)

scheduler = GreedyScheduler(job_rule=spt_rule)
schedule = scheduler.schedule(instance)
```

### NSGA-III

```python
from moea import (
    NSGA3,
    create_sfjssp_genome,
    create_sfjssp_seed_genomes,
    evaluate_sfjssp_genome,
)

nsga3 = NSGA3(
    n_objectives=4,
    population_size=100,
    n_generations=100,
)

nsga3.set_problem(
    evaluate_fn=evaluate_sfjssp_genome,
    create_individual_fn=create_sfjssp_genome,
    seed_individuals_fn=create_sfjssp_seed_genomes,
)

nsga3.evolve(instance)
pareto = nsga3.get_pareto_solutions()
```

## Benchmarks

### Generated Instances

The benchmark generator creates instances with explicit labeling:
- **FULLY_SYNTHETIC**: All parameters computer-generated
- **CALIBRATED_SYNTHETIC**: Synthetic instances whose parameter ranges were
  chosen with literature-informed defaults, not factory calibration

Saved benchmark documents are emitted under the canonical schema documented in
`utils/benchmark_document.schema.json`. New generated artifacts include
document type/version fields and generator provenance so benchmark inputs can be
reconstructed from one public contract.

Committed benchmark policy:
- `benchmarks/small/*.json` and `benchmarks/medium/*.json` are the canonical
  regression fixtures loaded by `pytest`.
- `benchmarks/large/`, `benchmarks/example_001.json`, and
  `experiments/results/` are disposable generated outputs. They should be
  regenerated locally when needed rather than treated as committed regression
  artifacts.

### Instance Sizes

| Size | Jobs | Machines | Workers |
|------|------|----------|---------|
| Small | 10 | 5 | 5 |
| Medium | 50 | 10 | 10 |
| Large | 200 | 20 | 20 |

The public generator does not expose an `industrial` preset. The environment
currently enforces the same upper bound as the `large` preset (`200/20/20`), so
larger synthetic scales should not be published as supported benchmarks until
runtime parity exists.

## Objectives

The SFJSSP optimizes multiple objectives:

1. **Makespan** (f1): Minimize completion time
2. **Energy** (f2): Minimize total energy consumption
3. **Carbon** (f3): Minimize CO2 emissions
4. **Ergonomic Risk** (f4): Minimize maximum OCRA index
5. **Labor Cost** (f5): Minimize worker costs
6. **Schedule Robustness** (f6): Balance machine/worker load and preserve slack buffers

## Constraints

### Hard Constraints
- Operation precedence
- Job arrival times
- Machine eligibility
- Worker eligibility
- Ergonomic exposure limits
- Worker rest requirements (12.5% minimum)

### Soft Constraints
- Energy budgets
- Comfort preferences
- Due date targets

## Limitations

1. **No real-world validation**: Parameters based on literature, not measured data
2. **Scalability**: Exact methods struggle with >50 jobs
3. **DRL stability**: Multi-agent training may require tuning
4. **Ergonomic approximation**: OCRA indices are simplified

## Testing

```bash
pytest -q
python -m experiments.test_nsga3
python verify_all_solvers.py
```

## Current Status

- Root package import works from a clean repo checkout.
- Small and medium benchmark JSON files load as full instances and are exercised in pytest.
- `benchmarks/small` and `benchmarks/medium` are the only committed benchmark
  fixtures; generated large/example slices and `experiments/results/` are
  intentionally disposable outputs.
- Due dates are modeled as soft constraints via tardiness objectives, not feasibility rejection.
- The Gym environment now returns a true flat `Box` observation when `use_graph_state=False`.
- NSGA-III now encodes machine modes and the bundled demo produces non-penalty schedules.
- `torch` and `ortools` are optional, and both optional paths have now been smoke-tested in this environment.
- The CP-SAT path is parity-tested only on narrow canonical fixtures for the
  `makespan` objective; non-makespan objectives remain experimental surrogate
  modes, and stored small benchmarks are not yet parity-validated under machine
  setup semantics.
- The torch-backed PPO training entrypoint runs one episode and writes checkpoints/history.
- The MIP exact solver is currently quarantined and raises a clear `NotImplementedError` instead of returning invalid schedules.
- Generated solver comparison artifacts embed provenance metadata, including git
  commit/dirty state, and record an explicit NSGA report-member policy
  (`best_makespan_feasible` by default) alongside the tardiness-best feasible
  Pareto representative.
- NSGA-III now supports deterministic greedy warm-start seeding via `create_sfjssp_seed_genomes`, including `least_slack_rule`, `critical_ratio_rule`, and `tardiness_composite_rule`; the current seed path accepts all 10 deterministic seed candidates on each stored small benchmark in the bundled `30`-generation comparison slice.
- The latest local NSGA budget sweep still recommends `30` generations with
  population `30`; larger budgets did not improve either the default NSGA
  report member or the tardiness-best feasible result under fixed seed `42`.

## Citation

If you use this code:

```bibtex
@software{sfjssp_2026,
  title = {SFJSSP Implementation},
  author = {SFJSSP Research Team},
  year = {2026},
  note = {Based on synthesis of 135 surveyed papers}
}
```

## License

MIT License (academic use)

## Contributing

This is a research implementation. Key areas for contribution:
1. Real industrial case studies
2. Parameter calibration from actual factories
3. Improved DRL architectures
4. Schedule robustness metric validation
