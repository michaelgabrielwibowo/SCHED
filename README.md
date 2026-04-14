# SFJSSP Implementation

Sustainable Flexible Job-Shop Scheduling Problem (SFJSSP) implementation for Industry 5.0 research.

## Overview

This codebase implements scheduling optimization for manufacturing systems considering:
- **Economic objectives**: Makespan, tardiness
- **Environmental objectives**: Energy consumption, carbon emissions
- **Human objectives**: Ergonomic risk, worker fatigue, labor cost
- **Schedule robustness**: Load balance and slack buffer against disruptions

## Evidence Status

**IMPORTANT**: This is a **research implementation** based on synthesis of literature components:

| Component | Source | Status |
|-----------|--------|--------|
| Basic FJSSP structure | Standard scheduling literature | CONFIRMED |
| Dual-resource (DRCFJSSP) | Gong et al. 2018, others | CONFIRMED |
| Energy modeling | E-DFJSP 2025 | CONFIRMED |
| Fatigue dynamics | DyDFJSP 2023 | CONFIRMED |
| Ergonomic indices (OCRA) | NSGA-III 2021 study | CONFIRMED |
| **Full SFJSSP integration** | **This work** | **PROPOSED** |

**No existing paper validates this exact combination.** This is a novel research implementation.

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

Only the CP `makespan` path is currently verified in this repository. CP
objectives `energy`, `ergonomic`, and `composite` remain experimental until
they are revalidated against the schedule-level metrics.

### Development

```bash
pytest -q
```

## Quick Start

### 1. Generate Benchmark Instance

```bash
python -m experiments.generate_benchmarks --mode example --output benchmarks
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
- **CALIBRATED_SYNTHETIC**: Calibrated against literature values

Calibration sources:
- Fatigue parameters: DyDFJSP 2023
- Energy parameters: E-DFJSP 2025
- Ergonomic parameters: NSGA-III 2021

### Instance Sizes

| Size | Jobs | Machines | Workers |
|------|------|----------|---------|
| Small | 10 | 5 | 5 |
| Medium | 50 | 10 | 10 |
| Large | 200 | 20 | 20 |

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
- Due dates are modeled as soft constraints via tardiness objectives, not feasibility rejection.
- The Gym environment now returns a true flat `Box` observation when `use_graph_state=False`.
- NSGA-III now encodes machine modes and the bundled demo produces non-penalty schedules.
- `torch` and `ortools` are optional, and both optional paths have now been smoke-tested in this environment.
- The CP exact solver is smoke-verified for the `makespan` objective on stored small benchmarks.
- CP objective variants `energy`, `ergonomic`, and `composite` remain experimental until they are revalidated against the schedule-level objective calculations.
- The torch-backed PPO training entrypoint runs one episode and writes checkpoints/history.
- The MIP exact solver is currently quarantined and raises a clear `NotImplementedError` instead of returning invalid schedules.
- The tracked solver comparison artifact now embeds provenance metadata, including git commit/dirty state, and records an explicit NSGA report-member policy (`best_makespan_feasible` by default) alongside the tardiness-best feasible Pareto representative.
- NSGA-III now supports deterministic greedy warm-start seeding via `create_sfjssp_seed_genomes`, including `least_slack_rule`, `critical_ratio_rule`, and `tardiness_composite_rule`; the current seed path accepts all 10 deterministic seed candidates on each stored small benchmark in the bundled `30`-generation comparison slice.
- The April 13, 2026 NSGA budget sweep still recommends `30` generations with population `30`; larger budgets did not improve either the default NSGA report member or the tardiness-best feasible result under fixed seed `42`.

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
