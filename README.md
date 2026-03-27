# SFJSSP Implementation

Sustainable Flexible Job-Shop Scheduling Problem (SFJSSP) implementation for Industry 5.0 research.

## Overview

This codebase implements scheduling optimization for manufacturing systems considering:
- **Economic objectives**: Makespan, tardiness
- **Environmental objectives**: Energy consumption, carbon emissions
- **Human objectives**: Ergonomic risk, worker fatigue, labor cost
- **Resilience**: Recovery from disruptions (dynamic events)

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

### Basic (for model + heuristics)

```bash
cd sfjssp_code
pip install numpy
```

### Full (with DRL)

```bash
cd sfjssp_code
pip install numpy torch gymnasium
```

### Development

```bash
pip install numpy torch gymnasium pytest
```

## Quick Start

### 1. Generate Benchmark Instance

```bash
cd sfjssp_code
python experiments/generate_benchmarks.py --mode example --output benchmarks
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
python experiments/test_nsga3.py
```

### 4. Train DRL Agent (requires PyTorch)

```bash
python training/train_drl.py --episodes 100
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
from environment import SFJSSPEnv

env = SFJSSPEnv(instance)
obs, info = env.reset(seed=42)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Baseline Solvers

```python
from baseline_solver import (
    GreedyScheduler,
    spt_rule,      # Shortest Processing Time
    fifo_rule,     # First In First Out
    edt_rule,      # Earliest Due Date
    composite_rule, # Multi-objective composite
)

scheduler = GreedyScheduler(job_rule=spt_rule)
schedule = scheduler.schedule(instance)
```

### NSGA-III

```python
from moea import NSGA3, create_sfjssp_genome, evaluate_sfjssp_genome

nsga3 = NSGA3(
    n_objectives=4,
    population_size=100,
    n_generations=100,
)

nsga3.set_problem(
    evaluate_fn=evaluate_sfjssp_genome,
    create_individual_fn=create_sfjssp_genome,
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
6. **Resilience** (f6): Minimize recovery time after disruptions

## Constraints

### Hard Constraints
- Operation precedence
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

## Related Documentation

- `PROJECT_PLAN_SFJSSP.md` - Implementation roadmap
- `RESEARCH_PROPOSAL_SFJSSP.md` - Research proposal
- `MATHEMATICAL_MODEL_SFJSSP.md` - MIP formulation
- `DATASET_INVENTORY_SFJSSP.md` - Dataset catalog

## Testing

```bash
cd sfjssp_code
python debug_test.py  # Quick smoke test
python experiments/test_nsga3.py  # NSGA-III test
```

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
4. Resilience metric validation
