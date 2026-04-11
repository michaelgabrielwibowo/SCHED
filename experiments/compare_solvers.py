#!/usr/bin/env python
"""
Comparative Experiments for SFJSSP Solvers

Compares:
1. Greedy heuristics (FIFO, SPT, EDD)
2. NSGA-III (multi-objective evolutionary)
3. CP-SAT (constraint programming, if OR-Tools available)

Evidence Status:
- Experimental comparison: PROPOSED
- Metrics: Standard scheduling metrics [CONFIRMED]
"""

import os
import json
import time
from datetime import datetime

try:
    from ..sfjssp_model.instance import SFJSSPInstance
    from ..baseline_solver.greedy_solvers import (
        GreedyScheduler,
        spt_rule,
        fifo_rule,
        edt_rule,
        composite_rule,
    )
    from ..moea.nsga3 import NSGA3, create_sfjssp_genome, evaluate_sfjssp_genome
except ImportError:  # pragma: no cover - supports repo-root imports
    from sfjssp_model.instance import SFJSSPInstance
    from baseline_solver.greedy_solvers import (
        GreedyScheduler,
        spt_rule,
        fifo_rule,
        edt_rule,
        composite_rule,
    )
    from moea.nsga3 import NSGA3, create_sfjssp_genome, evaluate_sfjssp_genome


def load_benchmark(filepath: str) -> SFJSSPInstance:
    """Load a benchmark instance from its stored JSON representation."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SFJSSPInstance.from_dict(data)


def run_greedy_experiment(instance: SFJSSPInstance, rule_name: str, rule_fn) -> dict:
    """Run greedy scheduler experiment"""
    start_time = time.time()

    scheduler = GreedyScheduler(job_rule=rule_fn)
    schedule = scheduler.schedule(instance, verbose=True)

    elapsed = time.time() - start_time

    # Evaluate
    objectives = schedule.evaluate(instance)
    
    if not schedule.is_feasible and schedule.constraint_violations:
        print(f"    (Infeasible: {len(schedule.constraint_violations)} violations, first: {schedule.constraint_violations[0]})")
        if any("overlap" in v for v in schedule.constraint_violations):
            print(f"    WARNING: Overlap violation detected!")

    return {
        'method': f'Greedy ({rule_name})',
        'makespan': schedule.makespan,
        'energy': objectives.get('total_energy', 0),
        'ergonomic_risk': objectives.get('max_ergonomic_exposure', 0),
        'labor_cost': objectives.get('total_labor_cost', 0),
        'time_seconds': elapsed,
        'feasible': schedule.is_feasible,
    }


def run_nsga3_experiment(instance: SFJSSPInstance, n_generations: int = 50) -> dict:
    """Run NSGA-III experiment"""
    start_time = time.time()

    nsga3 = NSGA3(
        n_objectives=4,
        population_size=30,
        n_generations=n_generations,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
    )

    nsga3.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        create_individual_fn=create_sfjssp_genome,
    )

    nsga3.evolve(instance, verbose=False)

    elapsed = time.time() - start_time

    # Get best solutions for each objective
    pareto = nsga3.get_pareto_solutions()
    feasible_pareto = [
        sol for sol in pareto
        if sol.objectives and all(obj < 1e8 for obj in sol.objectives)
    ]

    if feasible_pareto:
        best_makespan = min(s.makespan for s in feasible_pareto)
        best_energy = min(s.energy for s in feasible_pareto)
        best_ergonomic = min(s.ergonomic_risk for s in feasible_pareto)
        best_labor = min(s.labor_cost for s in feasible_pareto)
    else:
        best_makespan = best_energy = best_ergonomic = best_labor = float('inf')

    return {
        'method': f'NSGA-III ({n_generations} gen)',
        'makespan': best_makespan,
        'energy': best_energy,
        'ergonomic_risk': best_ergonomic,
        'labor_cost': best_labor,
        'time_seconds': elapsed,
        'pareto_size': len(pareto),
        'feasible_pareto_size': len(feasible_pareto),
        'feasible': bool(feasible_pareto),
    }


def run_cp_experiment(instance: SFJSSPInstance, time_limit: int = 30) -> dict:
    """Run CP-SAT experiment"""
    try:
        from ..exact_solvers.cp_solver import CPScheduler
    except ImportError:  # pragma: no cover - supports repo-root imports
        try:
            from exact_solvers.cp_solver import CPScheduler
        except ImportError:
            return {
                'method': 'CP-SAT',
                'error': 'OR-Tools not available',
                'feasible': False,
            }
    except Exception as exc:
        return {
            'method': 'CP-SAT',
            'error': str(exc),
            'feasible': False,
        }

    start_time = time.time()

    try:
        cp_solver = CPScheduler(time_limit=time_limit, num_workers=2)
        schedule = cp_solver.solve(instance, objective='makespan', verbose=False)
    except ImportError as exc:
        return {
            'method': 'CP-SAT',
            'error': str(exc),
            'feasible': False,
        }

    elapsed = time.time() - start_time

    if schedule is None:
        return {
            'method': 'CP-SAT',
            'error': 'No solution found',
            'time_seconds': elapsed,
            'feasible': False,
        }

    objectives = schedule.evaluate(instance)

    return {
        'method': 'CP-SAT',
        'makespan': schedule.makespan,
        'energy': objectives.get('total_energy', 0),
        'ergonomic_risk': objectives.get('max_ergonomic_exposure', 0),
        'labor_cost': objectives.get('total_labor_cost', 0),
        'time_seconds': elapsed,
        'feasible': schedule.is_feasible,
    }


def run_comparison(
    instance: SFJSSPInstance,
    instance_name: str,
    run_cp: bool = False,
    nsga3_generations: int = 50,
) -> dict:
    """Run full comparison on instance"""
    print(f"\n{'='*60}")
    print(f"Instance: {instance_name}")
    print(f"Jobs: {instance.n_jobs}, Machines: {instance.n_machines}, Workers: {instance.n_workers}")
    print(f"{'='*60}")

    results = {
        'instance': instance_name,
        'n_jobs': instance.n_jobs,
        'n_machines': instance.n_machines,
        'n_workers': instance.n_workers,
        'timestamp': datetime.now().isoformat(),
        'experiments': [],
    }

    # Greedy methods
    print("\nRunning greedy heuristics...")

    for rule_name, rule_fn in [('SPT', spt_rule), ('FIFO', fifo_rule), ('EDD', edt_rule)]:
        print(f"  {rule_name}...", end=' ', flush=True)
        result = run_greedy_experiment(instance, rule_name, rule_fn)
        results['experiments'].append(result)
        print(f"makespan={result['makespan']:.1f}, time={result['time_seconds']:.3f}s")

    # NSGA-III
    print(f"\nRunning NSGA-III ({nsga3_generations} generations)...")
    result = run_nsga3_experiment(instance, nsga3_generations)
    results['experiments'].append(result)
    print(f"  makespan={result['makespan']:.1f}, energy={result['energy']:.1f}, time={result['time_seconds']:.1f}s")

    # CP-SAT (optional)
    if run_cp:
        print("\nRunning CP-SAT...")
        result = run_cp_experiment(instance, time_limit=60)
        results['experiments'].append(result)
        if 'error' in result:
            print(f"  {result['error']}")
        else:
            print(f"  makespan={result['makespan']:.1f}, time={result['time_seconds']:.1f}s")

    return results


def run_suite_comparison(
    benchmark_dir: str = "benchmarks/small",
    output_path: str = "experiments/results/comparison_results.json",
    run_cp: bool = False,
    nsga3_generations: int = 30,
):
    """Run comparison on all instances in directory"""
    import glob

    print("=" * 60)
    print("SFJSSP Solver Comparison Experiment")
    print("=" * 60)

    # Find all benchmark files
    pattern = os.path.join(benchmark_dir, "*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No benchmark files found in {benchmark_dir}")
        return

    print(f"Found {len(files)} benchmark files")

    all_results = []

    for filepath in files:
        instance_name = os.path.basename(filepath).replace('.json', '')
        instance = load_benchmark(filepath)

        results = run_comparison(
            instance,
            instance_name,
            run_cp=run_cp,
            nsga3_generations=nsga3_generations,
        )
        all_results.append(results)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")

    # Print summary
    print_summary(all_results)

    return all_results


def print_summary(results: list):
    """Print summary table"""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Instance':<25} {'Method':<20} {'Makespan':>12} {'Energy':>12} {'Time(s)':>10}")
    print("-" * 80)

    for r in results:
        instance = r['instance']
        for exp in r['experiments']:
            method = exp['method']
            makespan = f"{exp['makespan']:.1f}" if 'makespan' in exp else "N/A"
            energy = f"{exp['energy']:.0f}" if 'energy' in exp else "N/A"
            time_s = f"{exp['time_seconds']:.2f}" if 'time_seconds' in exp else "N/A"

            print(f"{instance:<25} {method:<20} {makespan:>12} {energy:>12} {time_s:>10}")

    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFJSSP Solver Comparison")
    parser.add_argument("--benchmark-dir", type=str, default="benchmarks/small",
                       help="Directory with benchmark JSON files")
    parser.add_argument("--output", type=str, default="experiments/results/comparison.json",
                       help="Output JSON path")
    parser.add_argument("--cp", action="store_true", help="Run CP-SAT solver")
    parser.add_argument("--generations", type=int, default=30,
                       help="NSGA-III generations")

    args = parser.parse_args()

    run_suite_comparison(
        benchmark_dir=args.benchmark_dir,
        output_path=args.output,
        run_cp=args.cp,
        nsga3_generations=args.generations,
    )
