import time
import numpy as np
from typing import List, Dict, Any, Tuple
from sfjssp_model.instance import SFJSSPInstance
from sfjssp_model.schedule import Schedule
from baseline_solver.greedy_solvers import (
    GreedyScheduler, 
    spt_rule, 
    edt_rule, 
    min_energy_rule, 
    composite_rule
)
from utils.benchmark_generator import BenchmarkGenerator, GeneratorConfig, InstanceSize
from moea.nsga3 import NSGA3, create_sfjssp_genome, evaluate_sfjssp_genome

def random_rule(instance, schedule, ready_ops):
    return np.random.randint(0, len(ready_ops))

def verify_solvers():
    # Define scenarios
    scenarios = [
        ("SMALL", InstanceSize.SMALL, 10, 5, 5),
        ("MEDIUM", InstanceSize.MEDIUM, 30, 8, 8),
    ]
    
    # Define solvers
    solvers = [
        ("Greedy-SPT", spt_rule),
        ("Greedy-EDT", edt_rule),
        ("Greedy-Energy", min_energy_rule),
        ("Greedy-Composite", composite_rule),
        ("Greedy-Random", random_rule),
    ]
    
    for name, size_enum, n_j, n_m, n_w in scenarios:
        print(f"\n=== Testing Scenario: {name} ({n_j} jobs, {n_m} machines, {n_w} workers) ===")
        
        # Generate instance
        config = GeneratorConfig(
            instance_id=f"VERIFY_{name}",
            size=size_enum,
            n_jobs=n_j,
            n_machines=n_m,
            n_workers=n_w,
            seed=42,
            due_date_margin=(50.0, 100.0), # Generous due dates for feasibility
            ergonomic_risk_range=(0.001, 0.005)
        )
        gen = BenchmarkGenerator(config)
        instance = gen.generate()
        
        print(f"{'Solver':<20} | {'Makespan':<10} | {'Energy':<10} | {'OCRA':<6} | {'Feasible':<10} | {'Time':<6}")
        print("-" * 75)
        
        # 1. Run Greedy variants
        for s_name, rule in solvers:
            solver = GreedyScheduler(job_rule=rule)
            start_t = time.time()
            schedule = solver.schedule(instance)
            dur = time.time() - start_t
            
            metrics = schedule.evaluate(instance)
            is_feasible = schedule.check_feasibility(instance)
            violations = len(schedule.constraint_violations)
            
            f_str = "YES" if is_feasible else f"NO ({violations})"
            print(f"{s_name:<20} | {metrics['makespan']:<10.2f} | {metrics['total_energy']:<10.2f} | {metrics['max_ergonomic_exposure']:<6.2f} | {f_str:<10} | {dur:<6.2f}")

        # 2. Run NSGA-III
        print(f"  Running NSGA-III (50 generations)...")
        optimizer = NSGA3(
            n_objectives=4,
            population_size=100,
            n_generations=50,
            seed=42
        )
        optimizer.set_problem(evaluate_sfjssp_genome, create_sfjssp_genome)
        
        start_t = time.time()
        final_pop = optimizer.evolve(instance, verbose=False)
        dur = time.time() - start_t
        
        # Get best makespan individual
        best_ind = final_pop.get_best(0)
        
        # Re-evaluate for metrics
        # Note: evaluate_sfjssp_genome returns [Makespan+Pen, Energy+Pen, OCRA+Pen, Labor+Pen]
        # We need to un-penalize or just use the ind attributes if they were set
        # Actually, nsga3.py sets makespan/energy attributes
        
        f_str = "YES" if best_ind.makespan < 1e6 else "NO"
        print(f"{'NSGA-III-Best-MS':<20} | {best_ind.makespan:<10.2f} | {best_ind.energy:<10.2f} | {best_ind.ergonomic_risk:<6.2f} | {f_str:<10} | {dur:<6.2f}")

if __name__ == "__main__":
    verify_solvers()
