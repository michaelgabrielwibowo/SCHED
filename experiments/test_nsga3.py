#!/usr/bin/env python
"""
Test NSGA-III on SFJSSP Example

Evidence Status: NSGA-III algorithm CONFIRMED, application to SFJSSP PROPOSED.
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from sfjssp_model.instance import SFJSSPInstance
from sfjssp_model.machine import Machine
from sfjssp_model.worker import Worker
from sfjssp_model.job import Job, Operation

from moea.nsga3 import NSGA3, create_sfjssp_genome, evaluate_sfjssp_genome


def create_test_instance():
    """Create a small test instance"""
    instance = SFJSSPInstance(instance_id="NSGA3_TEST")

    # Add machines
    for i in range(3):
        m = Machine(
            machine_id=i,
            power_processing=20.0 + i * 10,
            power_idle=3.0 + i,
        )
        instance.add_machine(m)

    # Add workers
    for i in range(2):
        w = Worker(
            worker_id=i,
            labor_cost_per_hour=15.0 + i * 5,
            fatigue_rate=0.02 + i * 0.01,
        )
        instance.add_worker(w)

    # Add jobs - IMPORTANT: must set ergonomic_risk_map for each operation
    for job_id in range(4):
        ops = []
        for op_idx in range(2):
            op = Operation(
                job_id=job_id,
                op_id=op_idx,
                eligible_machines={0, 1, 2},
                eligible_workers={0, 1},
            )
            op.processing_times = {
                0: {0: 20.0 + op_idx * 5},
                1: {0: 25.0 + op_idx * 5},
                2: {0: 22.0 + op_idx * 5},
            }
            ops.append(op)
            # Set ergonomic risk for this operation
            instance.ergonomic_risk_map[(job_id, op_idx)] = 0.3

        job = Job(
            job_id=job_id,
            operations=ops,
            due_date=150.0,
            weight=1.0,
        )
        instance.add_job(job)

    return instance


def main():
    print("=" * 60)
    print("NSGA-III Test on SFJSSP")
    print("=" * 60)

    # Create test instance
    instance = create_test_instance()
    print(f"\nInstance: {instance.n_jobs} jobs, {instance.n_machines} machines, {instance.n_workers} workers")
    print(f"Total operations: {instance.n_operations}")

    # Setup NSGA-III
    nsga3 = NSGA3(
        n_objectives=4,
        population_size=20,
        n_generations=30,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
    )

    # Set problem functions
    nsga3.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        create_individual_fn=create_sfjssp_genome,
    )

    # Run optimization
    print("\nRunning NSGA-III...")
    nsga3.evolve(instance, verbose=True)

    # Report results
    print("\n" + "=" * 60)
    print("Pareto Front Solutions:")
    print("=" * 60)

    pareto = nsga3.get_pareto_solutions()
    for i, sol in enumerate(pareto[:5]):  # Show top 5
        print(f"\nSolution {i+1}:")
        print(f"  Makespan: {sol.makespan:.1f}")
        print(f"  Energy: {sol.energy:.1f} kWh")
        print(f"  Ergonomic Risk: {sol.ergonomic_risk:.2f}")
        print(f"  Labor Cost: ${sol.labor_cost:.1f}")

    print(f"\nTotal Pareto solutions: {len(pareto)}")

    # Compare with random baseline
    print("\n" + "=" * 60)
    print("Random Baseline Comparison:")
    print("=" * 60)

    import numpy as np
    rng = np.random.default_rng(123)

    random_objectives = []
    for _ in range(20):
        genome = create_sfjssp_genome(instance, rng)
        obj = evaluate_sfjssp_genome(instance, genome)
        random_objectives.append(obj)

    random_objectives = np.array(random_objectives)
    print(f"Random makespan: {random_objectives[:, 0].mean():.1f} (min: {random_objectives[:, 0].min():.1f})")
    print(f"Random energy: {random_objectives[:, 1].mean():.1f} (min: {random_objectives[:, 1].min():.1f})")

    if pareto:
        print(f"NSGA-III makespan: {min(s.makespan for s in pareto):.1f}")
        print(f"NSGA-III energy: {min(s.energy for s in pareto):.1f}")


if __name__ == "__main__":
    main()
