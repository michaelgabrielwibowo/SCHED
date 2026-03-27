
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from sfjssp_model.instance import SFJSSPInstance
from utils.benchmark_generator import BenchmarkGenerator, GeneratorConfig, InstanceSize
from moea.nsga3 import NSGA3, evaluate_sfjssp_genome, create_sfjssp_genome

def run_optimization():
    print("=== SFJSSP Multi-Objective Optimization (NSGA-III) ===")
    
    # 1. Generate or load a medium instance
    if not os.path.exists("benchmarks"):
        os.makedirs("benchmarks")
        
    print("Generating medium instance (50 jobs, 10 machines, 10 workers)...")
    config = GeneratorConfig(
        instance_id="MEDIUM_DEMO",
        size=InstanceSize.MEDIUM,
        n_jobs=50,
        n_machines=10,
        n_workers=10,
        seed=42
    )
    generator = BenchmarkGenerator(config)
    instance = generator.generate()
    
    # 2. Configure NSGA-III
    # We optimize 4 objectives: [Makespan, Energy, OCRA Risk, Labor Cost]
    optimizer = NSGA3(
        n_objectives=4,
        population_size=50,
        n_generations=30,
        mutation_rate=0.1,
        seed=42
    )
    
    optimizer.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        create_individual_fn=create_sfjssp_genome
    )
    
    # 3. Evolve
    print(f"\nStarting evolution for {optimizer.n_generations} generations...")
    start_time = time.time()
    final_pop = optimizer.evolve(instance, verbose=True)
    duration = time.time() - start_time
    
    print(f"\nOptimization complete in {duration:.2f} seconds.")
    
    # 4. Extract Pareto Solutions
    pareto_solutions = optimizer.get_pareto_solutions()
    print(f"Found {len(pareto_solutions)} Pareto-optimal solutions.")
    
    # 5. Visualization
    plot_pareto_front(pareto_solutions)
    
def plot_pareto_front(solutions):
    """Plot Makespan vs Energy vs OCRA Risk"""
    makespans = [ind.objectives[0] for ind in solutions]
    energies = [ind.objectives[1] for ind in solutions]
    ocra_risks = [ind.objectives[2] for ind in solutions]
    labor_costs = [ind.objectives[3] for ind in solutions]
    
    # Filter out penalized solutions for cleaner plotting if any
    valid_indices = [i for i, m in enumerate(makespans) if m < 1e6]
    if not valid_indices:
        print("Warning: All solutions in final Pareto front are invalid (penalized).")
        return
        
    m_valid = [makespans[i] for i in valid_indices]
    e_valid = [energies[i] for i in valid_indices]
    o_valid = [ocra_risks[i] for i in valid_indices]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(m_valid, e_valid, o_valid, c=o_valid, cmap='viridis', s=50)
    
    ax.set_xlabel('Makespan (min)')
    ax.set_ylabel('Total Energy (kWh)')
    ax.set_zlabel('Max OCRA Risk')
    ax.set_title('SFJSSP Pareto Frontier (NSGA-III)')
    
    plt.colorbar(scatter, label='OCRA Risk')
    
    output_plot = "plots/pareto_frontier.png"
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig(output_plot)
    print(f"Pareto plot saved to {output_plot}")
    plt.close()

if __name__ == "__main__":
    run_optimization()
