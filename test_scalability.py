import time
import json
from sfjssp_model.instance import SFJSSPInstance
from baseline_solver.greedy_solvers import GreedyScheduler

def test_scalability():
    # Load one of the large instances
    print("Loading large benchmark instance...")
    start_time = time.time()
    
    # We will just generate it freshly in memory to avoid writing a full JSON loader right now.
    from utils.benchmark_generator import BenchmarkGenerator, GeneratorConfig, InstanceSize
    config = GeneratorConfig(
        instance_id="SFJSSP_LARGE_TEST", 
        size=InstanceSize.LARGE, 
        n_jobs=200, 
        n_machines=20, 
        n_workers=20, 
        seed=100
    )
    gen = BenchmarkGenerator(config)
    instance = gen.generate()
    
    load_time = time.time() - start_time
    print(f"Instance generated in {load_time:.2f} seconds.")
    print(f"Total Operations: {instance.n_operations}")
    
    # Solve with greedy solver
    print("\nSolving with GreedyScheduler (makespan objective)...")
    solver = GreedyScheduler()
    
    solve_start = time.time()
    schedule = solver.schedule(instance)
    solve_time = time.time() - solve_start
    
    print(f"Solved in {solve_time:.4f} seconds.")
    print(f"Makespan: {schedule.makespan:.2f}")
    
    metrics = schedule.evaluate(instance)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    test_scalability()