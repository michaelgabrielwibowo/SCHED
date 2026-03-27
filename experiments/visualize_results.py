#!/usr/bin/env python
"""
Visualize experiment results

Creates comparison plots from solver comparison results.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(filepath: str) -> list:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_comparison(results: list, output_dir: str = "experiments/results/plots"):
    """Create comparison plots"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib required. Install: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    instances = [r['instance'] for r in results]
    
    makespan_data = {}
    time_data = {}

    for r in results:
        for exp in r['experiments']:
            method = exp['method']
            if method not in makespan_data:
                makespan_data[method] = []
                time_data[method] = []
            
            makespan_data[method].append(exp.get('makespan', 0))
            time_data[method].append(exp.get('time_seconds', 0))

    # Plot 1: Makespan comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(instances))
    width = 0.2
    
    for i, (method, makespans) in enumerate(makespan_data.items()):
        ax.bar([j + i*width for j in x], makespans, width, label=method)
    
    ax.set_xlabel('Instance')
    ax.set_ylabel('Makespan')
    ax.set_title('SFJSSP Solver Comparison - Makespan')
    ax.set_xticks([j + width*1.5 for j in x])
    ax.set_xticklabels(instances, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'makespan_comparison.png'), dpi=150)
    print(f"Saved: {output_dir}/makespan_comparison.png")

    # Plot 2: Time comparison (log scale)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    methods = list(time_data.keys())
    avg_times = [sum(times)/len(times) for times in time_data.values()]
    
    ax.bar(methods, avg_times)
    ax.set_ylabel('Average Time (seconds)')
    ax.set_title('SFJSSP Solver Comparison - Runtime')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=150)
    print(f"Saved: {output_dir}/time_comparison.png")

    # Plot 3: Summary statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Makespan by method
    makespan_means = [sum(m)/len(m) for m in makespan_data.values()]
    makespan_stds = [max(m) - min(m) for m in makespan_data.values()]
    
    axes[0].bar(methods, makespan_means, yerr=makespan_stds, capsize=5)
    axes[0].set_ylabel('Average Makespan')
    axes[0].set_title('Average Makespan by Method')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Energy by method (if available)
    energy_data = {}
    for r in results:
        for exp in r['experiments']:
            method = exp['method']
            if method not in energy_data:
                energy_data[method] = []
            energy_data[method].append(exp.get('energy', 0))
    
    if energy_data:
        energy_means = [sum(e)/len(e) for e in energy_data.values()]
        axes[1].bar(list(energy_data.keys()), energy_means)
        axes[1].set_ylabel('Average Energy (kWh)')
        axes[1].set_title('Average Energy by Method')
        axes[1].tick_params(axis='x', rotation=45)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'summary_stats.png'), dpi=150)
    print(f"Saved: {output_dir}/summary_stats.png")

    print(f"\nAll plots saved to {output_dir}")


def print_best_results(results: list):
    """Print best result per instance"""
    print("\n" + "=" * 70)
    print("BEST MAKESPAN PER INSTANCE")
    print("=" * 70)
    
    for r in results:
        instance = r['instance']
        best = min(r['experiments'], key=lambda x: x.get('makespan', float('inf')))
        print(f"{instance}: {best['method']} ({best['makespan']:.1f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize SFJSSP experiment results")
    parser.add_argument("--results", type=str, default="experiments/results/comparison.json",
                       help="Path to results JSON file")
    parser.add_argument("--output", type=str, default="experiments/results/plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    results = load_results(args.results)
    plot_comparison(results, args.output)
    print_best_results(results)
