#!/usr/bin/env python
"""
Visualize experiment results

Creates comparison plots from solver comparison results.
"""

import os
import json
from typing import Any, Dict, List, Optional
import math


def load_results(filepath: str) -> Any:
    """Load results from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def _extract_results(payload: Any) -> List[Dict[str, Any]]:
    """Accept both the legacy list payload and the v2 dict payload."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("results", [])
    raise TypeError(f"Unsupported results payload type: {type(payload).__name__}")


def _extract_provenance(payload: Any) -> Optional[Dict[str, Any]]:
    """Return provenance metadata when present."""
    if isinstance(payload, dict):
        return payload.get("provenance")
    return None


def _metric_or_nan(value: Any) -> float:
    """Convert optional JSON metric values into plot-friendly floats."""
    if value is None:
        return math.nan
    return float(value)


def _get_nsga_report_policy(
    exp: Dict[str, Any],
    provenance: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return the report-member policy name for an NSGA experiment row."""
    return exp.get("report_member_key") or (provenance or {}).get("nsga3_report_member_policy")


def _get_plot_method_label(
    exp: Dict[str, Any],
    provenance: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the display label used for charts and summaries."""
    method = exp["method"]
    if not method.startswith("NSGA-III"):
        return method
    report_policy = _get_nsga_report_policy(exp, provenance=provenance)
    if report_policy:
        return f"{method} [{report_policy}]"
    return method


def _get_experiment_metric(
    exp: Dict[str, Any],
    metric: str,
) -> Any:
    """Return plot-ready metrics, preferring explicit report-member fields for NSGA."""
    if exp["method"].startswith("NSGA-III"):
        report_metrics = exp.get("report_member_metrics") or {}
        if metric == "makespan":
            return report_metrics.get("makespan", exp.get("makespan"))
        if metric == "energy":
            return report_metrics.get("total_energy", exp.get("energy"))
        if metric == "weighted_tardiness":
            return exp.get("report_weighted_tardiness", exp.get("selected_weighted_tardiness"))
        if metric == "n_tardy_jobs":
            return exp.get("report_n_tardy_jobs", exp.get("selected_n_tardy_jobs"))
    return exp.get(metric)


def plot_comparison(results: Any, output_dir: str = "experiments/results/plots"):
    """Create comparison plots"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib required. Install: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)
    results_list = _extract_results(results)
    provenance = _extract_provenance(results)

    if not results_list:
        print("No comparison results available to plot.")
        return

    # Extract data
    instances = [r['instance'] for r in results_list]
    
    makespan_data = {}
    time_data = {}
    tardiness_data = {}

    for r in results_list:
        for exp in r['experiments']:
            method = _get_plot_method_label(exp, provenance=provenance)
            if method not in makespan_data:
                makespan_data[method] = []
                time_data[method] = []
                tardiness_data[method] = []
            
            makespan_data[method].append(_metric_or_nan(_get_experiment_metric(exp, 'makespan')))
            time_data[method].append(_metric_or_nan(exp.get('time_seconds', 0)))
            tardiness_data[method].append(_metric_or_nan(_get_experiment_metric(exp, 'weighted_tardiness')))

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

    # Plot 2: Weighted tardiness comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (method, tardiness_values) in enumerate(tardiness_data.items()):
        ax.bar([j + i*width for j in x], tardiness_values, width, label=method)

    ax.set_xlabel('Instance')
    ax.set_ylabel('Weighted Tardiness')
    ax.set_title('SFJSSP Solver Comparison - Weighted Tardiness')
    ax.set_xticks([j + width*1.5 for j in x])
    ax.set_xticklabels(instances, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'tardiness_comparison.png'), dpi=150)
    print(f"Saved: {output_dir}/tardiness_comparison.png")

    # Plot 3: Time comparison (log scale)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    methods = list(time_data.keys())
    avg_times = []
    for times in time_data.values():
        valid = [time for time in times if not math.isnan(time)]
        avg_times.append(sum(valid) / len(valid) if valid else math.nan)
    
    ax.bar(methods, avg_times)
    ax.set_ylabel('Average Time (seconds)')
    ax.set_title('SFJSSP Solver Comparison - Runtime')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=150)
    print(f"Saved: {output_dir}/time_comparison.png")

    # Plot 4: Summary statistics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Makespan by method
    makespan_means = []
    makespan_stds = []
    for makespans in makespan_data.values():
        valid = [value for value in makespans if not math.isnan(value)]
        if valid:
            makespan_means.append(sum(valid) / len(valid))
            makespan_stds.append(max(valid) - min(valid))
        else:
            makespan_means.append(math.nan)
            makespan_stds.append(0.0)
    
    axes[0].bar(methods, makespan_means, yerr=makespan_stds, capsize=5)
    axes[0].set_ylabel('Average Makespan')
    axes[0].set_title('Average Makespan by Method')
    axes[0].tick_params(axis='x', rotation=45)

    tardiness_means = []
    for tardiness_values in tardiness_data.values():
        valid = [value for value in tardiness_values if not math.isnan(value)]
        tardiness_means.append(sum(valid) / len(valid) if valid else math.nan)

    axes[1].bar(methods, tardiness_means)
    axes[1].set_ylabel('Average Weighted Tardiness')
    axes[1].set_title('Average Weighted Tardiness by Method')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Energy by method (if available)
    energy_data = {}
    for r in results_list:
        for exp in r['experiments']:
            method = _get_plot_method_label(exp, provenance=provenance)
            if method not in energy_data:
                energy_data[method] = []
            energy_data[method].append(_metric_or_nan(_get_experiment_metric(exp, 'energy')))
    
    if energy_data:
        energy_means = []
        for values in energy_data.values():
            valid = [value for value in values if not math.isnan(value)]
            energy_means.append(sum(valid) / len(valid) if valid else math.nan)
        axes[2].bar(list(energy_data.keys()), energy_means)
        axes[2].set_ylabel('Average Energy (kWh)')
        axes[2].set_title('Average Energy by Method')
        axes[2].tick_params(axis='x', rotation=45)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'summary_stats.png'), dpi=150)
    print(f"Saved: {output_dir}/summary_stats.png")

    print(f"\nAll plots saved to {output_dir}")


def print_best_results(results: Any):
    """Print best result per instance"""
    results_list = _extract_results(results)
    provenance = _extract_provenance(results)
    print("\n" + "=" * 70)
    print("BEST MAKESPAN PER INSTANCE")
    print("=" * 70)
    
    for r in results_list:
        instance = r['instance']
        valid_experiments = [
            exp for exp in r['experiments']
            if _get_experiment_metric(exp, 'makespan') is not None
        ]
        if not valid_experiments:
            print(f"{instance}: no finite makespan result in artifact")
            continue
        best = min(
            valid_experiments,
            key=lambda x: _get_experiment_metric(x, 'makespan'),
        )
        label = _get_plot_method_label(best, provenance=provenance)
        best_makespan = _get_experiment_metric(best, 'makespan')
        print(f"{instance}: {label} ({best_makespan:.1f})")

    print("\n" + "=" * 70)
    print("NSGA REPORT MEMBERS")
    print("=" * 70)
    for r in results_list:
        instance = r['instance']
        nsga = next((exp for exp in r['experiments'] if exp['method'].startswith('NSGA-III')), None)
        if nsga is None:
            continue
        label = _get_plot_method_label(nsga, provenance=provenance)
        report_makespan = _get_experiment_metric(nsga, 'makespan')
        report_weighted_tardiness = _get_experiment_metric(nsga, 'weighted_tardiness')
        print(
            f"{instance}: {label} (makespan={report_makespan:.1f}, "
            f"weighted_tardiness={_metric_or_nan(report_weighted_tardiness):.1f})"
        )
        if (
            nsga.get('tardiness_best_weighted_tardiness') is not None
            and nsga.get('report_weighted_tardiness') is not None
            and nsga['tardiness_best_weighted_tardiness'] < nsga['report_weighted_tardiness']
        ):
            print(
                f"  tardiness-best feasible member: makespan={nsga['tardiness_best_makespan']:.1f}, "
                f"weighted_tardiness={nsga['tardiness_best_weighted_tardiness']:.1f}"
            )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize SFJSSP experiment results")
    parser.add_argument("--results", type=str, default="experiments/results/comparison.json",
                       help="Path to results JSON file")
    parser.add_argument("--output", type=str, default="experiments/results/plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    payload = load_results(args.results)
    provenance = _extract_provenance(payload)
    if provenance:
        print("Loaded comparison artifact with provenance:")
        print(
            f"  commit={provenance.get('git_commit')}, "
            f"benchmark_dir={provenance.get('benchmark_dir')}, "
            f"generations={provenance.get('nsga3_generations')}, "
            f"report_policy={provenance.get('nsga3_report_member_policy')}, "
            f"cp_enabled={provenance.get('cp_enabled')}"
        )

    plot_comparison(payload, args.output)
    print_best_results(payload)
