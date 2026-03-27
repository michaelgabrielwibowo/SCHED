"""
Gantt Chart Visualization for SFJSSP Schedules

Evidence Status:
- Gantt charts: Standard scheduling visualization [CONFIRMED]
- Application to SFJSSP: PROPOSED
"""

from typing import Dict, List, Optional, Any
import json


def plot_gantt(
    schedule: Any,
    instance: Any,
    title: str = "SFJSSP Schedule",
    show_workers: bool = True,
    figsize: tuple = (14, 8),
) -> 'plt.Figure':
    """
    Create Gantt chart of schedule

    Args:
        schedule: Schedule object
        instance: SFJSSPInstance
        title: Chart title
        show_workers: Show worker assignments
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Matplotlib required. Install: pip install matplotlib")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Colors for jobs
    n_jobs = instance.n_jobs
    colors = plt.cm.tab20.colors[:max(1, n_jobs)]

    # Plot machine schedules
    y_positions = {}
    for i, machine in enumerate(instance.machines):
        y_positions[machine.machine_id] = i

    height = 0.8

    for machine_id, machine_sched in schedule.machine_schedules.items():
        y = y_positions.get(machine_id, 0)

        for op in machine_sched.operations:
            job_color = colors[op.job_id % len(colors)]

            # Draw bar
            rect = mpatches.Rectangle(
                (op.start_time, y - height/2),
                op.completion_time - op.start_time,
                height,
                facecolor=job_color,
                edgecolor='black',
                linewidth=0.5,
            )
            ax.add_patch(rect)

            # Add label
            if op.completion_time - op.start_time > 10:  # Only if bar is wide enough
                ax.text(
                    op.start_time + (op.completion_time - op.start_time) / 2,
                    y,
                    f'J{op.job_id}.O{op.op_id}',
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black' if op.job_id % 2 == 0 else 'white',
                )

        # Machine label
        ax.text(-5, y, f'M{machine_id}', ha='right', va='center', fontsize=10)

    # Configure axes
    ax.set_ylim(-1, len(instance.machines))
    ax.set_xlim(0, schedule.makespan * 1.05)
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title(f'{title}\nMakespan: {schedule.makespan:.1f}')

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f'M{m}' for m in sorted(y_positions.keys())])

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_schedule(
    schedule: Any,
    instance: Any,
    output_path: Optional[str] = None,
    format: str = 'png',
) -> str:
    """
    Plot schedule and optionally save to file

    Args:
        schedule: Schedule object
        instance: SFJSSPInstance
        output_path: Path to save figure
        format: Image format ('png', 'svg', 'pdf')

    Returns:
        Path to saved file or None
    """
    fig = plot_gantt(schedule, instance)

    if fig is None:
        return None

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved Gantt chart to {output_path}")

    return output_path


def save_gantt_html(
    schedule: Any,
    instance: Any,
    output_path: str,
    title: str = "SFJSSP Schedule",
) -> str:
    """
    Create interactive HTML Gantt chart

    Uses Plotly for interactivity.

    Args:
        schedule: Schedule object
        instance: SFJSSPInstance
        output_path: Path to save HTML
        title: Chart title

    Returns:
        Path to saved file
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly required. Install: pip install plotly")
        # Fallback to simple HTML
        return _save_simple_html(schedule, instance, output_path, title)

    # Build data for Plotly
    tasks = []

    for machine_id, machine_sched in schedule.machine_schedules.items():
        for op in machine_sched.operations:
            tasks.append({
                'machine': f'M{machine_id}',
                'job': f'J{op.job_id}.O{op.op_id}',
                'start': op.start_time,
                'end': op.completion_time,
                'worker': f'W{op.worker_id}',
                'duration': op.processing_time,
            })

    # Create figure
    fig = go.Figure()

    # Add bars for each machine
    machines = sorted(set(t['machine'] for t in tasks))

    for i, machine in enumerate(machines):
        machine_tasks = [t for t in tasks if t['machine'] == machine]

        fig.add_trace(go.Bar(
            name=machine,
            y=[machine] * len(machine_tasks),
            x=[t['end'] - t['start'] for t in machine_tasks],
            base=[t['start'] for t in machine_tasks],
            orientation='h',
            text=[t['job'] for t in machine_tasks],
            textposition='inside',
            hoverinfo='text',
            hovertext=[
                f"{t['job']}<br>Start: {t['start']:.1f}<br>End: {t['end']:.1f}<br>"
                f"Duration: {t['duration']:.1f}<br>Worker: {t['worker']}"
                for t in machine_tasks
            ],
        ))

    fig.update_layout(
        title=f'{title}<br>Makespan: {schedule.makespan:.1f}',
        xaxis_title='Time',
        yaxis_title='Machine',
        barmode='stack',
        height=400 + len(machines) * 40,
    )

    # Save to HTML
    fig.write_html(output_path)
    print(f"Saved interactive Gantt chart to {output_path}")

    return output_path


def _save_simple_html(
    schedule: Any,
    instance: Any,
    output_path: str,
    title: str,
) -> str:
    """Save simple HTML Gantt chart without Plotly"""

    # Build schedule data
    gantt_data = schedule.to_gantt_dict()

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .info {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .machine {{ margin-bottom: 15px; }}
        .machine-label {{ font-weight: bold; margin-bottom: 5px; }}
        .timeline {{ position: relative; height: 30px; background: #eee; border-radius: 3px; }}
        .bar {{ position: absolute; height: 26px; background: #4CAF50; border: 1px solid #333;
                 border-radius: 3px; color: white; font-size: 11px; line-height: 26px;
                 padding: 0 5px; overflow: hidden; white-space: nowrap; }}
        .legend {{ margin-top: 20px; }}
        .legend-item {{ display: inline-block; margin-right: 15px; }}
        .legend-color {{ display: inline-block; width: 20px; height: 20px; 
                         vertical-align: middle; margin-right: 5px; border: 1px solid #333; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <div class="info">
        <strong>Makespan:</strong> {schedule.makespan:.1f}<br>
        <strong>Jobs:</strong> {instance.n_jobs} | 
        <strong>Machines:</strong> {instance.n_machines} | 
        <strong>Workers:</strong> {instance.n_workers}<br>
        <strong>Operations:</strong> {instance.n_operations}
    </div>

    <div class="gantt">
"""

    # Colors for jobs
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0',
              '#00BCD4', '#8BC34A', '#FFC107', '#3F51B5', '#F44336']

    for machine_id in sorted(gantt_data['machine_schedules'].keys()):
        ops = gantt_data['machine_schedules'][machine_id]

        html += f"""
        <div class="machine">
            <div class="machine-label">Machine {machine_id}</div>
            <div class="timeline" style="width: {min(800, schedule.makespan / 2)}px;">
"""

        for op in ops:
            color = colors[op['job_id'] % len(colors)]
            left_pct = (op['start'] / schedule.makespan) * 100
            width_pct = ((op['end'] - op['start']) / schedule.makespan) * 100

            html += f"""
                <div class="bar" style="left: {left_pct}%; width: {max(2, width_pct)}%; 
                     background: {color};" 
                     title="Job {op['job_id']}, Op {op['op_id']}
Worker: {op['worker_id']}
Start: {op['start']:.1f}, End: {op['end']:.1f}">
                    J{op['job_id']}.O{op['op_id']}
                </div>
"""

        html += """
            </div>
        </div>
"""

    # Legend
    html += """
    <div class="legend">
        <strong>Job Colors:</strong><br>
"""

    for i in range(min(instance.n_jobs, 10)):
        html += f"""
        <span class="legend-item">
            <span class="legend-color" style="background: {colors[i % len(colors)]};"></span>
            Job {i}
        </span>
"""

    html += """
    </div>
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Saved HTML Gantt chart to {output_path}")
    return output_path


def print_schedule_summary(schedule: Any, instance: Any) -> str:
    """
    Print text summary of schedule

    Returns:
        Summary string
    """
    lines = [
        "=" * 60,
        f"SFJSSP Schedule Summary: {instance.instance_id}",
        "=" * 60,
        "",
        "Instance:",
        f"  Jobs: {instance.n_jobs}",
        f"  Machines: {instance.n_machines}",
        f"  Workers: {instance.n_workers}",
        f"  Operations: {instance.n_operations}",
        "",
        "Solution:",
        f"  Makespan: {schedule.makespan:.1f}",
        f"  Feasible: {schedule.is_feasible}",
        "",
    ]

    if schedule.objectives:
        lines.append("Objectives:")
        for key, value in schedule.objectives.items():
            lines.append(f"  {key}: {value:.2f}")
        lines.append("")

    if schedule.energy_breakdown:
        lines.append("Energy Breakdown:")
        for key, value in schedule.energy_breakdown.items():
            lines.append(f"  {key}: {value:.2f} kWh")
        lines.append("")

    # Machine schedule summary
    lines.append("Machine Schedule:")
    for machine_id, machine_sched in sorted(schedule.machine_schedules.items()):
        ops_str = ", ".join(
            f"J{op.job_id}.O{op.op_id}" for op in machine_sched.operations[:5]
        )
        if len(machine_sched.operations) > 5:
            ops_str += f" ... ({len(machine_sched.operations)} total)"
        lines.append(f"  M{machine_id}: {ops_str}")

    lines.append("")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    print(summary)

    return summary
