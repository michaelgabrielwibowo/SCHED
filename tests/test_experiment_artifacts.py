try:
    from ..experiments.visualize_results import (
        _extract_provenance,
        _extract_results,
        _get_experiment_metric,
        _get_plot_method_label,
    )
    from ..experiments.compare_solvers import (
        _apply_report_member_policy,
        _build_provenance,
        run_greedy_experiment,
        run_nsga3_experiment,
    )
    from ..experiments.sweep_nsga_budget import _summarize_budget_sweep
    from ..baseline_solver.greedy_solvers import (
        critical_ratio_rule,
        least_slack_rule,
        spt_rule,
    )
    from ..sfjssp_model.schedule import Schedule
    from ..sfjssp_model.instance import SFJSSPInstance
    from ..sfjssp_model.job import Job, Operation
    from ..sfjssp_model.machine import Machine, MachineMode
    from ..sfjssp_model.worker import Worker
except ImportError:  # pragma: no cover - supports repo-root imports
    from experiments.visualize_results import (
        _extract_provenance,
        _extract_results,
        _get_experiment_metric,
        _get_plot_method_label,
    )
    from experiments.compare_solvers import (
        _apply_report_member_policy,
        _build_provenance,
        run_greedy_experiment,
        run_nsga3_experiment,
    )
    from experiments.sweep_nsga_budget import _summarize_budget_sweep
    from baseline_solver.greedy_solvers import (
        critical_ratio_rule,
        least_slack_rule,
        spt_rule,
    )
    from sfjssp_model.schedule import Schedule
    from sfjssp_model.instance import SFJSSPInstance
    from sfjssp_model.job import Job, Operation
    from sfjssp_model.machine import Machine, MachineMode
    from sfjssp_model.worker import Worker


def test_visualizer_accepts_legacy_list_payload():
    payload = [{"instance": "legacy", "experiments": []}]

    assert _extract_results(payload) == payload
    assert _extract_provenance(payload) is None


def test_visualizer_accepts_v2_payload_with_provenance():
    payload = {
        "provenance": {"artifact_schema": "comparison_results_v4", "git_commit": "abc123"},
        "results": [{"instance": "current", "experiments": []}],
    }

    assert _extract_results(payload) == payload["results"]
    assert _extract_provenance(payload) == payload["provenance"]


def test_compare_provenance_includes_required_run_metadata():
    provenance = _build_provenance(
        benchmark_dir="benchmarks/small",
        output_path="experiments/results/comparison_2026-04-12.json",
        run_cp=True,
        nsga3_generations=30,
        nsga3_population_size=30,
        nsga3_warm_start=True,
        nsga3_seed=42,
        nsga3_constraint_handling="legacy_scalar_penalized_objectives",
        nsga3_parent_selection="random_pairing",
        nsga3_crossover_policy="legacy_pox_uniform_assignments",
        nsga3_local_improvement="none",
        nsga3_sequence_mutation="legacy_random_swap",
        nsga3_immigrant_policy="none",
        nsga3_immigrant_count=2,
        nsga3_immigrant_period=5,
        nsga3_immigrant_archive_size=8,
        nsga3_report_member_policy="best_makespan_feasible",
        command="python -m experiments.compare_solvers --benchmark-dir benchmarks/small",
    )

    assert provenance["artifact_schema"] == "comparison_results_v4"
    assert provenance["benchmark_dir"] == "benchmarks/small"
    assert provenance["output_path"] == "experiments/results/comparison_2026-04-12.json"
    assert provenance["nsga3_generations"] == 30
    assert provenance["nsga3_population_size"] == 30
    assert provenance["nsga3_seed"] == 42
    assert provenance["nsga3_constraint_handling"] == "legacy_scalar_penalized_objectives"
    assert provenance["nsga3_parent_selection"] == "random_pairing"
    assert provenance["nsga3_crossover_policy"] == "legacy_pox_uniform_assignments"
    assert provenance["nsga3_local_improvement"] == "none"
    assert provenance["nsga3_sequence_mutation"] == "legacy_random_swap"
    assert provenance["nsga3_immigrant_policy"] == "none"
    assert provenance["nsga3_immigrant_count"] == 2
    assert provenance["nsga3_immigrant_period"] == 5
    assert provenance["nsga3_immigrant_archive_size"] == 8
    assert provenance["nsga3_report_member_policy"] == "best_makespan_feasible"
    assert provenance["cp_enabled"] is True
    assert provenance["cp_verified_scope"] == "makespan only"
    assert provenance["python_version"]
    assert provenance["timestamp"]
    assert "git_dirty" in provenance
    assert "git_status_short" in provenance


def test_compare_solver_runs_on_cloned_instance_state():
    instance = SFJSSPInstance(instance_id="compare_clone_test")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(Worker(worker_id=0, min_rest_fraction=0.0, ocra_max_per_shift=999.0))
    instance.add_job(
        Job(
            job_id=0,
            operations=[
                Operation(
                    job_id=0,
                    op_id=0,
                    processing_times={0: {0: 10.0}},
                    eligible_machines={0},
                    eligible_workers={0},
                )
            ],
        )
    )
    instance.ergonomic_risk_map[(0, 0)] = 0.0

    machine = instance.get_machine(0)
    worker = instance.get_worker(0)

    result = run_greedy_experiment(instance, "SPT", spt_rule)

    assert result["feasible"] is True
    assert machine.available_time == 0.0
    assert worker.available_time == 0.0


def test_tardiness_dispatch_rules_prioritize_urgent_job():
    instance = SFJSSPInstance(instance_id="dispatch_urgency_test")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(Worker(worker_id=0, min_rest_fraction=0.0, ocra_max_per_shift=999.0))
    instance.add_job(
        Job(
            job_id=0,
            due_date=30.0,
            arrival_time=0.0,
            operations=[
                Operation(
                    job_id=0,
                    op_id=0,
                    processing_times={0: {0: 20.0}},
                    eligible_machines={0},
                    eligible_workers={0},
                )
            ],
        )
    )
    instance.add_job(
        Job(
            job_id=1,
            due_date=120.0,
            arrival_time=0.0,
            operations=[
                Operation(
                    job_id=1,
                    op_id=0,
                    processing_times={0: {0: 10.0}},
                    eligible_machines={0},
                    eligible_workers={0},
                )
            ],
        )
    )

    ready_ops = [(0, 0), (1, 0)]
    empty_schedule = Schedule(instance_id="dispatch_urgency_test")

    assert least_slack_rule(instance, empty_schedule, ready_ops) == 0
    assert critical_ratio_rule(instance, empty_schedule, ready_ops) == 0


def test_nsga3_experiment_reports_selected_member_penalty_breakdown():
    instance = SFJSSPInstance(instance_id="compare_nsga_test")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_machine(Machine(machine_id=1, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(Worker(worker_id=0, min_rest_fraction=0.0, ocra_max_per_shift=999.0))
    instance.add_worker(Worker(worker_id=1, min_rest_fraction=0.0, ocra_max_per_shift=999.0))
    for job_id in range(2):
        instance.add_job(
            Job(
                job_id=job_id,
                due_date=9999.0,
                operations=[
                    Operation(
                        job_id=job_id,
                        op_id=0,
                        processing_times={0: {0: 8.0}, 1: {0: 10.0}},
                        eligible_machines={0, 1},
                        eligible_workers={0, 1},
                    )
                ],
            )
        )
        instance.ergonomic_risk_map[(job_id, 0)] = 0.0

    result = run_nsga3_experiment(
        instance,
        n_generations=0,
        population_size=4,
        warm_start=False,
        seed=7,
        crossover_policy="legacy_pox_uniform_assignments",
        local_improvement="none",
        sequence_mutation="legacy_random_swap",
        immigrant_policy="none",
        immigrant_count=2,
        immigrant_period=5,
        immigrant_archive_size=8,
        collect_generation_diagnostics=True,
    )

    assert result["feasible"] is True
    assert result["report_member_key"] == "best_makespan_feasible"
    assert result["report_member_selection_policy"] == "best_makespan_feasible_pareto"
    assert result["report_member_is_feasible"] is True
    assert result["report_makespan"] == result["selected_member_metrics"]["makespan"]
    assert result["constraint_handling"] == "legacy_scalar_penalized_objectives"
    assert result["parent_selection"] == "random_pairing"
    assert result["crossover_policy"] == "legacy_pox_uniform_assignments"
    assert result["local_improvement"] == "none"
    assert result["sequence_mutation"] == "legacy_random_swap"
    assert result["immigrant_policy"] == "none"
    assert result["immigrant_count"] == 2
    assert result["immigrant_period"] == 5
    assert result["immigrant_archive_size"] == 8
    assert result["selected_member_policy"] == "best_makespan_feasible_pareto"
    assert result["selected_member_is_feasible"] is True
    assert result["selected_member_metrics"]["makespan"] is not None
    assert result["selected_member_penalties"]["hard_violations"] == 0
    assert result["selected_n_tardy_jobs"] == result["selected_member_metrics"]["n_tardy_jobs"]
    assert result["selected_total_penalty"] == result["selected_member_penalties"]["total_penalty"]
    assert set(result["representative_members"]) == {
        "best_makespan_feasible",
        "min_n_tardy_feasible",
        "min_weighted_tardiness_feasible",
    }
    assert result["tardiness_best_member_policy"] == "min_weighted_tardiness_feasible_pareto"
    assert result["tardiness_best_member_metrics"]["weighted_tardiness"] is not None
    assert result["repair_attempted_children"] == 0
    assert result["repair_accepted_children"] == 0
    assert result["urgent_crossover_attempts"] == 0
    assert result["urgent_crossover_applied"] == 0
    assert result["children_evaluated_count"] == 0
    assert result["urgent_sequence_mutation_attempts"] == 0
    assert result["urgent_sequence_mutation_applied"] == 0
    assert result["immigrant_archive_final_size"] == 0
    assert result["immigrant_injected_individuals"] == 0
    assert len(result["generation_diagnostics"]) == 1
    assert result["generation_diagnostics"][0]["generation"] == 0
    assert result["generation_diagnostics"][0]["report_member_key"] == "best_makespan_feasible"
    assert result["generation_diagnostics"][0]["crossover_policy"] == "legacy_pox_uniform_assignments"
    assert result["generation_diagnostics"][0]["local_improvement"] == "none"
    assert result["generation_diagnostics"][0]["sequence_mutation"] == "legacy_random_swap"
    assert result["generation_diagnostics"][0]["immigrant_policy"] == "none"
    assert result["generation_diagnostics"][0]["repair_attempted_children"] == 0
    assert result["generation_diagnostics"][0]["urgent_crossover_attempts"] == 0
    assert result["generation_diagnostics"][0]["urgent_sequence_mutation_attempts"] == 0
    assert result["generation_diagnostics"][0]["children_evaluated_count"] == 0
    assert result["generation_diagnostics"][0]["immigrant_archive_size"] == 0
    assert result["generation_diagnostics"][0]["immigrant_injected_individuals"] == 0


def test_nsga3_experiment_surfaces_local_improvement_provenance_and_counters():
    instance = SFJSSPInstance(instance_id="compare_nsga_repair_test")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(Worker(worker_id=0, min_rest_fraction=0.0, ocra_max_per_shift=999.0))

    for job_id, due_date in enumerate((3.0, 100.0, 100.0)):
        instance.add_job(
            Job(
                job_id=job_id,
                due_date=due_date,
                operations=[
                    Operation(
                        job_id=job_id,
                        op_id=0,
                        processing_times={0: {0: 3.0}},
                        eligible_machines={0},
                        eligible_workers={0},
                    )
                ],
            )
        )
        instance.ergonomic_risk_map[(job_id, 0)] = 0.0

    result = run_nsga3_experiment(
        instance,
        n_generations=1,
        population_size=6,
        warm_start=False,
        seed=7,
        crossover_policy="urgent_prefix_merge",
        local_improvement="tardiness_sequence_repair",
        sequence_mutation="urgent_pull_forward",
        immigrant_policy="feasible_tardiness_archive",
        immigrant_count=1,
        immigrant_period=1,
        immigrant_archive_size=4,
        collect_generation_diagnostics=True,
    )

    assert result["local_improvement"] == "tardiness_sequence_repair"
    assert result["crossover_policy"] == "urgent_prefix_merge"
    assert result["sequence_mutation"] == "urgent_pull_forward"
    assert result["immigrant_policy"] == "feasible_tardiness_archive"
    assert "repair_attempted_children" in result
    assert "repair_accepted_children" in result
    assert "urgent_crossover_attempts" in result
    assert "children_evaluated_count" in result
    assert "urgent_sequence_mutation_attempts" in result
    assert "urgent_sequence_mutation_applied" in result
    assert "immigrant_archive_final_size" in result
    assert "immigrant_injected_individuals" in result
    assert len(result["generation_diagnostics"]) == 2
    assert result["generation_diagnostics"][1]["crossover_policy"] == "urgent_prefix_merge"
    assert result["generation_diagnostics"][1]["local_improvement"] == "tardiness_sequence_repair"
    assert result["generation_diagnostics"][1]["sequence_mutation"] == "urgent_pull_forward"
    assert result["generation_diagnostics"][1]["immigrant_policy"] == "feasible_tardiness_archive"
    assert result["generation_diagnostics"][1]["repair_attempted_children"] >= 0
    assert result["generation_diagnostics"][1]["urgent_crossover_attempts"] >= 0
    assert result["generation_diagnostics"][1]["children_evaluated_count"] >= 0
    assert result["generation_diagnostics"][1]["urgent_sequence_mutation_attempts"] >= 0
    assert result["generation_diagnostics"][1]["immigrant_archive_size"] >= 0
    assert result["generation_diagnostics"][1]["immigrant_injected_individuals"] >= 0


def test_report_member_policy_aliases_follow_requested_policy():
    representative_members = {
        "best_makespan_feasible": {
            "policy": "best_makespan_feasible_pareto",
            "is_feasible": True,
            "metrics": {
                "makespan": 100.0,
                "total_energy": 200.0,
                "max_ergonomic_exposure": 0.4,
                "total_labor_cost": 50.0,
                "total_tardiness": 30.0,
                "weighted_tardiness": 40.0,
                "n_tardy_jobs": 4,
            },
            "penalties": {
                "hard_violations": 0,
                "n_tardy_jobs": 4,
                "weighted_tardiness": 40.0,
                "ocra_penalty": 0.0,
                "total_penalty": 440.0,
            },
            "constraint_violations": [],
        },
        "min_n_tardy_feasible": {
            "policy": "min_n_tardy_feasible_pareto",
            "is_feasible": True,
            "metrics": {
                "makespan": 120.0,
                "total_energy": 210.0,
                "max_ergonomic_exposure": 0.3,
                "total_labor_cost": 52.0,
                "total_tardiness": 12.0,
                "weighted_tardiness": 15.0,
                "n_tardy_jobs": 2,
            },
            "penalties": {
                "hard_violations": 0,
                "n_tardy_jobs": 2,
                "weighted_tardiness": 15.0,
                "ocra_penalty": 0.0,
                "total_penalty": 215.0,
            },
            "constraint_violations": [],
        },
        "min_weighted_tardiness_feasible": {
            "policy": "min_weighted_tardiness_feasible_pareto",
            "is_feasible": True,
            "metrics": {
                "makespan": 130.0,
                "total_energy": 220.0,
                "max_ergonomic_exposure": 0.2,
                "total_labor_cost": 55.0,
                "total_tardiness": 8.0,
                "weighted_tardiness": 10.0,
                "n_tardy_jobs": 2,
            },
            "penalties": {
                "hard_violations": 0,
                "n_tardy_jobs": 2,
                "weighted_tardiness": 10.0,
                "ocra_penalty": 0.0,
                "total_penalty": 210.0,
            },
            "constraint_violations": [],
        },
    }

    result = _apply_report_member_policy(
        {"representative_members": representative_members},
        representative_members=representative_members,
        report_member_policy="min_weighted_tardiness_feasible",
    )

    assert result["representative_members"] == representative_members
    assert result["report_member_key"] == "min_weighted_tardiness_feasible"
    assert result["report_member_selection_policy"] == "min_weighted_tardiness_feasible_pareto"
    assert result["report_makespan"] == 130.0
    assert result["report_weighted_tardiness"] == 10.0
    assert result["selected_member_policy"] == "min_weighted_tardiness_feasible_pareto"
    assert result["selected_weighted_tardiness"] == 10.0
    assert result["tardiness_best_is_same_as_selected"] is True

    makespan_result = _apply_report_member_policy(
        {"representative_members": representative_members},
        representative_members=representative_members,
        report_member_policy="best_makespan_feasible",
    )

    assert makespan_result["selected_member_policy"] == "best_makespan_feasible_pareto"
    assert makespan_result["selected_weighted_tardiness"] == 40.0
    assert makespan_result["tardiness_best_weighted_tardiness"] == 10.0
    assert makespan_result["tardiness_best_is_same_as_selected"] is False


def test_visualizer_labels_nsga_series_with_report_policy():
    exp = {
        "method": "NSGA-III (30 gen)",
        "report_member_key": "min_weighted_tardiness_feasible",
        "report_member_metrics": {"makespan": 120.0, "total_energy": 210.0},
        "report_weighted_tardiness": 10.0,
    }

    assert _get_plot_method_label(exp) == "NSGA-III (30 gen) [min_weighted_tardiness_feasible]"
    assert _get_experiment_metric(exp, "makespan") == 120.0
    assert _get_experiment_metric(exp, "energy") == 210.0
    assert _get_experiment_metric(exp, "weighted_tardiness") == 10.0


def test_nsga_budget_sweep_summary_aggregates_by_config():
    summary = _summarize_budget_sweep(
        [
            {
                "instance": "A",
                "runs": [
                    {
                        "generations": 30,
                        "population_size": 30,
                        "constraint_handling": "legacy_scalar_penalized_objectives",
                        "report_member_key": "best_makespan_feasible",
                        "report_member_metrics": {"makespan": 100.0},
                        "report_weighted_tardiness": 20.0,
                        "report_n_tardy_jobs": 2,
                        "report_total_penalty": 220.0,
                        "report_member_zero_tardy": False,
                        "selected_member_metrics": {"makespan": 100.0},
                        "selected_weighted_tardiness": 20.0,
                        "selected_n_tardy_jobs": 2,
                        "selected_total_penalty": 220.0,
                        "tardiness_best_member_metrics": {"makespan": 105.0},
                        "tardiness_best_weighted_tardiness": 10.0,
                        "tardiness_best_n_tardy_jobs": 1,
                        "min_total_penalty": 200.0,
                        "beats_best_published_greedy_makespan": True,
                        "zero_tardy_feasible_pareto_size": 0,
                        "selected_member_zero_tardy": False,
                        "tardiness_best_zero_tardy": False,
                        "tardiness_best_improves_selected": True,
                        "time_seconds": 1.0,
                    },
                    {
                        "generations": 60,
                        "population_size": 30,
                        "constraint_handling": "legacy_scalar_penalized_objectives",
                        "report_member_key": "best_makespan_feasible",
                        "report_member_metrics": {"makespan": 90.0},
                        "report_weighted_tardiness": 0.0,
                        "report_n_tardy_jobs": 0,
                        "report_total_penalty": 0.0,
                        "report_member_zero_tardy": True,
                        "selected_member_metrics": {"makespan": 90.0},
                        "selected_weighted_tardiness": 0.0,
                        "selected_n_tardy_jobs": 0,
                        "selected_total_penalty": 0.0,
                        "tardiness_best_member_metrics": {"makespan": 90.0},
                        "tardiness_best_weighted_tardiness": 0.0,
                        "tardiness_best_n_tardy_jobs": 0,
                        "min_total_penalty": 0.0,
                        "beats_best_published_greedy_makespan": True,
                        "zero_tardy_feasible_pareto_size": 1,
                        "selected_member_zero_tardy": True,
                        "tardiness_best_zero_tardy": True,
                        "tardiness_best_improves_selected": False,
                        "time_seconds": 2.0,
                    },
                ],
            },
            {
                "instance": "B",
                "runs": [
                    {
                        "generations": 30,
                        "population_size": 30,
                        "constraint_handling": "legacy_scalar_penalized_objectives",
                        "report_member_key": "best_makespan_feasible",
                        "report_member_metrics": {"makespan": 120.0},
                        "report_weighted_tardiness": 30.0,
                        "report_n_tardy_jobs": 3,
                        "report_total_penalty": 330.0,
                        "report_member_zero_tardy": False,
                        "selected_member_metrics": {"makespan": 120.0},
                        "selected_weighted_tardiness": 30.0,
                        "selected_n_tardy_jobs": 3,
                        "selected_total_penalty": 330.0,
                        "tardiness_best_member_metrics": {"makespan": 125.0},
                        "tardiness_best_weighted_tardiness": 15.0,
                        "tardiness_best_n_tardy_jobs": 1,
                        "min_total_penalty": 300.0,
                        "beats_best_published_greedy_makespan": False,
                        "zero_tardy_feasible_pareto_size": 0,
                        "selected_member_zero_tardy": False,
                        "tardiness_best_zero_tardy": False,
                        "tardiness_best_improves_selected": True,
                        "time_seconds": 1.5,
                    },
                    {
                        "generations": 60,
                        "population_size": 30,
                        "constraint_handling": "legacy_scalar_penalized_objectives",
                        "report_member_key": "best_makespan_feasible",
                        "report_member_metrics": {"makespan": 95.0},
                        "report_weighted_tardiness": 10.0,
                        "report_n_tardy_jobs": 1,
                        "report_total_penalty": 110.0,
                        "report_member_zero_tardy": False,
                        "selected_member_metrics": {"makespan": 95.0},
                        "selected_weighted_tardiness": 10.0,
                        "selected_n_tardy_jobs": 1,
                        "selected_total_penalty": 110.0,
                        "tardiness_best_member_metrics": {"makespan": 98.0},
                        "tardiness_best_weighted_tardiness": 5.0,
                        "tardiness_best_n_tardy_jobs": 0,
                        "min_total_penalty": 100.0,
                        "beats_best_published_greedy_makespan": True,
                        "zero_tardy_feasible_pareto_size": 0,
                        "selected_member_zero_tardy": False,
                        "tardiness_best_zero_tardy": True,
                        "tardiness_best_improves_selected": True,
                        "time_seconds": 2.5,
                    },
                ],
            },
        ]
    )

    thirty = next(row for row in summary if row["generations"] == 30 and row["population_size"] == 30)
    sixty = next(row for row in summary if row["generations"] == 60 and row["population_size"] == 30)

    assert thirty["constraint_handling"] == "legacy_scalar_penalized_objectives"
    assert thirty["avg_selected_makespan"] == 110.0
    assert thirty["avg_report_makespan"] == 110.0
    assert thirty["avg_selected_weighted_tardiness"] == 25.0
    assert thirty["avg_report_weighted_tardiness"] == 25.0
    assert thirty["avg_tardiness_best_weighted_tardiness"] == 12.5
    assert thirty["instances_beating_best_published_greedy_makespan"] == 1
    assert sixty["avg_selected_n_tardy_jobs"] == 0.5
    assert sixty["avg_report_n_tardy_jobs"] == 0.5
    assert sixty["instances_with_zero_tardy_feasible_member"] == 1
    assert sixty["instances_with_report_zero_tardy_member"] == 1
    assert sixty["instances_with_selected_zero_tardy_member"] == 1
    assert sixty["avg_tardiness_best_n_tardy_jobs"] == 0.0
    assert thirty["instances_where_tardiness_best_improves_selected"] == 2
