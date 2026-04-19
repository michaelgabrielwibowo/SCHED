import copy

import pytest

try:
    from ..experiments.artifact_schemas import (
        ArtifactSchemaError,
        COMPARISON_ARTIFACT_SCHEMA,
        validate_budget_sweep_payload,
        validate_comparison_payload,
        validate_policy_analysis_payload,
        validate_representation_audit_payload,
    )
    from ..experiments.analyze_nsga_policies import _build_policy_analysis_provenance
    from ..experiments.analyze_nsga_representation import _build_provenance as _build_representation_provenance
    from ..experiments.compare_solvers import (
        _apply_report_member_policy,
        _build_provenance,
        run_greedy_experiment,
        run_nsga3_experiment,
    )
    from ..experiments.sweep_nsga_budget import _build_sweep_provenance, _summarize_budget_sweep
    from ..experiments.visualize_results import (
        _extract_provenance,
        _extract_results,
        _get_experiment_metric,
        _get_plot_method_label,
    )
    from ..baseline_solver.greedy_solvers import critical_ratio_rule, least_slack_rule, spt_rule
    from ..sfjssp_model.instance import SFJSSPInstance
    from ..sfjssp_model.job import Job, Operation
    from ..sfjssp_model.machine import Machine, MachineMode
    from ..sfjssp_model.schedule import Schedule
    from ..sfjssp_model.worker import Worker
except ImportError:  # pragma: no cover - supports repo-root imports
    from experiments.artifact_schemas import (
        ArtifactSchemaError,
        COMPARISON_ARTIFACT_SCHEMA,
        validate_budget_sweep_payload,
        validate_comparison_payload,
        validate_policy_analysis_payload,
        validate_representation_audit_payload,
    )
    from experiments.analyze_nsga_policies import _build_policy_analysis_provenance
    from experiments.analyze_nsga_representation import _build_provenance as _build_representation_provenance
    from experiments.compare_solvers import (
        _apply_report_member_policy,
        _build_provenance,
        run_greedy_experiment,
        run_nsga3_experiment,
    )
    from experiments.sweep_nsga_budget import _build_sweep_provenance, _summarize_budget_sweep
    from experiments.visualize_results import (
        _extract_provenance,
        _extract_results,
        _get_experiment_metric,
        _get_plot_method_label,
    )
    from baseline_solver.greedy_solvers import critical_ratio_rule, least_slack_rule, spt_rule
    from sfjssp_model.instance import SFJSSPInstance
    from sfjssp_model.job import Job, Operation
    from sfjssp_model.machine import Machine, MachineMode
    from sfjssp_model.schedule import Schedule
    from sfjssp_model.worker import Worker


def _metrics(mk, energy, ocra, labor, tardiness, weighted, tardy_jobs):
    return {
        "makespan": mk,
        "total_energy": energy,
        "max_ergonomic_exposure": ocra,
        "total_labor_cost": labor,
        "total_tardiness": tardiness,
        "weighted_tardiness": weighted,
        "n_tardy_jobs": tardy_jobs,
    }


def _penalties(hard, tardy_jobs, weighted, total):
    return {
        "hard_violations": hard,
        "n_tardy_jobs": tardy_jobs,
        "weighted_tardiness": weighted,
        "ocra_penalty": 0.0,
        "total_penalty": total,
    }


def _comparison_provenance():
    return _build_provenance(
        benchmark_dir="benchmarks/small",
        output_path="experiments/results/comparison_2026-04-18.json",
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


def _wrap_comparison_payload(experiments):
    return {
        "provenance": _comparison_provenance(),
        "results": [{"instance": "fixture", "experiments": experiments}],
    }


def _sample_budget_results():
    specs = {
        ("A", 30): ((100.0, 200.0, 0.4, 50.0, 20.0, 20.0, 2), (105.0, 198.0, 0.38, 49.5, 10.0, 10.0, 1), 220.0, 110.0, True),
        ("A", 60): ((90.0, 190.0, 0.3, 48.0, 0.0, 0.0, 0), (90.0, 190.0, 0.3, 48.0, 0.0, 0.0, 0), 0.0, 110.0, False),
        ("B", 30): ((120.0, 210.0, 0.5, 55.0, 30.0, 30.0, 3), (125.0, 205.0, 0.45, 54.0, 15.0, 15.0, 1), 330.0, 100.0, True),
        ("B", 60): ((95.0, 195.0, 0.35, 49.0, 10.0, 10.0, 1), (98.0, 193.0, 0.33, 48.5, 5.0, 5.0, 0), 110.0, 100.0, True),
    }
    grouped = {"A": [], "B": []}
    for (instance, generations), (report_raw, tardy_raw, total_penalty, best_greedy, improves) in specs.items():
        report_metrics = _metrics(*report_raw)
        tardy_metrics = _metrics(*tardy_raw)
        report_penalties = _penalties(0.0, report_metrics["n_tardy_jobs"], report_metrics["weighted_tardiness"], total_penalty)
        tardy_penalties = _penalties(0.0, tardy_metrics["n_tardy_jobs"], tardy_metrics["weighted_tardiness"], max(tardy_metrics["weighted_tardiness"] + 100.0, 0.0))
        grouped[instance].append(
            {
                "generations": generations,
                "population_size": 30,
                "seed": 42,
                "time_seconds": 1.0 if generations == 30 else 2.0 if instance == "A" else 2.5,
                "feasible": True,
                "warm_start": True,
                "constraint_handling": "legacy_scalar_penalized_objectives",
                "report_member_key": "best_makespan_feasible",
                "report_member_selection_policy": "best_makespan_feasible_pareto",
                "report_member_is_feasible": True,
                "report_member_metrics": report_metrics,
                "report_member_penalties": report_penalties,
                "report_member_constraint_violations": [],
                "report_makespan": report_metrics["makespan"],
                "report_total_energy": report_metrics["total_energy"],
                "report_max_ergonomic_exposure": report_metrics["max_ergonomic_exposure"],
                "report_total_labor_cost": report_metrics["total_labor_cost"],
                "report_total_tardiness": report_metrics["total_tardiness"],
                "report_n_tardy_jobs": report_metrics["n_tardy_jobs"],
                "report_weighted_tardiness": report_metrics["weighted_tardiness"],
                "report_total_penalty": report_penalties["total_penalty"],
                "report_member_zero_tardy": report_metrics["n_tardy_jobs"] == 0,
                "tardiness_best_member_metrics": tardy_metrics,
                "tardiness_best_member_penalties": tardy_penalties,
                "tardiness_best_member_constraint_violations": [],
                "tardiness_best_n_tardy_jobs": tardy_metrics["n_tardy_jobs"],
                "tardiness_best_weighted_tardiness": tardy_metrics["weighted_tardiness"],
                "tardiness_best_makespan": tardy_metrics["makespan"],
                "tardiness_best_zero_tardy": tardy_metrics["n_tardy_jobs"] == 0,
                "tardiness_best_improves_report": improves,
                "min_n_tardy_jobs": tardy_metrics["n_tardy_jobs"],
                "min_weighted_tardiness": tardy_metrics["weighted_tardiness"],
                "min_total_penalty": total_penalty - 20.0 if total_penalty else 0.0,
                "zero_tardy_feasible_pareto_size": 1 if report_metrics["n_tardy_jobs"] == 0 else 0,
                "zero_penalty_pareto_size": 1 if total_penalty == 0.0 else 0,
                "feasible_pareto_size": 3,
                "pareto_size": 3,
                "beats_best_published_greedy_makespan": report_metrics["makespan"] < best_greedy,
            }
        )
    return [
        {
            "instance": instance,
            "best_published_greedy_method": "Greedy (SPT)",
            "best_published_greedy_makespan": 110.0 if instance == "A" else 100.0,
            "published_greedy_runs": [],
            "runs": sorted(runs, key=lambda run: run["generations"]),
        }
        for instance, runs in grouped.items()
    ]


def test_visualizer_accepts_legacy_list_payload():
    payload = [{"instance": "legacy", "experiments": []}]
    assert _extract_results(payload) == payload
    assert _extract_provenance(payload) is None


def test_visualizer_accepts_current_payload_with_provenance():
    payload = {
        "provenance": {"artifact_schema": COMPARISON_ARTIFACT_SCHEMA, "git_commit": "abc123"},
        "results": [{"instance": "current", "experiments": []}],
    }
    assert _extract_results(payload) == payload["results"]
    assert _extract_provenance(payload) == payload["provenance"]


def test_compare_provenance_includes_required_run_metadata():
    provenance = _comparison_provenance()
    assert provenance["artifact_schema"] == COMPARISON_ARTIFACT_SCHEMA
    assert provenance["benchmark_dir"] == "benchmarks/small"
    assert provenance["output_path"] == "experiments/results/comparison_2026-04-18.json"
    assert provenance["nsga3_generations"] == 30
    assert provenance["nsga3_population_size"] == 30
    assert provenance["nsga3_seed"] == 42
    assert provenance["nsga3_report_member_policy"] == "best_makespan_feasible"
    assert provenance["cp_enabled"] is True
    assert provenance["cp_verified_scope"] == "makespan only"
    assert provenance["python_version"]
    assert provenance["timestamp"]
    assert "git_dirty" in provenance
    assert "git_status_short" in provenance
    validate_comparison_payload({"provenance": provenance, "results": []})


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
    assert "total_energy" in result
    assert "total_labor_cost" in result


def test_tardiness_dispatch_rules_prioritize_urgent_job():
    instance = SFJSSPInstance(instance_id="dispatch_urgency_test")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(Worker(worker_id=0, min_rest_fraction=0.0, ocra_max_per_shift=999.0))
    for job_id, due_date, duration in ((0, 30.0, 20.0), (1, 120.0, 10.0)):
        instance.add_job(
            Job(
                job_id=job_id,
                due_date=due_date,
                arrival_time=0.0,
                operations=[
                    Operation(
                        job_id=job_id,
                        op_id=0,
                        processing_times={0: {0: duration}},
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


def test_nsga3_experiment_reports_canonical_report_member_contract():
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
    assert result["report_member_metrics"]["makespan"] is not None
    assert result["report_member_penalties"]["hard_violations"] == 0
    assert result["report_makespan"] == result["report_member_metrics"]["makespan"]
    assert result["report_total_energy"] == result["report_member_metrics"]["total_energy"]
    assert result["report_max_ergonomic_exposure"] == result["report_member_metrics"]["max_ergonomic_exposure"]
    assert result["report_total_labor_cost"] == result["report_member_metrics"]["total_labor_cost"]
    assert result["makespan"] == result["report_member_metrics"]["makespan"]
    assert result["total_energy"] == result["report_member_metrics"]["total_energy"]
    assert result["max_ergonomic_exposure"] == result["report_member_metrics"]["max_ergonomic_exposure"]
    assert result["total_labor_cost"] == result["report_member_metrics"]["total_labor_cost"]
    assert result["tardiness_best_member_policy"] == "min_weighted_tardiness_feasible_pareto"
    assert result["tardiness_best_member_metrics"]["weighted_tardiness"] is not None
    assert result["tardiness_best_is_same_as_report_member"] in {True, False}
    assert not any(key.startswith("selected_") for key in result)
    assert result["report_member_is_feasible"] is True
    assert result["report_member_penalties"]["hard_violations"] == 0
    assert result["report_member_constraint_violations"] == []
    assert len(result["generation_diagnostics"]) == 1
    assert result["generation_diagnostics"][0]["report_member_key"] == "best_makespan_feasible"
    validate_comparison_payload(_wrap_comparison_payload([result]))


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
    assert "urgent_crossover_attempts" in result
    assert "children_evaluated_count" in result
    assert "urgent_sequence_mutation_attempts" in result
    assert "immigrant_archive_final_size" in result
    assert "immigrant_injected_individuals" in result
    assert len(result["generation_diagnostics"]) == 2
    assert result["generation_diagnostics"][1]["crossover_policy"] == "urgent_prefix_merge"
    assert not any(key.startswith("selected_") for key in result)


def test_report_member_policy_contract_follows_requested_policy():
    representative_members = {
        "best_makespan_feasible": {
            "policy": "best_makespan_feasible_pareto",
            "is_feasible": True,
            "metrics": _metrics(100.0, 200.0, 0.4, 50.0, 30.0, 40.0, 4),
            "penalties": _penalties(0.0, 4, 40.0, 440.0),
            "constraint_violations": [],
        },
        "min_n_tardy_feasible": {
            "policy": "min_n_tardy_feasible_pareto",
            "is_feasible": True,
            "metrics": _metrics(120.0, 210.0, 0.3, 52.0, 12.0, 15.0, 2),
            "penalties": _penalties(0.0, 2, 15.0, 215.0),
            "constraint_violations": [],
        },
        "min_weighted_tardiness_feasible": {
            "policy": "min_weighted_tardiness_feasible_pareto",
            "is_feasible": True,
            "metrics": _metrics(130.0, 220.0, 0.2, 55.0, 8.0, 10.0, 2),
            "penalties": _penalties(0.0, 2, 10.0, 210.0),
            "constraint_violations": [],
        },
    }
    result = _apply_report_member_policy(
        {"representative_members": representative_members},
        representative_members=representative_members,
        report_member_policy="min_weighted_tardiness_feasible",
    )
    assert result["report_member_key"] == "min_weighted_tardiness_feasible"
    assert result["report_member_selection_policy"] == "min_weighted_tardiness_feasible_pareto"
    assert result["report_makespan"] == 130.0
    assert result["report_weighted_tardiness"] == 10.0
    assert result["report_total_energy"] == 220.0
    assert result["tardiness_best_is_same_as_report_member"] is True
    assert not any(key.startswith("selected_") for key in result)
    makespan_result = _apply_report_member_policy(
        {"representative_members": representative_members},
        representative_members=representative_members,
        report_member_policy="best_makespan_feasible",
    )
    assert makespan_result["report_member_selection_policy"] == "best_makespan_feasible_pareto"
    assert makespan_result["report_weighted_tardiness"] == 40.0
    assert makespan_result["tardiness_best_weighted_tardiness"] == 10.0
    assert makespan_result["tardiness_best_is_same_as_report_member"] is False


def test_visualizer_labels_nsga_series_with_report_policy_and_strict_current_schema():
    provenance = {"artifact_schema": COMPARISON_ARTIFACT_SCHEMA}
    exp = {
        "method": "NSGA-III (30 gen)",
        "report_member_key": "min_weighted_tardiness_feasible",
        "report_member_metrics": _metrics(120.0, 210.0, 0.3, 52.0, 8.0, 10.0, 2),
        "report_weighted_tardiness": 10.0,
        "report_n_tardy_jobs": 2,
    }
    assert _get_plot_method_label(exp) == "NSGA-III (30 gen) [min_weighted_tardiness_feasible]"
    assert _get_experiment_metric(exp, "makespan", provenance=provenance) == 120.0
    assert _get_experiment_metric(exp, "total_energy", provenance=provenance) == 210.0
    assert _get_experiment_metric(exp, "weighted_tardiness", provenance=provenance) == 10.0
    malformed = {"method": "NSGA-III (30 gen)", "report_member_key": "best_makespan_feasible"}
    with pytest.raises(ValueError, match="report_member_metrics"):
        _get_experiment_metric(malformed, "makespan", provenance=provenance)


def test_nsga_budget_sweep_summary_aggregates_by_report_contract():
    results = _sample_budget_results()
    summary = _summarize_budget_sweep(results)
    thirty = next(row for row in summary if row["generations"] == 30 and row["population_size"] == 30)
    sixty = next(row for row in summary if row["generations"] == 60 and row["population_size"] == 30)
    assert thirty["constraint_handling"] == "legacy_scalar_penalized_objectives"
    assert thirty["avg_report_makespan"] == 110.0
    assert thirty["avg_report_total_energy"] == 205.0
    assert thirty["avg_report_weighted_tardiness"] == 25.0
    assert thirty["avg_tardiness_best_weighted_tardiness"] == 12.5
    assert thirty["instances_beating_best_published_greedy_makespan"] == 1
    assert sixty["avg_report_n_tardy_jobs"] == 0.5
    assert sixty["instances_with_zero_tardy_feasible_member"] == 1
    assert sixty["instances_with_report_zero_tardy_member"] == 1
    assert sixty["avg_tardiness_best_n_tardy_jobs"] == 0.0
    assert thirty["instances_where_tardiness_best_improves_report"] == 2
    assert not any("selected" in key for key in thirty)
    payload = {
        "provenance": _build_sweep_provenance(
            benchmark_dir="benchmarks/small",
            output_path="experiments/results/nsga_budget_sweep.json",
            generations=[30, 60],
            population_sizes=[30],
            warm_start=True,
            seed=42,
            constraint_handling="legacy_scalar_penalized_objectives",
            crossover_policy="legacy_pox_uniform_assignments",
            sequence_mutation="legacy_random_swap",
            immigrant_policy="none",
            immigrant_count=2,
            immigrant_period=5,
            immigrant_archive_size=8,
            report_member_policy="best_makespan_feasible",
            command="python -m experiments.sweep_nsga_budget",
        ),
        "results": results,
        "summary": summary,
        "recommended_budget": summary[0],
    }
    validate_budget_sweep_payload(payload)


def test_current_artifact_validators_reject_stale_aliases():
    experiment = {
        "method": "NSGA-III (30 gen)",
        "time_seconds": 1.0,
        "feasible": True,
        "report_member_key": "best_makespan_feasible",
        "report_member_selection_policy": "best_makespan_feasible_pareto",
        "report_member_is_feasible": True,
        "report_member_metrics": _metrics(100.0, 200.0, 0.4, 50.0, 20.0, 20.0, 2),
        "report_member_penalties": _penalties(0.0, 2, 20.0, 220.0),
        "report_member_constraint_violations": [],
        "report_makespan": 100.0,
        "report_total_energy": 200.0,
        "report_max_ergonomic_exposure": 0.4,
        "report_total_labor_cost": 50.0,
        "report_total_tardiness": 20.0,
        "report_n_tardy_jobs": 2,
        "report_weighted_tardiness": 20.0,
        "report_total_penalty": 220.0,
        "representative_members": {},
        "tardiness_best_member_policy": "min_weighted_tardiness_feasible_pareto",
        "tardiness_best_member_metrics": _metrics(105.0, 198.0, 0.38, 49.5, 10.0, 10.0, 1),
        "tardiness_best_member_penalties": _penalties(0.0, 1, 10.0, 110.0),
        "tardiness_best_member_constraint_violations": [],
        "makespan": 100.0,
        "total_energy": 200.0,
        "max_ergonomic_exposure": 0.4,
        "total_labor_cost": 50.0,
        "total_tardiness": 20.0,
        "weighted_tardiness": 20.0,
        "n_tardy_jobs": 2,
    }
    valid_payload = _wrap_comparison_payload([experiment])
    validate_comparison_payload(valid_payload)
    stale_selected = copy.deepcopy(valid_payload)
    stale_selected["results"][0]["experiments"][0]["selected_member_metrics"] = {}
    with pytest.raises(ArtifactSchemaError, match="selected"):
        validate_comparison_payload(stale_selected)
    stale_alias = copy.deepcopy(valid_payload)
    stale_alias["results"][0]["experiments"][0]["labor_cost"] = 50.0
    with pytest.raises(ArtifactSchemaError, match="legacy metric aliases"):
        validate_comparison_payload(stale_alias)


def test_policy_and_representation_artifacts_require_current_provenance():
    policy_payload = {
        "provenance": _build_policy_analysis_provenance(
            benchmark_dir="benchmarks/small",
            output_path="experiments/results/nsga_policy_analysis.json",
            generations=30,
            population_size=30,
            seed=42,
            warm_start=True,
            report_member_policy="best_makespan_feasible",
            command="python -m experiments.analyze_nsga_policies",
        ),
        "results": [],
        "summary": [],
    }
    validate_policy_analysis_payload(policy_payload)
    representation_payload = {
        "provenance": _build_representation_provenance(
            benchmark_dir="benchmarks/small",
            output_path="experiments/results/nsga_representation_audit.json",
            baseline_artifact="experiments/results/comparison-baseline.json",
            reference_artifact="experiments/results/comparison-reference.json",
            seed=42,
            command="python -m experiments.analyze_nsga_representation",
        ),
        "baseline_truth_table": {},
        "results": [],
        "aggregate_summary": {},
    }
    validate_representation_audit_payload(representation_payload)
