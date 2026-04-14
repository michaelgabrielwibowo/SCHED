import json
from pathlib import Path

import numpy as np

try:
    from ..agents.policy_networks import MultiAgentPPO, _safe_categorical_from_logits
    from ..environment.sfjssp_env import SFJSSPEnv
    from ..experiments.analyze_nsga_representation import (
        _best_generation0_seed,
        _build_assignment_family_candidates,
        _build_mixed_family_candidates,
        _build_offset_family_candidates,
        _build_sequence_family_candidates,
        _metric_signature,
        _summarize_family,
    )
    from ..moea.nsga3 import (
        NSGA3,
        NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        NSGA3_DEFAULT_CROSSOVER_POLICY,
        NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        NSGA3_DEFAULT_SEQUENCE_MUTATION,
        NSGA3_DEFAULT_IMMIGRANT_POLICY,
        _attempt_tardiness_sequence_repair,
        _build_urgent_prefix_merged_sequence,
        _apply_urgent_pull_forward_sequence_mutation,
        Individual,
        Population,
        _collect_seed_genome_candidates,
        create_sfjssp_genome,
        create_sfjssp_seed_genomes,
        evaluate_sfjssp_genome,
        evaluate_sfjssp_genome_detailed,
        schedule_to_sfjssp_genome,
    )
    from ..sfjssp_model.instance import SFJSSPInstance
    from ..sfjssp_model.job import Job, Operation
    from ..sfjssp_model.machine import Machine, MachineMode
    from ..sfjssp_model.schedule import Schedule
    from ..sfjssp_model.worker import Worker
    from ..training.train_drl import TrainingConfig, TrainingPipeline, TORCH_AVAILABLE
except ImportError:  # pragma: no cover - supports repo-root imports
    from agents.policy_networks import MultiAgentPPO, _safe_categorical_from_logits
    from environment.sfjssp_env import SFJSSPEnv
    from experiments.analyze_nsga_representation import (
        _best_generation0_seed,
        _build_assignment_family_candidates,
        _build_mixed_family_candidates,
        _build_offset_family_candidates,
        _build_sequence_family_candidates,
        _metric_signature,
        _summarize_family,
    )
    from moea.nsga3 import (
        NSGA3,
        NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        NSGA3_DEFAULT_CROSSOVER_POLICY,
        NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        NSGA3_DEFAULT_SEQUENCE_MUTATION,
        NSGA3_DEFAULT_IMMIGRANT_POLICY,
        _attempt_tardiness_sequence_repair,
        _build_urgent_prefix_merged_sequence,
        _apply_urgent_pull_forward_sequence_mutation,
        Individual,
        Population,
        _collect_seed_genome_candidates,
        create_sfjssp_genome,
        create_sfjssp_seed_genomes,
        evaluate_sfjssp_genome,
        evaluate_sfjssp_genome_detailed,
        schedule_to_sfjssp_genome,
    )
    from sfjssp_model.instance import SFJSSPInstance
    from sfjssp_model.job import Job, Operation
    from sfjssp_model.machine import Machine, MachineMode
    from sfjssp_model.schedule import Schedule
    from sfjssp_model.worker import Worker
    from training.train_drl import TrainingConfig, TrainingPipeline, TORCH_AVAILABLE


def _build_simple_instance() -> SFJSSPInstance:
    instance = SFJSSPInstance(instance_id="RUNTIME_TEST")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(
        Worker(
            worker_id=0,
            min_rest_fraction=0.0,
            ocra_max_per_shift=999.0,
        )
    )
    instance.add_job(
        Job(
            job_id=0,
            arrival_time=0.0,
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
    return instance


def _build_seedable_nsga_instance() -> SFJSSPInstance:
    instance = SFJSSPInstance(instance_id="NSGA_SEED_TEST")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_machine(Machine(machine_id=1, modes=[MachineMode(mode_id=0)]))
    for worker_id in range(2):
        instance.add_worker(
            Worker(
                worker_id=worker_id,
                min_rest_fraction=0.0,
                ocra_max_per_shift=999.0,
            )
        )

    processing_options = [
        ({0: {0: 8.0}, 1: {0: 14.0}}, {0: {0: 12.0}, 1: {0: 7.0}}),
        ({0: {0: 9.0}, 1: {0: 15.0}}, {0: {0: 13.0}, 1: {0: 8.0}}),
        ({0: {0: 7.0}, 1: {0: 16.0}}, {0: {0: 14.0}, 1: {0: 6.0}}),
        ({0: {0: 10.0}, 1: {0: 18.0}}, {0: {0: 16.0}, 1: {0: 9.0}}),
    ]

    for job_id, (op0_times, op1_times) in enumerate(processing_options):
        instance.add_job(
            Job(
                job_id=job_id,
                due_date=10_000.0,
                weight=1.0,
                operations=[
                    Operation(
                        job_id=job_id,
                        op_id=0,
                        processing_times=op0_times,
                        eligible_machines={0, 1},
                        eligible_workers={0, 1},
                    ),
                    Operation(
                        job_id=job_id,
                        op_id=1,
                        processing_times=op1_times,
                        eligible_machines={0, 1},
                        eligible_workers={0, 1},
                    ),
                ],
            )
        )
        instance.ergonomic_risk_map[(job_id, 0)] = 0.0
        instance.ergonomic_risk_map[(job_id, 1)] = 0.0

    return instance


def _build_masked_policy_instance() -> SFJSSPInstance:
    instance = SFJSSPInstance(instance_id="PPO_MASK_TEST")
    for machine_id in range(3):
        instance.add_machine(Machine(machine_id=machine_id, modes=[MachineMode(mode_id=0)]))
    for worker_id in range(3):
        instance.add_worker(
            Worker(
                worker_id=worker_id,
                min_rest_fraction=0.0,
                ocra_max_per_shift=999.0,
            )
        )

    job_specs = [
        ({1}, {2}),
        ({2}, {1}),
        ({0}, {0}),
    ]
    for job_id, (eligible_machines, eligible_workers) in enumerate(job_specs):
        processing_times = {
            machine_id: {0: 5.0 + job_id}
            for machine_id in eligible_machines
        }
        instance.add_job(
            Job(
                job_id=job_id,
                arrival_time=0.0,
                operations=[
                    Operation(
                        job_id=job_id,
                        op_id=0,
                        processing_times=processing_times,
                        eligible_machines=eligible_machines,
                        eligible_workers=eligible_workers,
                    )
                ],
            )
        )
        instance.ergonomic_risk_map[(job_id, 0)] = 0.0

    return instance


def _build_tardiness_repair_instance() -> SFJSSPInstance:
    instance = SFJSSPInstance(instance_id="NSGA_REPAIR_TEST")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(
        Worker(
            worker_id=0,
            min_rest_fraction=0.0,
            ocra_max_per_shift=999.0,
        )
    )
    due_dates = {0: 3.0, 1: 100.0, 2: 100.0}
    for job_id in range(3):
        instance.add_job(
            Job(
                job_id=job_id,
                arrival_time=0.0,
                due_date=due_dates[job_id],
                weight=1.0,
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
    return instance


def _build_tardy_repair_genome() -> dict:
    return {
        "sequence": np.array([1, 0, 2], dtype=int),
        "machines": np.array([0, 0, 0], dtype=int),
        "workers": np.array([0, 0, 0], dtype=int),
        "modes": np.array([0, 0, 0], dtype=int),
        "offsets": np.array([0, 0, 0], dtype=int),
        "op_list": [(0, 0), (1, 0), (2, 0)],
    }


def _build_assignment_audit_instance() -> SFJSSPInstance:
    instance = SFJSSPInstance(instance_id="REPRESENTATION_AUDIT_ASSIGNMENT")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_machine(Machine(machine_id=1, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(
        Worker(
            worker_id=0,
            min_rest_fraction=0.0,
            ocra_max_per_shift=999.0,
        )
    )
    instance.add_job(
        Job(
            job_id=0,
            arrival_time=0.0,
            due_date=6.0,
            weight=1.0,
            operations=[
                Operation(
                    job_id=0,
                    op_id=0,
                    processing_times={0: {0: 20.0}, 1: {0: 5.0}},
                    eligible_machines={0, 1},
                    eligible_workers={0},
                )
            ],
        )
    )
    instance.ergonomic_risk_map[(0, 0)] = 0.0
    return instance


def _build_worker_rest_gap_instance() -> SFJSSPInstance:
    instance = SFJSSPInstance(instance_id="ENV_WORKER_REST_GAP")
    instance.add_machine(Machine(machine_id=0, setup_time=0.0, modes=[MachineMode(mode_id=0)]))
    instance.add_machine(Machine(machine_id=1, setup_time=15.0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(
        Worker(
            worker_id=0,
            min_rest_fraction=0.0,
            ocra_max_per_shift=999.0,
        )
    )
    instance.add_job(
        Job(
            job_id=0,
            arrival_time=0.0,
            operations=[
                Operation(
                    job_id=0,
                    op_id=0,
                    processing_times={0: {0: 5.0}},
                    eligible_machines={0},
                    eligible_workers={0},
                )
            ],
        )
    )
    instance.add_job(
        Job(
            job_id=1,
            arrival_time=0.0,
            operations=[
                Operation(
                    job_id=1,
                    op_id=0,
                    processing_times={1: {0: 5.0}},
                    eligible_machines={1},
                    eligible_workers={0},
                )
            ],
        )
    )
    instance.ergonomic_risk_map[(0, 0)] = 0.0
    instance.ergonomic_risk_map[(1, 0)] = 0.0
    return instance


def _load_fresh_small_benchmark() -> SFJSSPInstance:
    benchmark_path = Path(__file__).resolve().parents[1] / "benchmarks" / "small" / "SFJSSP_small_000.json"
    return SFJSSPInstance.from_dict(json.loads(benchmark_path.read_text(encoding="utf-8")))


def _genome_signature(genome: dict) -> tuple:
    return (
        tuple(int(value) for value in genome["sequence"]),
        tuple(int(value) for value in genome["machines"]),
        tuple(int(value) for value in genome["workers"]),
        tuple(int(value) for value in genome["modes"]),
        tuple(int(value) for value in genome["offsets"]),
    )


def test_operation_serialization_preserves_period_bounds():
    op = Operation(
        job_id=1,
        op_id=2,
        processing_times={0: {0: 15.0}},
        eligible_machines={0},
        eligible_workers={0},
        period_start=120.0,
        period_end=240.0,
    )

    restored = Operation.from_dict(op.to_dict())

    assert restored.period_start == 120.0
    assert restored.period_end == 240.0


def test_instance_eligible_workers_falls_back_to_operation_eligibility():
    instance = _build_simple_instance()

    assert instance.get_eligible_workers(0, 0) == [0]


def test_env_flat_observation_matches_space():
    env = SFJSSPEnv(_build_simple_instance(), use_graph_state=False)

    obs, _ = env.reset()

    assert isinstance(obs, np.ndarray)
    assert env.observation_space.contains(obs)


def test_env_resource_mask_respects_machine_breakdown():
    instance = _build_simple_instance()
    env = SFJSSPEnv(instance, use_graph_state=False)
    env.reset()
    instance.get_machine(0).is_broken = True

    mask = env.compute_resource_mask(0)

    assert float(mask.sum()) == 0.0


def test_env_records_worker_rest_for_delayed_start():
    env = SFJSSPEnv(_build_worker_rest_gap_instance(), use_graph_state=False)
    env.reset()

    _, _, _, _, info0 = env.step(
        {
            "job_idx": 0,
            "op_idx": 0,
            "machine_idx": 0,
            "worker_idx": 0,
            "mode_idx": 0,
        }
    )
    worker = env.instance.get_worker(0)
    assert worker.total_rest_time == 0.0

    _, _, _, _, info1 = env.step(
        {
            "job_idx": 1,
            "op_idx": 0,
            "machine_idx": 1,
            "worker_idx": 0,
            "mode_idx": 0,
        }
    )

    expected_rest = (
        info1["exec_info"]["start_time"] - info0["exec_info"]["completion_time"]
    )
    expected_work = (
        info0["exec_info"]["processing_time"] + info1["exec_info"]["processing_time"]
    )

    assert expected_rest > 0.0
    assert abs(worker.total_rest_time - expected_rest) < 1e-9
    assert abs(worker.total_work_time - expected_work) < 1e-9


def test_nsga3_genome_includes_modes_and_can_score_feasible_instance():
    instance = SFJSSPInstance(instance_id="NSGA_RUNTIME")
    instance.add_machine(
        Machine(
            machine_id=0,
            modes=[MachineMode(mode_id=0), MachineMode(mode_id=1, power_multiplier=1.1)],
        )
    )
    instance.add_worker(
        Worker(
            worker_id=0,
            min_rest_fraction=0.0,
            ocra_max_per_shift=999.0,
        )
    )
    for job_id in range(2):
        instance.add_job(
            Job(
                job_id=job_id,
                operations=[
                    Operation(
                        job_id=job_id,
                        op_id=0,
                        processing_times={0: {0: 10.0, 1: 8.0}},
                        eligible_machines={0},
                        eligible_workers={0},
                    )
                ],
            )
        )
        instance.ergonomic_risk_map[(job_id, 0)] = 0.0

    genome = create_sfjssp_genome(instance, np.random.default_rng(7))
    objectives = evaluate_sfjssp_genome(instance, genome)

    assert "modes" in genome
    assert len(genome["modes"]) == len(genome["op_list"])
    assert objectives[0] < 1e8


def test_nsga3_detailed_evaluation_exposes_raw_metrics_separately():
    instance = _build_simple_instance()
    genome = {
        "sequence": np.array([0]),
        "machines": np.array([0]),
        "workers": np.array([0]),
        "modes": np.array([0]),
        "offsets": np.array([0]),
        "op_list": [(0, 0)],
    }

    details = evaluate_sfjssp_genome_detailed(instance, genome)

    assert details["is_feasible"] is True
    assert details["metrics"]["makespan"] == 10.0
    assert details["raw_objectives"][0] == 10.0
    assert details["penalized_objectives"][0] >= details["raw_objectives"][0]


def test_nsga3_evaluate_population_uses_penalized_objectives_for_legacy_policy():
    instance = _build_simple_instance()
    instance.jobs[0].due_date = 5.0
    genome = {
        "sequence": np.array([0]),
        "machines": np.array([0]),
        "workers": np.array([0]),
        "modes": np.array([0]),
        "offsets": np.array([0]),
        "op_list": [(0, 0)],
    }

    nsga3 = NSGA3(
        n_objectives=4,
        population_size=1,
        n_generations=0,
        seed=42,
        constraint_handling="legacy_scalar_penalized_objectives",
    )
    nsga3.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
    )
    population = Population(individuals=[Individual(genome=genome)], size=1)

    nsga3.evaluate_population(population, instance)
    details = evaluate_sfjssp_genome_detailed(instance, genome)

    individual = population[0]
    assert individual.objectives == details["penalized_objectives"]
    assert individual.raw_objectives == details["raw_objectives"]
    assert individual.penalized_objectives == details["penalized_objectives"]
    assert individual.metrics["makespan"] == details["metrics"]["makespan"]
    assert individual.constraint_key == (
        0.0,
        1.0,
        details["penalties"]["weighted_tardiness"],
        0.0,
    )
    assert individual.is_feasible is True


def test_nsga3_evaluate_population_uses_raw_objectives_for_constrained_policy():
    instance = _build_simple_instance()
    instance.jobs[0].due_date = 5.0
    genome = {
        "sequence": np.array([0]),
        "machines": np.array([0]),
        "workers": np.array([0]),
        "modes": np.array([0]),
        "offsets": np.array([0]),
        "op_list": [(0, 0)],
    }

    nsga3 = NSGA3(
        n_objectives=4,
        population_size=1,
        n_generations=0,
        seed=42,
        constraint_handling="feasibility_first_constrained_domination",
    )
    nsga3.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
    )
    population = Population(individuals=[Individual(genome=genome)], size=1)

    nsga3.evaluate_population(population, instance)
    details = evaluate_sfjssp_genome_detailed(instance, genome)

    individual = population[0]
    assert individual.objectives == details["raw_objectives"]
    assert individual.raw_objectives == details["raw_objectives"]
    assert individual.penalized_objectives == details["penalized_objectives"]


def test_nsga3_constraint_ranking_prefers_hard_feasible_over_better_raw_objectives():
    nsga3 = NSGA3(
        n_objectives=4,
        population_size=2,
        n_generations=0,
        seed=42,
        constraint_handling="feasibility_first_constrained_domination",
    )
    population = Population(
        individuals=[
            Individual(
                genome={},
                objectives=[100.0, 100.0, 0.4, 40.0],
                raw_objectives=[100.0, 100.0, 0.4, 40.0],
                constraint_key=(0.0, 2.0, 30.0, 0.0),
                is_feasible=True,
            ),
            Individual(
                genome={},
                objectives=[1.0, 1.0, 0.1, 1.0],
                raw_objectives=[1.0, 1.0, 0.1, 1.0],
                constraint_key=(1.0, 0.0, 0.0, 0.0),
                is_feasible=False,
            ),
        ],
        size=2,
    )

    fronts = nsga3._non_dominated_sort(population)

    assert fronts[0] == [0]
    assert population[0].rank == 0
    assert population[1].rank == 1


def test_nsga3_constraint_ranking_prefers_lower_tardiness_before_raw_objectives():
    nsga3 = NSGA3(
        n_objectives=4,
        population_size=2,
        n_generations=0,
        seed=42,
        constraint_handling="feasibility_first_lexicographic",
    )
    population = Population(
        individuals=[
            Individual(
                genome={},
                objectives=[200.0, 200.0, 0.5, 50.0],
                raw_objectives=[200.0, 200.0, 0.5, 50.0],
                constraint_key=(0.0, 1.0, 12.0, 0.0),
                is_feasible=True,
            ),
            Individual(
                genome={},
                objectives=[50.0, 50.0, 0.2, 10.0],
                raw_objectives=[50.0, 50.0, 0.2, 10.0],
                constraint_key=(0.0, 2.0, 0.0, 0.0),
                is_feasible=True,
            ),
        ],
        size=2,
    )

    fronts = nsga3._non_dominated_sort(population)

    assert fronts[0] == [0]
    assert population[0].rank == 0
    assert population[1].rank == 1


def test_nsga3_default_constraint_handling_keeps_hard_feasible_tradeoffs_nondominated():
    nsga3 = NSGA3(
        n_objectives=4,
        population_size=2,
        n_generations=0,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
    )
    population = Population(
        individuals=[
            Individual(
                genome={},
                objectives=[200.0, 50.0, 0.5, 50.0],
                raw_objectives=[200.0, 50.0, 0.5, 50.0],
                constraint_key=(0.0, 1.0, 12.0, 0.0),
                is_feasible=True,
            ),
            Individual(
                genome={},
                objectives=[50.0, 200.0, 0.2, 10.0],
                raw_objectives=[50.0, 200.0, 0.2, 10.0],
                constraint_key=(0.0, 2.0, 0.0, 0.0),
                is_feasible=True,
            ),
        ],
        size=2,
    )

    fronts = nsga3._non_dominated_sort(population)

    assert fronts[0] == [0, 1]


def test_nsga3_hybrid_policy_uses_soft_penalties_within_hard_feasible_set():
    nsga3 = NSGA3(
        n_objectives=4,
        population_size=2,
        n_generations=0,
        seed=42,
        constraint_handling="hard_feasible_first_soft_penalties",
    )
    population = Population(
        individuals=[
            Individual(
                genome={},
                objectives=[1100.0, 1100.0, 0.301, 1040.0],
                raw_objectives=[100.0, 100.0, 0.3, 40.0],
                penalized_objectives=[1100.0, 1100.0, 0.301, 1040.0],
                constraint_key=(0.0, 5.0, 100.0, 0.0),
                is_feasible=True,
            ),
            Individual(
                genome={},
                objectives=[320.0, 320.0, 0.2002, 260.0],
                raw_objectives=[120.0, 120.0, 0.2, 60.0],
                penalized_objectives=[320.0, 320.0, 0.2002, 260.0],
                constraint_key=(0.0, 1.0, 20.0, 0.0),
                is_feasible=True,
            ),
        ],
        size=2,
    )

    fronts = nsga3._non_dominated_sort(population)

    assert fronts[0] == [1]


def test_nsga3_feasible_tardiness_tournament_prefers_lower_tardiness_on_rank_tie():
    nsga3 = NSGA3(
        n_objectives=4,
        population_size=2,
        n_generations=0,
        seed=42,
        constraint_handling="hard_feasible_first_soft_penalties",
        parent_selection="feasible_tardiness_tournament",
    )
    left = Individual(
        genome={},
        objectives=[1200.0, 1200.0, 0.3012, 1120.0],
        raw_objectives=[100.0, 100.0, 0.3, 40.0],
        penalized_objectives=[1200.0, 1200.0, 0.3012, 1120.0],
        penalties={"n_tardy_jobs": 4, "weighted_tardiness": 120.0},
        metrics={"n_tardy_jobs": 4, "weighted_tardiness": 120.0},
        rank=0,
        constraint_key=(0.0, 4.0, 120.0, 0.0),
        is_feasible=True,
        makespan=100.0,
    )
    right = Individual(
        genome={},
        objectives=[330.0, 330.0, 0.2002, 270.0],
        raw_objectives=[130.0, 130.0, 0.2, 70.0],
        penalized_objectives=[330.0, 330.0, 0.2002, 270.0],
        penalties={"n_tardy_jobs": 1, "weighted_tardiness": 20.0},
        metrics={"n_tardy_jobs": 1, "weighted_tardiness": 20.0},
        rank=0,
        constraint_key=(0.0, 1.0, 20.0, 0.0),
        is_feasible=True,
        makespan=130.0,
    )

    preferred = nsga3._prefer_mating_parent(left, right)

    assert preferred is right


def test_nsga3_detailed_evaluation_advances_worker_availability():
    instance = SFJSSPInstance(instance_id="NSGA_WORKER_STATE")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_machine(Machine(machine_id=1, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(
        Worker(
            worker_id=0,
            min_rest_fraction=0.0,
            ocra_max_per_shift=999.0,
        )
    )
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
    instance.add_job(
        Job(
            job_id=1,
            operations=[
                Operation(
                    job_id=1,
                    op_id=0,
                    processing_times={1: {0: 7.0}},
                    eligible_machines={1},
                    eligible_workers={0},
                )
            ],
        )
    )
    instance.ergonomic_risk_map[(0, 0)] = 0.0
    instance.ergonomic_risk_map[(1, 0)] = 0.0

    genome = {
        "sequence": np.array([0, 1]),
        "machines": np.array([0, 1]),
        "workers": np.array([0, 0]),
        "modes": np.array([0, 0]),
        "offsets": np.array([0, 0]),
        "op_list": [(0, 0), (1, 0)],
    }

    details = evaluate_sfjssp_genome_detailed(instance, genome)
    schedule = details["schedule"]

    assert details["is_feasible"] is True
    assert schedule is not None
    assert schedule.get_operation(1, 0).start_time >= schedule.get_operation(0, 0).completion_time
    assert not any("Worker overlap" in violation for violation in details["constraint_violations"])


def test_nsga3_seed_genomes_are_hard_feasible():
    instance = _build_seedable_nsga_instance()

    seed_genomes = create_sfjssp_seed_genomes(instance)

    assert seed_genomes

    for genome in seed_genomes:
        details = evaluate_sfjssp_genome_detailed(_build_seedable_nsga_instance(), genome)
        assert details["is_feasible"] is True
        assert details["penalties"]["hard_violations"] == 0


def test_schedule_to_sfjssp_genome_preserves_dispatch_order():
    instance = _build_simple_instance()
    instance.add_job(
        Job(
            job_id=1,
            arrival_time=0.0,
            operations=[
                Operation(
                    job_id=1,
                    op_id=0,
                    processing_times={0: {0: 5.0}},
                    eligible_machines={0},
                    eligible_workers={0},
                )
            ],
        )
    )
    instance.ergonomic_risk_map[(1, 0)] = 0.0

    schedule = Schedule(instance_id="DISPATCH_ORDER")
    schedule.add_operation(1, 0, 0, 0, 0, 100.0, 105.0, 5.0)
    schedule.add_operation(0, 0, 0, 0, 0, 0.0, 10.0, 10.0)

    genome = schedule_to_sfjssp_genome(instance, schedule)

    assert list(genome["sequence"]) == [1, 0]


def test_seed_generation_diagnostics_capture_rule_outcomes():
    accepted, diagnostics = _collect_seed_genome_candidates(_build_seedable_nsga_instance())

    assert diagnostics
    assert len(diagnostics) >= len(accepted)
    assert all(diag["benchmark_id"] == "NSGA_SEED_TEST" for diag in diagnostics)
    assert all(diag["source_rule"] for diag in diagnostics)
    assert all(diag["status"] in {"accepted", "rejected"} for diag in diagnostics)
    assert {"least_slack_rule", "critical_ratio_rule", "tardiness_composite_rule"}.issubset(
        {diag["source_rule"] for diag in diagnostics}
    )
    for diag in diagnostics:
        if diag["status"] == "rejected":
            assert diag["rejection_reason"]


def test_nsga3_initialize_population_places_seed_genomes_first():
    instance = _build_seedable_nsga_instance()
    seed_genomes = create_sfjssp_seed_genomes(instance)

    assert len(seed_genomes) >= 2

    nsga3 = NSGA3(
        n_objectives=4,
        population_size=6,
        n_generations=1,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
    )
    nsga3.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
        seed_individuals_fn=lambda _instance, seeds=seed_genomes[:2]: seeds,
    )

    population = nsga3.initialize_population(instance)

    for idx, expected in enumerate(seed_genomes[:2]):
        actual = population[idx].genome
        assert np.array_equal(actual["sequence"], expected["sequence"])
        assert np.array_equal(actual["machines"], expected["machines"])
        assert np.array_equal(actual["workers"], expected["workers"])
        assert np.array_equal(actual["modes"], expected["modes"])


def test_nsga3_warm_start_initial_population_is_not_worse_than_random():
    instance = _build_seedable_nsga_instance()

    seeded = NSGA3(
        n_objectives=4,
        population_size=10,
        n_generations=0,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
    )
    seeded.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
        seed_individuals_fn=create_sfjssp_seed_genomes,
    )
    seeded_population = seeded.evolve(instance, verbose=False)
    seeded_best = min(ind.makespan for ind in seeded_population.individuals)

    random_only = NSGA3(
        n_objectives=4,
        population_size=10,
        n_generations=0,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
    )
    random_only.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
    )
    random_population = random_only.evolve(_build_seedable_nsga_instance(), verbose=False)
    random_best = min(ind.makespan for ind in random_population.individuals)

    assert seeded_best <= random_best


def test_seed_generation_does_not_mutate_original_instance(small_benchmark_instance):
    original_state = small_benchmark_instance.to_dict()

    create_sfjssp_seed_genomes(small_benchmark_instance)

    assert small_benchmark_instance.to_dict() == original_state


def test_seed_generation_returns_valid_seed_on_stored_small_benchmark():
    seed_genomes = create_sfjssp_seed_genomes(_load_fresh_small_benchmark())

    assert seed_genomes
    assert len(seed_genomes) >= 1


def test_seed_generation_is_deterministic_on_stored_small_benchmark():
    first = create_sfjssp_seed_genomes(_load_fresh_small_benchmark())
    second = create_sfjssp_seed_genomes(_load_fresh_small_benchmark())

    assert [_genome_signature(genome) for genome in first] == [
        _genome_signature(genome) for genome in second
    ]


def test_seed_generation_accepts_tardiness_biased_variants_on_stored_small_benchmark():
    accepted, diagnostics = _collect_seed_genome_candidates(_load_fresh_small_benchmark())

    accepted_variant_diags = [
        diag
        for diag in diagnostics
        if diag["status"] == "accepted" and "::" in diag["source_rule"]
    ]

    assert accepted_variant_diags
    assert len(accepted) >= len(accepted_variant_diags)
    assert all(diag["penalties"]["hard_violations"] == 0 for diag in accepted_variant_diags)
    assert all(diag["metrics"]["weighted_tardiness"] is not None for diag in accepted_variant_diags)


def test_tardiness_sequence_repair_improves_crafted_tardy_instance():
    instance = _build_tardiness_repair_instance()
    genome = _build_tardy_repair_genome()
    base_details = evaluate_sfjssp_genome_detailed(instance, genome)

    assert base_details["is_feasible"] is True
    assert base_details["metrics"]["n_tardy_jobs"] == 1

    repaired_genome, repaired_details, diagnostics = _attempt_tardiness_sequence_repair(
        instance=instance,
        genome=genome,
        base_details=base_details,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
    )

    assert repaired_genome is not None
    assert repaired_details is not None
    assert diagnostics["repair_accepted_children"] == 1
    assert diagnostics["repair_improved_n_tardy_jobs_count"] == 1
    assert diagnostics["repair_improved_weighted_tardiness_count"] == 1
    assert repaired_details["metrics"]["n_tardy_jobs"] == 0
    assert repaired_details["metrics"]["weighted_tardiness"] == 0.0
    assert tuple(int(value) for value in repaired_genome["sequence"]) == (0, 1, 2)
    assert tuple(int(value) for value in genome["sequence"]) == (1, 0, 2)


def test_tardiness_sequence_repair_is_deterministic():
    first_instance = _build_tardiness_repair_instance()
    second_instance = _build_tardiness_repair_instance()
    base_genome = _build_tardy_repair_genome()
    first_details = evaluate_sfjssp_genome_detailed(first_instance, base_genome)
    second_details = evaluate_sfjssp_genome_detailed(second_instance, _build_tardy_repair_genome())

    first_genome, first_repaired, first_diag = _attempt_tardiness_sequence_repair(
        instance=first_instance,
        genome=base_genome,
        base_details=first_details,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
    )
    second_genome, second_repaired, second_diag = _attempt_tardiness_sequence_repair(
        instance=second_instance,
        genome=_build_tardy_repair_genome(),
        base_details=second_details,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
    )

    assert first_genome is not None and second_genome is not None
    assert tuple(int(value) for value in first_genome["sequence"]) == tuple(
        int(value) for value in second_genome["sequence"]
    )
    assert first_repaired["metrics"]["weighted_tardiness"] == second_repaired["metrics"]["weighted_tardiness"]
    assert first_diag == second_diag


def test_tardiness_sequence_repair_rejects_over_cap_without_mutating_original():
    instance = _build_tardiness_repair_instance()
    genome = _build_tardy_repair_genome()
    base_sequence = tuple(int(value) for value in genome["sequence"])
    base_details = evaluate_sfjssp_genome_detailed(instance, genome)

    repaired_genome, repaired_details, diagnostics = _attempt_tardiness_sequence_repair(
        instance=instance,
        genome=genome,
        base_details=base_details,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        makespan_regression_cap=-0.01,
    )

    assert repaired_genome is None
    assert repaired_details is None
    assert diagnostics["repair_accepted_children"] == 0
    assert diagnostics["repair_rejected_due_to_makespan_cap_count"] >= 1
    assert tuple(int(value) for value in genome["sequence"]) == base_sequence


def test_urgent_pull_forward_mutation_moves_urgent_job_forward():
    instance = _build_tardiness_repair_instance()
    genome = _build_tardy_repair_genome()

    variant, distance = _apply_urgent_pull_forward_sequence_mutation(instance, genome)

    assert variant is not None
    assert distance == 1
    assert tuple(int(value) for value in variant["sequence"]) == (0, 1, 2)


def test_urgent_pull_forward_mutation_preserves_job_multiset_and_resource_arrays():
    instance = _build_tardiness_repair_instance()
    genome = _build_tardy_repair_genome()
    original_machines = tuple(int(value) for value in genome["machines"])
    original_workers = tuple(int(value) for value in genome["workers"])
    original_modes = tuple(int(value) for value in genome["modes"])
    original_offsets = tuple(int(value) for value in genome["offsets"])

    variant, _ = _apply_urgent_pull_forward_sequence_mutation(instance, genome)

    assert variant is not None
    assert sorted(int(value) for value in variant["sequence"]) == sorted(int(value) for value in genome["sequence"])
    assert tuple(int(value) for value in variant["machines"]) == original_machines
    assert tuple(int(value) for value in variant["workers"]) == original_workers
    assert tuple(int(value) for value in variant["modes"]) == original_modes
    assert tuple(int(value) for value in variant["offsets"]) == original_offsets


def test_urgent_pull_forward_mutation_noops_when_no_better_move_exists():
    instance = SFJSSPInstance(instance_id="NSGA_MUTATION_NOOP")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(
        Worker(
            worker_id=0,
            min_rest_fraction=0.0,
            ocra_max_per_shift=999.0,
        )
    )
    for job_id in range(3):
        instance.add_job(
            Job(
                job_id=job_id,
                arrival_time=0.0,
                due_date=100.0,
                weight=1.0,
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
    already_urgent = {
        "sequence": np.array([0, 1, 2], dtype=int),
        "machines": np.array([0, 0, 0], dtype=int),
        "workers": np.array([0, 0, 0], dtype=int),
        "modes": np.array([0, 0, 0], dtype=int),
        "offsets": np.array([0, 0, 0], dtype=int),
        "op_list": [(0, 0), (1, 0), (2, 0)],
    }

    variant, distance = _apply_urgent_pull_forward_sequence_mutation(
        instance,
        already_urgent,
        max_window=0,
    )

    assert variant is None
    assert distance == 0
    assert tuple(int(value) for value in already_urgent["sequence"]) == (0, 1, 2)


def test_nsga3_sequence_mutation_legacy_random_swap_keeps_default_stable():
    instance = _build_seedable_nsga_instance()

    baseline = NSGA3(
        n_objectives=4,
        population_size=10,
        n_generations=1,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        parent_selection="random_pairing",
        local_improvement=NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        sequence_mutation=NSGA3_DEFAULT_SEQUENCE_MUTATION,
    )
    baseline.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
        seed_individuals_fn=create_sfjssp_seed_genomes,
    )
    baseline_population = baseline.evolve(instance, verbose=False)

    explicit_legacy = NSGA3(
        n_objectives=4,
        population_size=10,
        n_generations=1,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        parent_selection="random_pairing",
        local_improvement=NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        sequence_mutation="legacy_random_swap",
    )
    explicit_legacy.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
        seed_individuals_fn=create_sfjssp_seed_genomes,
    )
    explicit_population = explicit_legacy.evolve(_build_seedable_nsga_instance(), verbose=False)

    baseline_best = min(ind.makespan for ind in baseline_population.individuals)
    explicit_best = min(ind.makespan for ind in explicit_population.individuals)

    assert baseline_best == explicit_best
    assert explicit_legacy.last_run_diagnostics["sequence_mutation"] == "legacy_random_swap"
    assert explicit_legacy.last_run_diagnostics["urgent_sequence_mutation_attempts"] == 0


def test_nsga3_crossover_policy_legacy_keeps_default_stable():
    instance = _build_seedable_nsga_instance()

    baseline = NSGA3(
        n_objectives=4,
        population_size=10,
        n_generations=1,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        parent_selection="random_pairing",
        local_improvement=NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        sequence_mutation=NSGA3_DEFAULT_SEQUENCE_MUTATION,
        immigrant_policy=NSGA3_DEFAULT_IMMIGRANT_POLICY,
    )
    baseline.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
        seed_individuals_fn=create_sfjssp_seed_genomes,
    )
    baseline_population = baseline.evolve(instance, verbose=False)

    explicit_legacy = NSGA3(
        n_objectives=4,
        population_size=10,
        n_generations=1,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        parent_selection="random_pairing",
        crossover_policy="legacy_pox_uniform_assignments",
        local_improvement=NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        sequence_mutation=NSGA3_DEFAULT_SEQUENCE_MUTATION,
        immigrant_policy=NSGA3_DEFAULT_IMMIGRANT_POLICY,
    )
    explicit_legacy.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
        seed_individuals_fn=create_sfjssp_seed_genomes,
    )
    explicit_population = explicit_legacy.evolve(_build_seedable_nsga_instance(), verbose=False)

    baseline_best = min(ind.makespan for ind in baseline_population.individuals)
    explicit_best = min(ind.makespan for ind in explicit_population.individuals)

    assert baseline_best == explicit_best
    assert explicit_legacy.last_run_diagnostics["crossover_policy"] == "legacy_pox_uniform_assignments"
    assert explicit_legacy.last_run_diagnostics["urgent_crossover_attempts"] == 0


def test_urgent_prefix_merge_builds_expected_prefix_without_mutating_parent_resources():
    instance = SFJSSPInstance(instance_id="URGENT_PREFIX_TEST")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(Worker(worker_id=0, min_rest_fraction=0.0, ocra_max_per_shift=999.0))

    for job_id, due_date in enumerate((4.0, 8.0, 12.0, 16.0)):
        instance.add_job(
            Job(
                job_id=job_id,
                due_date=due_date,
                operations=[
                    Operation(
                        job_id=job_id,
                        op_id=0,
                        processing_times={0: {0: 1.0}},
                        eligible_machines={0},
                        eligible_workers={0},
                    )
                ],
            )
        )
        instance.ergonomic_risk_map[(job_id, 0)] = 0.0

    urgent_parent = {
        "sequence": np.array([3, 2, 1, 0], dtype=int),
        "machines": np.array([0, 0, 0, 0], dtype=int),
        "workers": np.array([0, 0, 0, 0], dtype=int),
        "modes": np.array([0, 0, 0, 0], dtype=int),
        "offsets": np.array([0, 0, 0, 0], dtype=int),
        "op_list": [(0, 0), (1, 0), (2, 0), (3, 0)],
    }
    support_parent = {
        "sequence": np.array([0, 1, 2, 3], dtype=int),
        "machines": np.array([0, 0, 0, 0], dtype=int),
        "workers": np.array([0, 0, 0, 0], dtype=int),
        "modes": np.array([0, 0, 0, 0], dtype=int),
        "offsets": np.array([0, 0, 0, 0], dtype=int),
        "op_list": [(0, 0), (1, 0), (2, 0), (3, 0)],
    }
    original_machine_assignments = tuple(int(value) for value in urgent_parent["machines"])

    child_sequence, prefix_size = _build_urgent_prefix_merged_sequence(
        instance,
        urgent_genome=urgent_parent,
        support_genome=support_parent,
    )

    assert child_sequence is not None
    assert prefix_size == 2
    assert tuple(int(value) for value in child_sequence[:2]) == (1, 0)
    assert sorted(int(value) for value in child_sequence) == [0, 1, 2, 3]
    assert tuple(int(value) for value in urgent_parent["machines"]) == original_machine_assignments
    assert tuple(int(value) for value in urgent_parent["sequence"]) == (3, 2, 1, 0)


def test_urgent_prefix_merge_returns_none_when_parent_sequence_already_matches_prefix():
    instance = SFJSSPInstance(instance_id="URGENT_PREFIX_NOOP")
    instance.add_machine(Machine(machine_id=0, modes=[MachineMode(mode_id=0)]))
    instance.add_worker(Worker(worker_id=0, min_rest_fraction=0.0, ocra_max_per_shift=999.0))

    for job_id, due_date in enumerate((4.0, 8.0, 12.0)):
        instance.add_job(
            Job(
                job_id=job_id,
                due_date=due_date,
                operations=[
                    Operation(
                        job_id=job_id,
                        op_id=0,
                        processing_times={0: {0: 1.0}},
                        eligible_machines={0},
                        eligible_workers={0},
                    )
                ],
            )
        )
        instance.ergonomic_risk_map[(job_id, 0)] = 0.0

    already_urgent = {
        "sequence": np.array([0, 1, 2], dtype=int),
        "machines": np.array([0, 0, 0], dtype=int),
        "workers": np.array([0, 0, 0], dtype=int),
        "modes": np.array([0, 0, 0], dtype=int),
        "offsets": np.array([0, 0, 0], dtype=int),
        "op_list": [(0, 0), (1, 0), (2, 0)],
    }

    child_sequence, prefix_size = _build_urgent_prefix_merged_sequence(
        instance,
        urgent_genome=already_urgent,
        support_genome=already_urgent,
    )

    assert child_sequence is None
    assert prefix_size == 0


def test_nsga3_urgent_prefix_merge_exposes_lineage_diagnostics():
    instance = _build_seedable_nsga_instance()
    nsga3 = NSGA3(
        n_objectives=4,
        population_size=10,
        n_generations=1,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        parent_selection="random_pairing",
        crossover_policy="urgent_prefix_merge",
        local_improvement=NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        sequence_mutation=NSGA3_DEFAULT_SEQUENCE_MUTATION,
        immigrant_policy=NSGA3_DEFAULT_IMMIGRANT_POLICY,
    )
    nsga3.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
        seed_individuals_fn=create_sfjssp_seed_genomes,
    )

    nsga3.evolve(instance, verbose=False, record_history=True, include_population_records=True)

    assert nsga3.last_run_diagnostics["crossover_policy"] == "urgent_prefix_merge"
    assert nsga3.last_run_diagnostics["children_evaluated_count"] > 0
    assert nsga3.last_run_diagnostics["children_hard_feasible_count"] >= 0
    assert nsga3.last_run_diagnostics["children_improve_both_parents_weighted_tardiness_count"] >= 0
    assert nsga3.last_run_diagnostics["urgent_crossover_attempts"] >= 0
    assert nsga3.history[1]["crossover_policy"] == "urgent_prefix_merge"
    assert nsga3.history[1]["children_evaluated_count"] > 0


def test_nsga3_local_improvement_none_keeps_legacy_run_stable():
    instance = _build_seedable_nsga_instance()

    baseline = NSGA3(
        n_objectives=4,
        population_size=10,
        n_generations=1,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        parent_selection="random_pairing",
        local_improvement=NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
    )
    baseline.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
        seed_individuals_fn=create_sfjssp_seed_genomes,
    )
    baseline_population = baseline.evolve(instance, verbose=False)

    explicit_none = NSGA3(
        n_objectives=4,
        population_size=10,
        n_generations=1,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        parent_selection="random_pairing",
        local_improvement="none",
    )
    explicit_none.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
        seed_individuals_fn=create_sfjssp_seed_genomes,
    )
    explicit_population = explicit_none.evolve(_build_seedable_nsga_instance(), verbose=False)

    baseline_best = min(ind.makespan for ind in baseline_population.individuals)
    explicit_best = min(ind.makespan for ind in explicit_population.individuals)

    assert baseline_best == explicit_best
    assert explicit_none.last_run_diagnostics["repair_attempted_children"] == 0


def test_nsga3_immigrant_policy_none_keeps_default_stable():
    instance = _build_seedable_nsga_instance()

    baseline = NSGA3(
        n_objectives=4,
        population_size=10,
        n_generations=1,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        parent_selection="random_pairing",
        local_improvement=NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        sequence_mutation=NSGA3_DEFAULT_SEQUENCE_MUTATION,
        immigrant_policy=NSGA3_DEFAULT_IMMIGRANT_POLICY,
    )
    baseline.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
        seed_individuals_fn=create_sfjssp_seed_genomes,
    )
    baseline_population = baseline.evolve(instance, verbose=False)

    explicit_none = NSGA3(
        n_objectives=4,
        population_size=10,
        n_generations=1,
        mutation_rate=0.2,
        crossover_rate=0.9,
        seed=42,
        constraint_handling=NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        parent_selection="random_pairing",
        local_improvement=NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        sequence_mutation=NSGA3_DEFAULT_SEQUENCE_MUTATION,
        immigrant_policy="none",
    )
    explicit_none.set_problem(
        evaluate_fn=evaluate_sfjssp_genome,
        evaluate_details_fn=evaluate_sfjssp_genome_detailed,
        create_individual_fn=create_sfjssp_genome,
        seed_individuals_fn=create_sfjssp_seed_genomes,
    )
    explicit_population = explicit_none.evolve(_build_seedable_nsga_instance(), verbose=False)

    baseline_best = min(ind.makespan for ind in baseline_population.individuals)
    explicit_best = min(ind.makespan for ind in explicit_population.individuals)

    assert baseline_best == explicit_best
    assert explicit_none.last_run_diagnostics["immigrant_policy"] == "none"
    assert explicit_none.last_run_diagnostics["immigrant_injected_individuals"] == 0


def test_feasible_tardiness_archive_accepts_only_hard_feasible_members_and_deduplicates():
    instance = _build_seedable_nsga_instance()
    genome = create_sfjssp_seed_genomes(instance)[0]

    def clone_genome(source):
        return {
            "sequence": np.array(source["sequence"], copy=True),
            "machines": np.array(source["machines"], copy=True),
            "workers": np.array(source["workers"], copy=True),
            "modes": np.array(source["modes"], copy=True),
            "offsets": np.array(source["offsets"], copy=True),
            "op_list": list(source["op_list"]),
        }

    nsga3 = NSGA3(
        n_objectives=4,
        population_size=4,
        n_generations=1,
        seed=42,
        immigrant_policy="feasible_tardiness_archive",
        immigrant_count=1,
        immigrant_period=1,
        immigrant_archive_size=2,
    )

    feasible = Individual(genome=clone_genome(genome))
    nsga3._assign_evaluation(feasible, evaluate_sfjssp_genome_detailed(instance, feasible.genome))

    duplicate = Individual(genome=clone_genome(genome))
    nsga3._assign_evaluation(duplicate, evaluate_sfjssp_genome_detailed(instance, duplicate.genome))

    infeasible = Individual(genome=clone_genome(genome))
    infeasible_details = evaluate_sfjssp_genome_detailed(instance, infeasible.genome)
    infeasible_details["is_feasible"] = False
    infeasible_details["penalties"] = dict(infeasible_details.get("penalties") or {})
    infeasible_details["penalties"]["hard_violations"] = 1.0
    infeasible_details["constraint_violations"] = ["forced hard violation"]
    nsga3._assign_evaluation(infeasible, infeasible_details)

    archive = {}
    stats = nsga3._update_immigrant_archive(
        archive,
        candidates=[feasible, duplicate, infeasible],
        source_generation=1,
    )

    assert len(archive) == 1
    assert stats["immigrant_archive_admission_count"] == 1
    assert next(iter(archive.values())).score[0] >= 0.0


def test_feasible_tardiness_archive_injection_clones_without_mutating_archive():
    instance = _build_seedable_nsga_instance()
    genome = create_sfjssp_seed_genomes(instance)[0]

    def clone_genome(source):
        return {
            "sequence": np.array(source["sequence"], copy=True),
            "machines": np.array(source["machines"], copy=True),
            "workers": np.array(source["workers"], copy=True),
            "modes": np.array(source["modes"], copy=True),
            "offsets": np.array(source["offsets"], copy=True),
            "op_list": list(source["op_list"]),
        }

    nsga3 = NSGA3(
        n_objectives=4,
        population_size=4,
        n_generations=1,
        seed=42,
        immigrant_policy="feasible_tardiness_archive",
        immigrant_count=1,
        immigrant_period=1,
        immigrant_archive_size=2,
    )

    archived = Individual(genome=clone_genome(genome))
    nsga3._assign_evaluation(archived, evaluate_sfjssp_genome_detailed(instance, archived.genome))
    archive = {}
    nsga3._update_immigrant_archive(archive, candidates=[archived], source_generation=1)

    population = Population(size=1)
    offspring = Population(size=1)
    injected, stats = nsga3._select_immigrant_injections(archive, population, offspring)

    assert len(injected) == 1
    assert stats["immigrant_injected_individuals"] == 1
    stored_sequence = tuple(int(value) for value in next(iter(archive.values())).individual.genome["sequence"])

    injected[0].genome["sequence"][0] = int(injected[0].genome["sequence"][0]) + 99

    assert tuple(int(value) for value in next(iter(archive.values())).individual.genome["sequence"]) == stored_sequence


def test_feasible_tardiness_archive_skips_duplicates_already_in_population():
    instance = _build_seedable_nsga_instance()
    genome = create_sfjssp_seed_genomes(instance)[0]
    clone = {
        "sequence": np.array(genome["sequence"], copy=True),
        "machines": np.array(genome["machines"], copy=True),
        "workers": np.array(genome["workers"], copy=True),
        "modes": np.array(genome["modes"], copy=True),
        "offsets": np.array(genome["offsets"], copy=True),
        "op_list": list(genome["op_list"]),
    }

    nsga3 = NSGA3(
        n_objectives=4,
        population_size=4,
        n_generations=1,
        seed=42,
        immigrant_policy="feasible_tardiness_archive",
        immigrant_count=1,
        immigrant_period=1,
        immigrant_archive_size=2,
    )

    archived = Individual(genome=clone)
    nsga3._assign_evaluation(archived, evaluate_sfjssp_genome_detailed(instance, archived.genome))
    archive = {}
    nsga3._update_immigrant_archive(archive, candidates=[archived], source_generation=1)

    population = Population(individuals=[archived], size=1)
    offspring = Population(size=0)
    injected, stats = nsga3._select_immigrant_injections(archive, population, offspring)

    assert injected == []
    assert stats["immigrant_injected_individuals"] == 0
    assert stats["immigrant_skipped_duplicate_count"] >= 1


def test_feasible_tardiness_archive_injection_is_deterministic():
    instance = _build_seedable_nsga_instance()
    genome = create_sfjssp_seed_genomes(instance)[0]

    def build_archive_and_population():
        nsga3 = NSGA3(
            n_objectives=4,
            population_size=4,
            n_generations=1,
            seed=42,
            immigrant_policy="feasible_tardiness_archive",
            immigrant_count=1,
            immigrant_period=1,
            immigrant_archive_size=2,
        )
        archived = Individual(
            genome={
                "sequence": np.array(genome["sequence"], copy=True),
                "machines": np.array(genome["machines"], copy=True),
                "workers": np.array(genome["workers"], copy=True),
                "modes": np.array(genome["modes"], copy=True),
                "offsets": np.array(genome["offsets"], copy=True),
                "op_list": list(genome["op_list"]),
            }
        )
        nsga3._assign_evaluation(archived, evaluate_sfjssp_genome_detailed(instance, archived.genome))
        archive = {}
        nsga3._update_immigrant_archive(archive, candidates=[archived], source_generation=1)
        return nsga3, archive

    first_nsga3, first_archive = build_archive_and_population()
    second_nsga3, second_archive = build_archive_and_population()
    first_injected, _ = first_nsga3._select_immigrant_injections(first_archive, Population(size=0), Population(size=0))
    second_injected, _ = second_nsga3._select_immigrant_injections(second_archive, Population(size=0), Population(size=0))

    assert len(first_injected) == len(second_injected) == 1
    assert tuple(int(value) for value in first_injected[0].genome["sequence"]) == tuple(
        int(value) for value in second_injected[0].genome["sequence"]
    )


def test_representation_audit_best_seed_reconstruction_is_deterministic():
    instance = _build_seedable_nsga_instance()

    first = _best_generation0_seed(instance)
    second = _best_generation0_seed(instance)

    assert first["best_source_rule"] == second["best_source_rule"]
    assert tuple(int(value) for value in first["best_genome"]["sequence"]) == tuple(
        int(value) for value in second["best_genome"]["sequence"]
    )
    assert first["best_details"]["metrics"]["weighted_tardiness"] == second["best_details"]["metrics"]["weighted_tardiness"]


def test_representation_audit_sequence_family_preserves_resource_arrays():
    instance = _build_seedable_nsga_instance()
    seed_info = _best_generation0_seed(instance)
    seed_genome = seed_info["best_genome"]

    candidates = _build_sequence_family_candidates(instance, seed_genome, seed_info["best_details"])

    assert candidates
    for candidate in candidates:
        assert np.array_equal(candidate.genome["machines"], seed_genome["machines"])
        assert np.array_equal(candidate.genome["workers"], seed_genome["workers"])
        assert np.array_equal(candidate.genome["modes"], seed_genome["modes"])
        assert np.array_equal(candidate.genome["offsets"], seed_genome["offsets"])


def test_representation_audit_assignment_family_preserves_sequence_and_offsets():
    instance = _build_seedable_nsga_instance()
    seed_info = _best_generation0_seed(instance)
    seed_genome = seed_info["best_genome"]

    candidates = _build_assignment_family_candidates(instance, seed_genome, seed_info["best_details"])

    assert candidates
    for candidate in candidates:
        assert np.array_equal(candidate.genome["sequence"], seed_genome["sequence"])
        assert np.array_equal(candidate.genome["offsets"], seed_genome["offsets"])


def test_representation_audit_offset_family_preserves_sequence_and_assignments():
    instance = _build_seedable_nsga_instance()
    seed_info = _best_generation0_seed(instance)
    seed_genome = seed_info["best_genome"]

    candidates = _build_offset_family_candidates(instance, seed_genome, seed_info["best_details"])

    assert candidates
    for candidate in candidates:
        assert np.array_equal(candidate.genome["sequence"], seed_genome["sequence"])
        assert np.array_equal(candidate.genome["machines"], seed_genome["machines"])
        assert np.array_equal(candidate.genome["workers"], seed_genome["workers"])
        assert np.array_equal(candidate.genome["modes"], seed_genome["modes"])


def test_representation_audit_mixed_family_stays_within_bounded_candidate_count():
    instance = _build_seedable_nsga_instance()
    seed_info = _best_generation0_seed(instance)
    seed_genome = seed_info["best_genome"]

    sequence_candidates = _build_sequence_family_candidates(instance, seed_genome, seed_info["best_details"])
    assignment_candidates = _build_assignment_family_candidates(instance, seed_genome, seed_info["best_details"])
    mixed_candidates = _build_mixed_family_candidates(
        instance,
        seed_genome,
        sequence_candidates,
        assignment_candidates,
    )

    assert len(mixed_candidates) <= 40


def test_representation_audit_collision_accounting_is_deterministic():
    instance = _build_seedable_nsga_instance()
    seed_info = _best_generation0_seed(instance)
    seed_genome = seed_info["best_genome"]
    candidates = _build_sequence_family_candidates(instance, seed_genome, seed_info["best_details"])

    first = _summarize_family("sequence_only", seed_genome, seed_info["best_details"], candidates)
    second = _summarize_family("sequence_only", seed_genome, seed_info["best_details"], candidates)

    assert first["collision"] == second["collision"]
    assert _metric_signature(seed_info["best_details"]) == _metric_signature(seed_info["best_details"])


def test_representation_audit_detects_improving_assignment_move_when_present():
    instance = _build_assignment_audit_instance()
    seed_genome = {
        "sequence": np.array([0], dtype=int),
        "machines": np.array([0], dtype=int),
        "workers": np.array([0], dtype=int),
        "modes": np.array([0], dtype=int),
        "offsets": np.array([0], dtype=int),
        "op_list": [(0, 0)],
    }
    seed_details = evaluate_sfjssp_genome_detailed(instance, seed_genome)

    candidates = _build_assignment_family_candidates(instance, seed_genome, seed_details)

    improving = [
        candidate
        for candidate in candidates
        if candidate.details["metrics"]["weighted_tardiness"] < seed_details["metrics"]["weighted_tardiness"]
    ]
    assert improving
    assert any(candidate.name == "machine_reassign_0_0_to_1" for candidate in improving)


def test_training_pipeline_random_fallback_smoke():
    if TORCH_AVAILABLE:
        return

    env = SFJSSPEnv(_build_simple_instance(), use_graph_state=False)
    pipeline = TrainingPipeline(
        TrainingConfig(n_episodes=1, max_steps_per_episode=10, log_interval=1)
    )

    history = pipeline.train(env, n_episodes=1, verbose=False)

    assert len(history["rewards"]) == 1
    assert history["makespans"][0] > 0.0


def test_multi_agent_ppo_select_actions_stores_runtime_masks():
    if not TORCH_AVAILABLE:
        return

    env = SFJSSPEnv(_build_masked_policy_instance(), use_graph_state=True)
    obs, _ = env.reset()
    obs["job_mask"] = np.zeros_like(obs["job_mask"])
    obs["job_mask"][0] = 1.0

    agent = MultiAgentPPO(
        job_feature_dim=obs["job_nodes"].shape[1],
        op_feature_dim=obs["op_nodes"].shape[1],
        machine_feature_dim=obs["machine_nodes"].shape[1],
        worker_feature_dim=obs["worker_nodes"].shape[1],
        embed_dim=32,
        n_machines=env.instance.n_machines,
        n_workers=env.instance.n_workers,
    )

    action_dict = agent.select_actions(obs, env, deterministic=False)
    states = action_dict["states"]

    assert tuple(states["job_mask"].shape) == (1, obs["job_nodes"].shape[0])
    assert tuple(states["machine_mask"].shape) == (1, env.instance.n_machines)
    assert tuple(states["worker_mask"].shape) == (1, env.instance.n_workers)
    assert states["mode_mask"].shape[0] == 1
    assert states["mode_mask"].shape[1] >= len(env.instance.get_machine(1).modes)
    assert states["job_mask"][0, 0].item() == 1.0
    assert float(states["job_mask"][0, 1].item()) == 0.0
    assert states["mode_mask"][0, 0].item() == 1.0
    assert action_dict["job_action"].item() == 0
    assert action_dict["machine_action"].item() == 1
    assert action_dict["worker_action"].item() == 2


def test_multi_agent_ppo_update_reuses_selection_masks_for_log_probs():
    if not TORCH_AVAILABLE:
        return

    import torch

    torch.manual_seed(7)

    env = SFJSSPEnv(_build_masked_policy_instance(), use_graph_state=True)
    obs, _ = env.reset()
    obs["job_mask"] = np.zeros_like(obs["job_mask"])
    obs["job_mask"][0] = 1.0

    agent = MultiAgentPPO(
        job_feature_dim=obs["job_nodes"].shape[1],
        op_feature_dim=obs["op_nodes"].shape[1],
        machine_feature_dim=obs["machine_nodes"].shape[1],
        worker_feature_dim=obs["worker_nodes"].shape[1],
        embed_dim=32,
        n_machines=env.instance.n_machines,
        n_workers=env.instance.n_workers,
    )

    action_dict = agent.select_actions(obs, env, deterministic=False)
    agent.store_transition(
        states=action_dict["states"],
        actions=action_dict,
        rewards=torch.tensor([0.0], dtype=torch.float32),
        next_states=None,
        dones=torch.tensor([0.0], dtype=torch.float32),
    )

    batch = agent._prepare_buffer_tensors()
    node_embs, global_state = agent._encode_batch(batch)
    current = agent._compute_current_policy_outputs(batch, node_embs, global_state)

    assert torch.allclose(current["job_log_probs"], batch["old_job_log_probs"], atol=1e-6)
    assert torch.allclose(current["machine_log_probs"], batch["old_mac_log_probs"], atol=1e-6)
    assert torch.allclose(current["worker_log_probs"], batch["old_wrk_log_probs"], atol=1e-6)

    logits_j, _ = agent.job_agent(global_state, node_embs["jobs"])
    logits_m, logits_mode, _ = agent.machine_agent(global_state)
    logits_w, _ = agent.worker_agent(global_state)

    unmasked_job = _safe_categorical_from_logits(logits_j).log_prob(batch["job_actions"])
    unmasked_machine = (
        _safe_categorical_from_logits(logits_m).log_prob(batch["mac_actions"])
        + _safe_categorical_from_logits(logits_mode).log_prob(batch["mode_actions"])
    )
    unmasked_worker = _safe_categorical_from_logits(logits_w).log_prob(batch["wrk_actions"])

    assert not torch.allclose(unmasked_job, batch["old_job_log_probs"], atol=1e-6)
    assert not torch.allclose(unmasked_machine, batch["old_mac_log_probs"], atol=1e-6)
    assert not torch.allclose(unmasked_worker, batch["old_wrk_log_probs"], atol=1e-6)
