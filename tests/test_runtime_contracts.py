import numpy as np

try:
    from ..environment.sfjssp_env import SFJSSPEnv
    from ..moea.nsga3 import create_sfjssp_genome, evaluate_sfjssp_genome
    from ..sfjssp_model.instance import SFJSSPInstance
    from ..sfjssp_model.job import Job, Operation
    from ..sfjssp_model.machine import Machine, MachineMode
    from ..sfjssp_model.worker import Worker
    from ..training.train_drl import TrainingConfig, TrainingPipeline, TORCH_AVAILABLE
except ImportError:  # pragma: no cover - supports repo-root imports
    from environment.sfjssp_env import SFJSSPEnv
    from moea.nsga3 import create_sfjssp_genome, evaluate_sfjssp_genome
    from sfjssp_model.instance import SFJSSPInstance
    from sfjssp_model.job import Job, Operation
    from sfjssp_model.machine import Machine, MachineMode
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
