import json

try:
    from ..experiments.generate_benchmarks import (
        BenchmarkGenerator as ExperimentsBenchmarkGenerator,
        GeneratorConfig as ExperimentsGeneratorConfig,
        InstanceSize as ExperimentsInstanceSize,
    )
    from ..utils.benchmark_generator import (
        BenchmarkGenerator as UtilsBenchmarkGenerator,
        GeneratorConfig as UtilsGeneratorConfig,
        InstanceSize as UtilsInstanceSize,
    )
except ImportError:  # pragma: no cover - supports repo-root imports
    from experiments.generate_benchmarks import (
        BenchmarkGenerator as ExperimentsBenchmarkGenerator,
        GeneratorConfig as ExperimentsGeneratorConfig,
        InstanceSize as ExperimentsInstanceSize,
    )
    from utils.benchmark_generator import (
        BenchmarkGenerator as UtilsBenchmarkGenerator,
        GeneratorConfig as UtilsGeneratorConfig,
        InstanceSize as UtilsInstanceSize,
    )


def _instance_signature(instance):
    return {
        "instance_id": instance.instance_id,
        "instance_type": instance.instance_type.value,
        "n_jobs": instance.n_jobs,
        "n_machines": instance.n_machines,
        "n_workers": instance.n_workers,
        "n_operations": instance.n_operations,
        "job_operation_counts": [len(job.operations) for job in instance.jobs],
        "job_arrivals": [round(job.arrival_time, 6) for job in instance.jobs],
        "job_due_dates": [round(job.due_date, 6) for job in instance.jobs],
        "machine_ids": [machine.machine_id for machine in instance.machines],
        "worker_ids": [worker.worker_id for worker in instance.workers],
        "ergonomic_risk_count": len(instance.ergonomic_risk_map),
    }


def test_generator_config_preserves_explicit_counts_across_public_surfaces():
    exp_config = ExperimentsGeneratorConfig(
        size=ExperimentsInstanceSize.SMALL,
        n_jobs=3,
        n_machines=2,
        n_workers=4,
        seed=11,
    )
    utils_config = UtilsGeneratorConfig(
        size=UtilsInstanceSize.SMALL,
        n_jobs=3,
        n_machines=2,
        n_workers=4,
        seed=11,
    )

    assert exp_config.n_jobs == 3
    assert exp_config.n_machines == 2
    assert exp_config.n_workers == 4
    assert utils_config.n_jobs == 3
    assert utils_config.n_machines == 2
    assert utils_config.n_workers == 4


def test_experiments_and_utils_generators_share_config_semantics():
    exp_generator = ExperimentsBenchmarkGenerator(
        ExperimentsGeneratorConfig(
            instance_id="GEN_COMPAT",
            size=ExperimentsInstanceSize.SMALL,
            n_jobs=4,
            n_machines=3,
            n_workers=2,
            seed=19,
            is_dynamic=True,
        )
    )
    utils_generator = UtilsBenchmarkGenerator(
        UtilsGeneratorConfig(
            instance_id="GEN_COMPAT",
            size=UtilsInstanceSize.SMALL,
            n_jobs=4,
            n_machines=3,
            n_workers=2,
            seed=19,
            is_dynamic=True,
        )
    )

    exp_instance = exp_generator.generate()
    utils_instance = utils_generator.generate()

    assert _instance_signature(exp_instance) == _instance_signature(utils_instance)


def test_save_suite_metadata_separates_sizes_from_instance_types(tmp_path):
    generator = UtilsBenchmarkGenerator(
        UtilsGeneratorConfig(
            instance_id="SFJSSP_small_000",
            size=UtilsInstanceSize.SMALL,
            seed=23,
            n_jobs=2,
            n_machines=2,
            n_workers=2,
        )
    )
    small_instance = generator.generate()
    generator.config = UtilsGeneratorConfig(
        instance_id="SFJSSP_medium_000",
        size=UtilsInstanceSize.MEDIUM,
        seed=29,
        n_jobs=2,
        n_machines=2,
        n_workers=2,
        is_dynamic=True,
    )
    medium_instance = generator.generate()

    generator.save_suite([small_instance, medium_instance], str(tmp_path))

    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["sizes"] == ["medium", "small"]
    assert metadata["instance_types"] == ["dynamic", "static"]
