import json
from pathlib import Path

import pytest

try:
    from ..sfjssp_model.instance import InstanceLabel
except ImportError:  # pragma: no cover - supports repo-root imports
    from sfjssp_model.instance import InstanceLabel

try:
    from ..experiments.generate_benchmarks import (
        BENCHMARK_DOCUMENT_SCHEMA,
        BENCHMARK_DOCUMENT_VERSION,
        BenchmarkGenerator as ExperimentsBenchmarkGenerator,
        DEFAULT_SUITE_SIZES,
        GeneratorConfig as ExperimentsGeneratorConfig,
        InstanceSize as ExperimentsInstanceSize,
    )
    from ..utils.benchmark_generator import (
        BENCHMARK_GENERATOR_VERSION,
        BenchmarkGenerator as UtilsBenchmarkGenerator,
        GeneratorConfig as UtilsGeneratorConfig,
        InstanceSize as UtilsInstanceSize,
        get_size_preset_table,
    )
except ImportError:  # pragma: no cover - supports repo-root imports
    from experiments.generate_benchmarks import (
        BENCHMARK_DOCUMENT_SCHEMA,
        BENCHMARK_DOCUMENT_VERSION,
        BenchmarkGenerator as ExperimentsBenchmarkGenerator,
        DEFAULT_SUITE_SIZES,
        GeneratorConfig as ExperimentsGeneratorConfig,
        InstanceSize as ExperimentsInstanceSize,
    )
    from utils.benchmark_generator import (
        BENCHMARK_GENERATOR_VERSION,
        BenchmarkGenerator as UtilsBenchmarkGenerator,
        GeneratorConfig as UtilsGeneratorConfig,
        InstanceSize as UtilsInstanceSize,
        get_size_preset_table,
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


def test_public_generator_surface_matches_runtime_supported_sizes():
    assert [size.value for size in ExperimentsInstanceSize] == ["small", "medium", "large"]
    assert [size.value for size in UtilsInstanceSize] == ["small", "medium", "large"]
    assert [size.value for size in DEFAULT_SUITE_SIZES] == ["small", "medium"]


def test_benchmark_document_schema_file_exists():
    schema_path = Path(__file__).resolve().parents[1] / "utils" / "benchmark_document.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["properties"]["document_schema"]["const"] == BENCHMARK_DOCUMENT_SCHEMA
    assert schema["properties"]["document_version"]["const"] == BENCHMARK_DOCUMENT_VERSION


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

    assert metadata["document_schema"] == BENCHMARK_DOCUMENT_SCHEMA
    assert metadata["document_version"] == BENCHMARK_DOCUMENT_VERSION
    assert metadata["document_type"] == "suite_manifest"
    assert metadata["generator_version"] == BENCHMARK_GENERATOR_VERSION
    assert metadata["sizes"] == ["medium", "small"]
    assert metadata["instance_types"] == ["dynamic", "static"]
    assert metadata["size_preset_table"] == get_size_preset_table()
    assert {entry["filename"] for entry in metadata["instances"]} == {
        "SFJSSP_small_000.json",
        "SFJSSP_medium_000.json",
    }


def test_save_instance_embeds_schema_and_full_generator_provenance(tmp_path):
    config = UtilsGeneratorConfig(
        instance_id="SCHEMA_TEST",
        size=UtilsInstanceSize.SMALL,
        seed=31,
        n_jobs=2,
        n_machines=2,
        n_workers=2,
        is_dynamic=True,
        absence_probability=0.75,
    )
    generator = UtilsBenchmarkGenerator(config)
    instance = generator.generate()

    output_path = tmp_path / "SCHEMA_TEST.json"
    generator.save_instance(instance, str(output_path))
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["document_schema"] == BENCHMARK_DOCUMENT_SCHEMA
    assert payload["document_version"] == BENCHMARK_DOCUMENT_VERSION
    assert payload["document_type"] == "instance"
    assert payload["calibration_status"] == "fully_synthetic"
    assert payload["calibration_status_justification"]
    assert payload["generator_config"]["size"] == "small"
    assert payload["generator_config"]["calibration_status"] == "fully_synthetic"
    assert payload["generator_config"]["absence_probability"] == 0.75
    assert payload["generator_provenance"]["generator_version"] == BENCHMARK_GENERATOR_VERSION
    assert payload["generator_provenance"]["calibration_status"] == "fully_synthetic"
    assert payload["generator_provenance"]["runtime_supported_sizes"] == ["small", "medium", "large"]
    assert payload["size_preset_table"] == get_size_preset_table()


def test_generator_rejects_calibrated_claim_without_calibration_sources():
    generator = UtilsBenchmarkGenerator(
        UtilsGeneratorConfig(
            instance_id="BAD_CALIBRATION",
            size=UtilsInstanceSize.SMALL,
            seed=37,
            n_jobs=2,
            n_machines=2,
            n_workers=2,
            label=InstanceLabel.CALIBRATED_SYNTHETIC,
            calibration_sources=[],
        )
    )

    with pytest.raises(ValueError, match="requires at least one calibration source"):
        generator.generate()
