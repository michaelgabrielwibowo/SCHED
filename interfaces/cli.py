"""
Thin CLI for external import, solve, audit, and export workflows.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .adapters import SUPPORTED_SOURCE_ADAPTERS
from .audit import build_schedule_audit
from .csv_importers import load_instance_from_csv_bundle
from .errors import InterfaceValidationError
from .exporters import export_schedule_artifacts
from .importers import load_instance_from_json
from .runbook import (
    CLI_ERROR_CATALOG,
    DEFAULT_RUN_OUTPUT_ROOT,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    REQUIRED_MANIFEST_ARTIFACTS,
    RunDirectoryContractError,
    build_default_run_output_dir,
    build_default_spreadsheet_export_dir,
    load_run_directory_bundle,
)
from .site_profiles import SUPPORTED_SITE_PROFILES

try:
    from ..baseline_solver.greedy_solvers import (
        GreedyScheduler,
        composite_rule,
        critical_ratio_rule,
        earliest_ready_rule,
        edt_rule,
        fifo_rule,
        least_slack_rule,
        spt_rule,
        tardiness_composite_rule,
    )
    from ..exact_solvers.cp_solver import solve_sfjssp
except ImportError:  # pragma: no cover - supports repo-root imports
    from baseline_solver.greedy_solvers import (
        GreedyScheduler,
        composite_rule,
        critical_ratio_rule,
        earliest_ready_rule,
        edt_rule,
        fifo_rule,
        least_slack_rule,
        spt_rule,
        tardiness_composite_rule,
    )
    from exact_solvers.cp_solver import solve_sfjssp


GREEDY_RULES = {
    "fifo": fifo_rule,
    "spt": spt_rule,
    "edd": edt_rule,
    "earliest_ready": earliest_ready_rule,
    "least_slack": least_slack_rule,
    "critical_ratio": critical_ratio_rule,
    "composite": composite_rule,
    "tardiness_composite": tardiness_composite_rule,
}
SUPPORTED_SOLVERS = tuple(
    [f"greedy:{name}" for name in GREEDY_RULES] + ["cp:makespan"]
)


class CLIError(Exception):
    """Controlled CLI failure with a stable exit code and JSON payload."""

    def __init__(
        self,
        exit_code: int,
        code: str,
        message: str,
        *,
        error_class: str,
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.exit_code = exit_code
        self.payload = {
            "status": "error",
            "exit_code": exit_code,
            "error_class": error_class,
            "code": code,
            "message": message,
            "details": details,
        }


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if not resolved_argv:
        parser.print_help(sys.stdout)
        return EXIT_SUCCESS

    args = parser.parse_args(resolved_argv)

    try:
        if args.command == "validate-input":
            payload = _handle_validate_input(args)
        elif args.command in {"run", "solve"}:
            payload = _handle_solve(args)
        elif args.command == "audit":
            payload = _handle_audit(args)
        elif args.command == "export":
            payload = _handle_export(args)
        else:  # pragma: no cover - argparse constrains this
            raise _catalog_error(
                "unknown_command",
                f"Unknown command {args.command!r}.",
            )
    except InterfaceValidationError as exc:
        _emit_json(sys.stderr, _validation_error_payload(exc))
        return CLI_ERROR_CATALOG["input_validation_failed"]["exit_code"]
    except CLIError as exc:
        _emit_json(sys.stderr, exc.payload)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive catch for CLI surface
        _emit_json(
            sys.stderr,
            _error_payload(
                EXIT_RUNTIME_ERROR,
                CLI_ERROR_CATALOG["unhandled_exception"]["error_class"],
                "unhandled_exception",
                str(exc),
                {"exception_type": type(exc).__name__},
            ),
        )
        return EXIT_RUNTIME_ERROR

    _emit_json(sys.stdout, payload)
    return EXIT_SUCCESS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_detect_prog_name(),
        description="Validate external SFJSSP inputs and run documented operator workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-input",
        help=(
            "Validate a JSON input document or CSV bundle against the supported "
            "sfjssp_external_v1/v2 contracts."
        ),
    )
    validate_parser.add_argument(
        "--input",
        required=True,
        help="Path to the external JSON input file or CSV bundle directory.",
    )
    validate_parser.add_argument(
        "--lenient",
        action="store_true",
        help="Allow unknown fields during import validation.",
    )
    validate_parser.add_argument(
        "--adapter",
        help=(
            "Optional raw-source adapter name for plant-like JSON inputs. "
            f"Supported values: {', '.join(sorted(SUPPORTED_SOURCE_ADAPTERS))}"
        ),
    )
    validate_parser.add_argument(
        "--site-profile",
        help=(
            "Optional explicit site-parameter overlay. "
            f"Supported values: {', '.join(sorted(SUPPORTED_SITE_PROFILES))}"
        ),
    )

    solve_parser = subparsers.add_parser(
        "solve",
        help="Import, solve, audit, and export one schedule run to a deterministic run directory.",
    )
    _add_solve_like_arguments(solve_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Compatibility alias for `solve`.",
    )
    _add_solve_like_arguments(run_parser)

    audit_parser = subparsers.add_parser(
        "audit",
        help="Validate and summarize a previously exported run directory.",
    )
    audit_parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a run directory containing run_manifest.json and the stable exported artifacts.",
    )
    audit_parser.add_argument(
        "--max-violations",
        type=int,
        default=10,
        help="Maximum number of hard violations to include in the summary payload.",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Copy the spreadsheet-facing artifact set from a validated run directory.",
    )
    export_parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a validated run directory containing run_manifest.json.",
    )
    export_parser.add_argument(
        "--target-dir",
        help=(
            "Optional handoff directory. Defaults to a deterministic "
            "`spreadsheet_export/` subdirectory inside the run directory."
        ),
    )

    return parser


def _detect_prog_name(argv0: Optional[str] = None) -> str:
    """Render help for the active entrypoint without changing CLI behavior."""

    override = os.environ.get("SCHED_CLI_PROG")
    if override:
        return override

    candidate = Path(argv0 or sys.argv[0]).name.lower()
    if candidate in {"sched", "sched.exe", "sched.cmd", "sched.ps1"}:
        return "SCHED"
    return "python -m interfaces.cli"


def _add_solve_like_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the external JSON input file or CSV bundle directory.",
    )
    parser.add_argument(
        "--solver",
        required=True,
        help=f"Solver spec. Supported values: {', '.join(SUPPORTED_SOLVERS)}",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional explicit run directory. If omitted, the CLI uses the deterministic "
            "`<output-root>/<instance-id>/<solver-spec>` convention."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_RUN_OUTPUT_ROOT,
        help=(
            "Root directory for deterministic run directories when --output-dir is omitted. "
            f"Default: {DEFAULT_RUN_OUTPUT_ROOT}"
        ),
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=60,
        help="Solver time limit in seconds for exact solvers.",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Allow unknown fields during import validation.",
    )
    parser.add_argument(
        "--adapter",
        help=(
            "Optional raw-source adapter name for plant-like JSON inputs. "
            f"Supported values: {', '.join(sorted(SUPPORTED_SOURCE_ADAPTERS))}"
        ),
    )
    parser.add_argument(
        "--site-profile",
        help=(
            "Optional explicit site-parameter overlay. "
            f"Supported values: {', '.join(sorted(SUPPORTED_SITE_PROFILES))}"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable solver verbosity where supported.",
    )


def _handle_validate_input(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input)
    imported, input_format = _load_imported_instance(
        input_path,
        strict=not args.lenient,
        adapter_name=args.adapter,
        site_profile_name=args.site_profile,
    )
    return {
        "status": "ok",
        "exit_code": EXIT_SUCCESS,
        "command": "validate-input",
        "schema": imported.schema,
        "input_path": str(input_path),
        "input_format": input_format,
        "instance_id": imported.instance.instance_id,
        "instance_name": imported.instance.instance_name,
        "counts": {
            "jobs": imported.instance.n_jobs,
            "machines": imported.instance.n_machines,
            "workers": imported.instance.n_workers,
            "operations": imported.instance.n_operations,
        },
        "strict": not args.lenient,
        "provenance": imported.provenance,
    }


def _handle_solve(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input)
    imported, input_format = _load_imported_instance(
        input_path,
        strict=not args.lenient,
        adapter_name=args.adapter,
        site_profile_name=args.site_profile,
    )
    output_dir = _resolve_output_dir(args, imported.instance.instance_id, args.solver)
    solver_kind, solver_option = _parse_solver_spec(args.solver)

    start = time.perf_counter()
    schedule = _solve_instance(
        imported.instance,
        solver_kind=solver_kind,
        solver_option=solver_option,
        time_limit=args.time_limit,
        verbose=args.verbose,
    )
    runtime_seconds = time.perf_counter() - start

    if schedule is None:
        raise _catalog_error(
            "solver_no_solution",
            f"Solver {args.solver!r} returned no schedule.",
            {"solver": args.solver},
        )

    schedule.metadata.update(
        {
            "solver": solver_kind,
            "solver_spec": args.solver,
            "runtime_seconds": runtime_seconds,
        }
    )
    if solver_kind == "greedy":
        schedule.metadata["solver_objective"] = "dispatch_rule"
        schedule.metadata["dispatch_rule"] = solver_option

    audit_payload = build_schedule_audit(
        schedule,
        imported.instance,
        provenance={
            **imported.provenance,
            "solver": args.solver,
            "objective": "makespan" if solver_kind == "cp" else "dispatch_rule",
            "runtime_seconds": runtime_seconds,
            "input_source_id": str(input_path),
            "input_schema": imported.schema,
            "input_format": input_format,
        },
        id_maps=imported.id_maps,
        input_schema=imported.schema,
        input_source_id=str(input_path),
    )
    manifest = export_schedule_artifacts(
        output_dir,
        schedule,
        imported.instance,
        audit_payload=audit_payload,
        id_maps=imported.id_maps,
    )
    run_bundle = _load_validated_run_directory(output_dir)
    manifest_path = run_bundle.manifest_path

    return {
        "status": "ok",
        "exit_code": EXIT_SUCCESS,
        "command": args.command,
        "input_path": str(input_path),
        "input_format": input_format,
        "output_dir": str(output_dir),
        "solver": args.solver,
        "instance_id": imported.instance.instance_id,
        "manifest_path": str(manifest_path),
        "feasible": manifest["feasible"],
        "hard_violation_count": manifest["hard_violation_count"],
        "manifest_complete": True,
        "artifacts": dict(manifest["artifacts"]),
        "metrics": {
            "makespan": audit_payload["soft_summary"]["metrics"].get("makespan"),
            "total_energy": audit_payload["soft_summary"]["metrics"].get("total_energy"),
            "weighted_tardiness": audit_payload["soft_summary"]["metrics"].get("weighted_tardiness"),
            "n_tardy_jobs": audit_payload["soft_summary"]["metrics"].get("n_tardy_jobs"),
        },
        "provenance": imported.provenance,
    }


def _handle_audit(args: argparse.Namespace) -> Dict[str, Any]:
    run_bundle = _load_validated_run_directory(args.run_dir)
    audit_payload = run_bundle.audit_payload
    hard_violations = list(audit_payload.get("hard_violations", []))
    return {
        "status": "ok",
        "exit_code": EXIT_SUCCESS,
        "command": "audit",
        "run_dir": str(run_bundle.run_dir),
        "manifest_path": str(run_bundle.manifest_path),
        "instance_id": run_bundle.manifest.get("instance_id"),
        "feasible": audit_payload["feasible"],
        "hard_violation_count": audit_payload["hard_violation_count"],
        "hard_violation_counts": dict(audit_payload.get("hard_violation_counts", {})),
        "manifest_complete": True,
        "artifacts": dict(run_bundle.manifest["artifacts"]),
        "calibration": dict(run_bundle.manifest["calibration"]),
        "top_hard_violations": hard_violations[: max(args.max_violations, 0)],
        "provenance": dict(run_bundle.manifest["provenance"]),
    }


def _handle_export(args: argparse.Namespace) -> Dict[str, Any]:
    run_bundle = _load_validated_run_directory(args.run_dir)
    target_dir = (
        Path(args.target_dir)
        if args.target_dir
        else build_default_spreadsheet_export_dir(run_bundle.run_dir)
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    files_to_copy = ["run_manifest.json"] + list(REQUIRED_MANIFEST_ARTIFACTS.values())
    for filename in files_to_copy:
        source = run_bundle.run_dir / filename
        destination = target_dir / filename
        shutil.copy2(source, destination)
        copied_files.append(filename)

    return {
        "status": "ok",
        "exit_code": EXIT_SUCCESS,
        "command": "export",
        "run_dir": str(run_bundle.run_dir),
        "target_dir": str(target_dir),
        "manifest_complete": True,
        "copied_files": copied_files,
        "calibration": dict(run_bundle.manifest["calibration"]),
        "provenance": dict(run_bundle.manifest["provenance"]),
    }


def _load_imported_instance(
    path: Path,
    *,
    strict: bool,
    adapter_name: Optional[str],
    site_profile_name: Optional[str],
):
    if path.is_dir():
        if adapter_name:
            raise _catalog_error(
                "unsupported_adapter_input",
                details={"adapter": adapter_name, "input_path": str(path)},
            )
        return (
            load_instance_from_csv_bundle(
                path,
                strict=strict,
                site_profile_name=site_profile_name,
            ),
            "csv_bundle",
        )
    return (
        load_instance_from_json(
            path,
            strict=strict,
            adapter_name=adapter_name,
            site_profile_name=site_profile_name,
        ),
        "json",
    )


def _parse_solver_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise _catalog_error(
            "unsupported_solver",
            f"Solver {spec!r} is invalid. Expected '<family>:<mode>'.",
            {"supported_solvers": list(SUPPORTED_SOLVERS)},
        )

    family, option = spec.split(":", 1)
    family = family.strip().lower()
    option = option.strip().lower()

    if family == "greedy" and option in GREEDY_RULES:
        return family, option
    if family == "cp" and option == "makespan":
        return family, option

    raise _catalog_error(
        "unsupported_solver",
        f"Solver {spec!r} is not supported by the CLI.",
        {"supported_solvers": list(SUPPORTED_SOLVERS)},
    )


def _solve_instance(
    instance,
    *,
    solver_kind: str,
    solver_option: str,
    time_limit: int,
    verbose: bool,
):
    if solver_kind == "greedy":
        scheduler = GreedyScheduler(job_rule=GREEDY_RULES[solver_option])
        return scheduler.schedule(instance, verbose=verbose)

    if solver_kind == "cp":
        try:
            return solve_sfjssp(
                instance,
                method="cp",
                objective="makespan",
                time_limit=time_limit,
                verbose=verbose,
            )
        except ImportError as exc:
            raise _catalog_error(
                "missing_dependency",
                str(exc),
                {"solver": "cp:makespan"},
            ) from exc
        except NotImplementedError as exc:
            raise _catalog_error(
                "solver_unavailable",
                str(exc),
                {"solver": "cp:makespan"},
            ) from exc

    raise _catalog_error(
        "unsupported_solver",
        f"Solver family {solver_kind!r} is not supported.",
        {"supported_solvers": list(SUPPORTED_SOLVERS)},
    )


def _validation_error_payload(exc: InterfaceValidationError) -> Dict[str, Any]:
    return _error_payload(
        CLI_ERROR_CATALOG["input_validation_failed"]["exit_code"],
        CLI_ERROR_CATALOG["input_validation_failed"]["error_class"],
        "input_validation_failed",
        CLI_ERROR_CATALOG["input_validation_failed"]["default_message"],
        [asdict(issue) for issue in exc.issues],
    )


def _resolve_output_dir(
    args: argparse.Namespace,
    instance_id: str,
    solver_spec: str,
) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return build_default_run_output_dir(args.output_root, instance_id, solver_spec)


def _load_validated_run_directory(run_dir: Any):
    try:
        return load_run_directory_bundle(run_dir)
    except RunDirectoryContractError as exc:
        raise _catalog_error(
            exc.code,
            str(exc),
            {"run_dir": str(run_dir)},
        ) from exc


def _catalog_error(
    code: str,
    message: Optional[str] = None,
    details: Optional[Any] = None,
) -> CLIError:
    catalog_entry = CLI_ERROR_CATALOG[code]
    return CLIError(
        catalog_entry["exit_code"],
        code,
        message or catalog_entry["default_message"],
        error_class=catalog_entry["error_class"],
        details=details,
    )


def _error_payload(
    exit_code: int,
    error_class: str,
    code: str,
    message: str,
    details: Optional[Any],
) -> Dict[str, Any]:
    return {
        "status": "error",
        "exit_code": exit_code,
        "error_class": error_class,
        "code": code,
        "message": message,
        "details": details,
    }


def _emit_json(stream, payload: Dict[str, Any]) -> None:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess tests
    raise SystemExit(main())
