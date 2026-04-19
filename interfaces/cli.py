"""
Thin CLI for external import, solve, audit, and export workflows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .audit import build_schedule_audit
from .csv_importers import load_instance_from_csv_bundle
from .errors import InterfaceValidationError
from .exporters import export_schedule_artifacts
from .importers import load_instance_from_json

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


EXIT_SUCCESS = 0
EXIT_VALIDATION_ERROR = 2
EXIT_UNSUPPORTED_SOLVER = 3
EXIT_SOLVER_FAILURE = 4
EXIT_RUNTIME_ERROR = 5

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
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.exit_code = exit_code
        self.payload = {
            "status": "error",
            "code": code,
            "message": message,
            "details": details,
        }


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate-input":
            payload = _handle_validate_input(args)
        elif args.command == "run":
            payload = _handle_run(args)
        else:  # pragma: no cover - argparse constrains this
            raise CLIError(
                EXIT_RUNTIME_ERROR,
                "unknown_command",
                f"Unknown command {args.command!r}.",
            )
    except InterfaceValidationError as exc:
        _emit_json(sys.stderr, _validation_error_payload(exc))
        return EXIT_VALIDATION_ERROR
    except CLIError as exc:
        _emit_json(sys.stderr, exc.payload)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive catch for CLI surface
        _emit_json(
            sys.stderr,
            {
                "status": "error",
                "code": "unhandled_exception",
                "message": str(exc),
                "details": {"exception_type": type(exc).__name__},
            },
        )
        return EXIT_RUNTIME_ERROR

    _emit_json(sys.stdout, payload)
    return EXIT_SUCCESS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m interfaces.cli",
        description="Validate external SFJSSP inputs and run supported solver workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-input",
        help="Validate a JSON input document or CSV bundle against sfjssp_external_v1.",
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

    run_parser = subparsers.add_parser(
        "run",
        help="Import, solve, audit, and export one schedule run.",
    )
    run_parser.add_argument(
        "--input",
        required=True,
        help="Path to the external JSON input file or CSV bundle directory.",
    )
    run_parser.add_argument(
        "--solver",
        required=True,
        help=f"Solver spec. Supported values: {', '.join(SUPPORTED_SOLVERS)}",
    )
    run_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for run_manifest.json, schedule.json, CSV timelines, and violations exports.",
    )
    run_parser.add_argument(
        "--time-limit",
        type=int,
        default=60,
        help="Solver time limit in seconds for exact solvers.",
    )
    run_parser.add_argument(
        "--lenient",
        action="store_true",
        help="Allow unknown fields during import validation.",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable solver verbosity where supported.",
    )

    return parser


def _handle_validate_input(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input)
    imported, input_format = _load_imported_instance(input_path, strict=not args.lenient)
    return {
        "status": "ok",
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
    }


def _handle_run(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    imported, input_format = _load_imported_instance(input_path, strict=not args.lenient)
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
        raise CLIError(
            EXIT_SOLVER_FAILURE,
            "solver_no_solution",
            f"Solver {args.solver!r} returned no schedule.",
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
    manifest_path = output_dir / "run_manifest.json"

    return {
        "status": "ok",
        "command": "run",
        "input_path": str(input_path),
        "input_format": input_format,
        "output_dir": str(output_dir),
        "solver": args.solver,
        "instance_id": imported.instance.instance_id,
        "manifest_path": str(manifest_path),
        "feasible": manifest["feasible"],
        "hard_violation_count": manifest["hard_violation_count"],
        "metrics": {
            "makespan": audit_payload["soft_summary"]["metrics"].get("makespan"),
            "total_energy": audit_payload["soft_summary"]["metrics"].get("total_energy"),
            "weighted_tardiness": audit_payload["soft_summary"]["metrics"].get("weighted_tardiness"),
            "n_tardy_jobs": audit_payload["soft_summary"]["metrics"].get("n_tardy_jobs"),
        },
    }


def _load_imported_instance(path: Path, *, strict: bool):
    if path.is_dir():
        return load_instance_from_csv_bundle(path, strict=strict), "csv_bundle"
    return load_instance_from_json(path, strict=strict), "json"


def _parse_solver_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise CLIError(
            EXIT_UNSUPPORTED_SOLVER,
            "unsupported_solver",
            f"Solver {spec!r} is invalid. Expected '<family>:<mode>'.",
            details={"supported_solvers": list(SUPPORTED_SOLVERS)},
        )

    family, option = spec.split(":", 1)
    family = family.strip().lower()
    option = option.strip().lower()

    if family == "greedy" and option in GREEDY_RULES:
        return family, option
    if family == "cp" and option == "makespan":
        return family, option

    raise CLIError(
        EXIT_UNSUPPORTED_SOLVER,
        "unsupported_solver",
        f"Solver {spec!r} is not supported by the CLI.",
        details={"supported_solvers": list(SUPPORTED_SOLVERS)},
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
            raise CLIError(
                EXIT_SOLVER_FAILURE,
                "missing_dependency",
                str(exc),
                details={"solver": "cp:makespan"},
            ) from exc
        except NotImplementedError as exc:
            raise CLIError(
                EXIT_SOLVER_FAILURE,
                "solver_unavailable",
                str(exc),
                details={"solver": "cp:makespan"},
            ) from exc

    raise CLIError(
        EXIT_UNSUPPORTED_SOLVER,
        "unsupported_solver",
        f"Solver family {solver_kind!r} is not supported.",
        details={"supported_solvers": list(SUPPORTED_SOLVERS)},
    )


def _validation_error_payload(exc: InterfaceValidationError) -> Dict[str, Any]:
    return {
        "status": "error",
        "code": "input_validation_failed",
        "message": "External input validation failed.",
        "details": [asdict(issue) for issue in exc.issues],
    }


def _emit_json(stream, payload: Dict[str, Any]) -> None:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess tests
    raise SystemExit(main())
