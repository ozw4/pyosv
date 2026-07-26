"""Command-line orchestration for the canonical F3 full-volume comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from pyosv.evaluation.f3d_mode_comparison.artifacts import RUN_MANIFEST_FILE
from pyosv.evaluation.f3d_mode_comparison import (
    RUN_COMPLETION_FILE,
    F3ModeComparisonConfig,
    F3ModeComparisonResult,
    F3VolumeSource,
    PeakRSSRecorder,
    build_f3d_mode_comparison_plan,
    ensure_output_not_in_data_root,
    extract_f3d_diagnostics,
    extract_f3d_metrics,
    extract_f3d_resources,
    finalize_f3d_bundle,
    prepare_run_workspace,
    run_f3d_mode_comparison,
    run_scanner_stages,
    validate_completed_f3d_bundle,
)
from pyosv.f3d_reference import F3D_ENV_VAR, resolve_f3d_data_root


def _nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


class _ArgumentParser(argparse.ArgumentParser):
    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        if parsed.validate_only and parsed.resume:
            self.error("--validate-only cannot be combined with --resume")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the canonical F3 mode-comparison argument parser."""

    parser = _ArgumentParser(
        description=(
            "Run the canonical full-volume F3 2x2 scanner/workflow comparison "
            "(four cells) and report public-reference agreement, not geological "
            "accuracy."
        ),
        epilog=(
            "Without --resume, OUTPUT_DIR must not exist. --resume accepts only an "
            "incomplete workspace with the matching run fingerprint or a valid "
            "complete bundle; a complete bundle is validated without recomputation."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"External F3 data root. Defaults to {F3D_ENV_VAR}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="F3 comparison workspace and final bundle path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only an exact matching incomplete workspace or valid bundle.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing complete bundle without running experiment stages.",
    )
    parser.add_argument(
        "--deep-validate",
        action="store_true",
        help="Recompute full-volume metrics during validation.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Request human-readable JSON formatting where the bundle contract permits it.",
    )
    parser.add_argument(
        "--no-skinning",
        action="store_true",
        help="Disable skinning identically in all four canonical cells.",
    )
    parser.add_argument(
        "--boundary-margin",
        type=_nonnegative_int,
        default=16,
        metavar="N",
        help="Boundary-shell diagnostic margin applied identically to all four cells.",
    )
    return parser


def run_experiment(
    *,
    config: F3ModeComparisonConfig,
    data_root: Path,
    output_dir: Path,
    resume: bool,
    deep: bool,
    pretty: bool = False,
) -> Path:
    """Run or resume one canonical comparison and return its bundle path."""

    # JSON bytes are canonical parts of the F3 result contract. ``pretty`` is
    # accepted by the public CLI uniformly but cannot alter those bytes.
    del pretty
    plan = build_f3d_mode_comparison_plan(config)
    root = Path(output_dir)
    completion: Path | None = None
    completion_preexisted = False

    try:
        with F3VolumeSource(data_root) as source:
            workspace = prepare_run_workspace(
                root,
                plan,
                source.identity,
                resume=resume,
            )
            completion = workspace.path / RUN_COMPLETION_FILE
            completion_preexisted = os.path.lexists(completion)
            if completion_preexisted:
                validate_completed_f3d_bundle(workspace.path, deep=deep)
                return workspace.path

            rss = PeakRSSRecorder()
            scanner_stages = run_scanner_stages(
                workspace,
                source,
                plan,
                rss_recorder=rss,
            )
            cell_result = run_f3d_mode_comparison(
                workspace,
                plan,
                scanner_stages,
                rss_recorder=rss,
            )
            rss.process_peak()
            metrics = extract_f3d_metrics(source, cell_result.cells)
            diagnostics = extract_f3d_diagnostics(
                source,
                cell_result.cells,
                plan,
            )
            resources = extract_f3d_resources(
                cell_result.stage_runtime,
                shape=plan.dataset_spec.shape,
                workspace=workspace,
                scanner_stages=scanner_stages,
                rss_recorder=rss,
            )
            result = F3ModeComparisonResult.from_extractions(
                workspace=workspace,
                cells=cell_result.cells,
                metrics=metrics,
                diagnostics=diagnostics,
                resources=resources,
            )
            finalize_f3d_bundle(workspace, result, resume=resume, deep=False)
            validate_completed_f3d_bundle(workspace.path, deep=deep)
            return workspace.path
    except Exception:
        if completion is not None and not completion_preexisted and os.path.lexists(completion):
            completion.unlink(missing_ok=True)
        raise


def _recorded_data_root(bundle: Path) -> Path:
    manifest_path = bundle / RUN_MANIFEST_FILE
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read bundle data-root provenance: {manifest_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("bundle run manifest must contain an object")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("bundle run manifest has no data-root provenance")
    data_root = provenance.get("data_root")
    if not isinstance(data_root, str) or not data_root:
        raise ValueError("bundle run manifest has invalid data-root provenance")
    return Path(data_root)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute, resume, or validate a canonical F3 comparison."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            configured_data_root = (
                args.data_root if args.data_root is not None else os.environ.get(F3D_ENV_VAR)
            )
            requested_bundle = args.output_dir.resolve(strict=False)
            data_root = (
                _recorded_data_root(requested_bundle)
                if configured_data_root is None
                else Path(configured_data_root)
            )
            bundle = ensure_output_not_in_data_root(requested_bundle, data_root)
            validate_completed_f3d_bundle(bundle, deep=args.deep_validate)
        else:
            data_root = resolve_f3d_data_root(args.data_root).resolve(strict=False)
            requested_output_exists = os.path.lexists(args.output_dir)
            output_dir = ensure_output_not_in_data_root(args.output_dir, data_root)
            output_exists = requested_output_exists or os.path.lexists(output_dir)
            if output_exists and not args.resume:
                raise FileExistsError(f"run workspace already exists: {output_dir}")
            if args.resume and not output_exists:
                raise FileNotFoundError(f"run workspace does not exist: {output_dir}")
            config = F3ModeComparisonConfig(
                skinning_enabled=not args.no_skinning,
                boundary_diagnostic_margin=args.boundary_margin,
            )
            if not args.resume:
                output_dir.parent.mkdir(parents=True, exist_ok=True)
            bundle = run_experiment(
                config=config,
                data_root=data_root,
                output_dir=output_dir,
                resume=args.resume,
                deep=args.deep_validate,
                pretty=args.pretty,
            )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
