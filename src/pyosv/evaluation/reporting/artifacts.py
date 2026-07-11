"""Writers for controlled synthetic quality report artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from pyosv.evaluation.synthetic_quality.variants import VARIANT_NAMES

VOLUME_NAMES = (
    "truth_fault_mask",
    "truth_distance",
    "truth_strike",
    "truth_dip",
    "ft_oracle",
    "pt_oracle",
    "tt_oracle",
    "fv_py",
    "vp_py",
    "vt_py",
    "fvt_py",
    "skin_mask_py",
)
FIGURE_VOLUME_NAMES = ("ft_oracle", "fv_py", "fvt_py")
SCANNER_VOLUME_NAMES = (
    ("scanner_input", "scanner_input"),
    ("scanner_ft", "ft_scan"),
    ("scanner_pt", "pt_scan"),
    ("scanner_tt", "tt_scan"),
    ("scanner_fet", "ft_used"),
    ("scanner_fpt", "pt_used"),
    ("scanner_ftt", "tt_used"),
    ("scanner_confidence", "scanner_confidence"),
)
SCANNER_FIGURE_VOLUME_NAMES = (
    ("scanner_input", "scanner_input"),
    ("scanner_ft", "ft_scan"),
    ("scanner_fet", "ft_used"),
)
THINNING_DIAGNOSTIC_VOLUME_NAMES = (
    ("fvt_reference_thinning_diagnostic", "fvt_reference"),
    ("fvt_normal_thinning_diagnostic", "fvt_normal"),
    ("keep_reference_thinning_diagnostic", "keep_reference"),
    ("keep_normal_thinning_diagnostic", "keep_normal"),
    ("keep_both_thinning_diagnostic", "keep_both"),
    ("keep_reference_only_thinning_diagnostic", "keep_reference_only"),
    ("keep_normal_only_thinning_diagnostic", "keep_normal_only"),
)
PIPELINE_OUTPUTS_KEY = "__pipelines__"
PIPELINE_NAMES = ("oracle", "scanner")


def write_case_volumes(
    volume_outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    output_dir: str | PathLike[str],
) -> list[Path]:
    from pyosv.io import write_dat

    written = []
    output_root = Path(output_dir)
    for case_id, variants in volume_outputs.items():
        case_dir = _case_output_dir(output_root, case_id)
        for variant, volumes in variants.items():
            output_dir_for_variant = _variant_output_dir(case_dir, variant, len(variants) == 1)
            for pipeline, pipeline_volumes in _iter_pipeline_volume_outputs(volumes):
                volume_dir = _pipeline_output_dir(output_dir_for_variant, pipeline)
                written.extend(_write_pipeline_volumes(volume_dir, pipeline_volumes, write_dat))
    return written


def _iter_pipeline_volume_outputs(
    volumes: Mapping[str, Any],
) -> Sequence[tuple[str | None, Mapping[str, np.ndarray]]]:
    pipeline_outputs = volumes.get(PIPELINE_OUTPUTS_KEY)
    if isinstance(pipeline_outputs, Mapping):
        unknown = sorted(set(pipeline_outputs).difference(PIPELINE_NAMES))
        if unknown:
            raise ValueError(f"unknown pipeline(s): {','.join(unknown)}")
        return tuple(
            (pipeline, pipeline_outputs[pipeline])
            for pipeline in PIPELINE_NAMES
            if pipeline in pipeline_outputs
        )
    return ((None, volumes),)


def _write_pipeline_volumes(
    output_dir: Path,
    volumes: Mapping[str, np.ndarray],
    write_dat: Callable[[str | PathLike[str], np.ndarray], Path],
) -> list[Path]:
    written = []
    for name in VOLUME_NAMES:
        written.append(write_dat(output_dir / f"{name}.dat", volumes[name]))
    for source_name, output_name in SCANNER_VOLUME_NAMES:
        if source_name in volumes:
            written.append(write_dat(output_dir / f"{output_name}.dat", volumes[source_name]))
    written.extend(_write_thinning_diagnostic_volumes(output_dir, volumes, write_dat))
    return written


def _write_thinning_diagnostic_volumes(
    output_dir: Path,
    volumes: Mapping[str, np.ndarray],
    write_dat: Callable[[str | PathLike[str], np.ndarray], Path],
) -> list[Path]:
    if "fvt_reference_thinning_diagnostic" not in volumes:
        return []

    diagnostic_dir = output_dir / "thinning_diagnostic"
    return [
        write_dat(diagnostic_dir / f"{output_name}.dat", volumes[source_name])
        for source_name, output_name in THINNING_DIAGNOSTIC_VOLUME_NAMES
    ]


def write_case_skins_json(
    skin_outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_dir: str | PathLike[str],
) -> list[Path]:
    written = []
    output_root = Path(output_dir)
    for case_id, variants in skin_outputs.items():
        case_dir = _case_output_dir(output_root, case_id)
        for variant, skins_output in variants.items():
            output_dir_for_variant = _variant_output_dir(case_dir, variant, len(variants) == 1)
            for pipeline, pipeline_skins in _iter_pipeline_skin_outputs(skins_output):
                skin_dir = _pipeline_output_dir(output_dir_for_variant, pipeline)
                output_path = skin_dir / "skins.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(pipeline_skins, sort_keys=True) + "\n", encoding="utf-8"
                )
                written.append(output_path)
    return written


def _iter_pipeline_skin_outputs(
    skins_output: Mapping[str, Any],
) -> Sequence[tuple[str | None, Mapping[str, Any]]]:
    pipeline_outputs = skins_output.get(PIPELINE_OUTPUTS_KEY)
    if isinstance(pipeline_outputs, Mapping):
        unknown = sorted(set(pipeline_outputs).difference(PIPELINE_NAMES))
        if unknown:
            raise ValueError(f"unknown pipeline(s): {','.join(unknown)}")
        return tuple(
            (pipeline, pipeline_outputs[pipeline])
            for pipeline in PIPELINE_NAMES
            if pipeline in pipeline_outputs
        )
    return ((None, skins_output),)


def _pipeline_output_dir(output_dir_for_variant: Path, pipeline: str | None) -> Path:
    if pipeline is None:
        return output_dir_for_variant
    if pipeline not in {"oracle", "scanner"}:
        raise ValueError(f"unknown pipeline: {pipeline}")
    return output_dir_for_variant / pipeline


def write_case_figures(
    volume_outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    output_dir: str | PathLike[str],
    *,
    buffer_radius: float = 2.0,
) -> list[Path]:
    written = []
    output_root = Path(output_dir)
    for case_id, variants in volume_outputs.items():
        case_dir = _case_output_dir(output_root, case_id)
        for variant, volumes in variants.items():
            output_dir_for_variant = _variant_output_dir(case_dir, variant, len(variants) == 1)
            for pipeline, pipeline_volumes in _iter_pipeline_volume_outputs(volumes):
                figure_dir = _pipeline_output_dir(output_dir_for_variant, pipeline) / "figures"
                written.extend(
                    _write_pipeline_figures(
                        pipeline_volumes,
                        figure_dir,
                        case_id=case_id,
                        variant=variant,
                        pipeline=pipeline,
                        buffer_radius=buffer_radius,
                    )
                )
    return written


def _write_pipeline_figures(
    volumes: Mapping[str, np.ndarray],
    figures_dir: Path,
    *,
    case_id: str,
    variant: str,
    pipeline: str | None,
    buffer_radius: float,
) -> list[Path]:
    from pyosv import viz

    written = []
    title_parts = [case_id, variant]
    if pipeline is not None:
        title_parts.append(pipeline)
    title_prefix = " ".join(title_parts)
    indices = viz.select_center_slices(np.asarray(volumes["fvt_py"]).shape)
    for axis in ("i3", "i2", "i1"):
        index = indices[axis]
        for name in FIGURE_VOLUME_NAMES:
            figure_path = figures_dir / f"{name}_{axis}_center.png"
            written.append(
                viz.save_slice_panel(
                    figure_path,
                    [(name, viz.slice_2d(volumes[name], axis, index))],
                    title=f"{title_prefix} {name} {axis}=center",
                )
            )
        if "scanner_input" in volumes:
            for source_name, output_name in SCANNER_FIGURE_VOLUME_NAMES:
                figure_path = figures_dir / f"{output_name}_{axis}_center.png"
                written.append(
                    viz.save_slice_panel(
                        figure_path,
                        [
                            (
                                output_name,
                                viz.slice_2d(volumes[source_name], axis, index),
                            )
                        ],
                        title=f"{title_prefix} {output_name} {axis}=center",
                    )
                )
        if axis == "i3":
            written.append(
                viz.save_slice_panel(
                    figures_dir / "skin_mask_py_i3_center.png",
                    [
                        (
                            "skin_mask_py",
                            viz.slice_2d(volumes["skin_mask_py"], axis, index),
                        )
                    ],
                    title=f"{title_prefix} skin_mask_py {axis}=center",
                    clip_percentiles=(0.0, 100.0),
                )
            )
        written.append(
            viz.save_ridge_overlay_slice(
                figures_dir / f"truth_vs_fvt_overlay_{axis}_center.png",
                reference=volumes["truth_fault_mask"],
                candidate=volumes["fvt_py"],
                axis=axis,
                index=index,
                percentile=99.0,
                buffer_radius=buffer_radius,
                title=f"{title_prefix} truth vs fvt {axis}=center",
            )
        )
        written.append(
            viz.save_ridge_overlay_slice(
                figures_dir / f"truth_vs_skin_overlay_{axis}_center.png",
                reference=volumes["truth_fault_mask"],
                candidate=volumes["skin_mask_py"].astype(np.float32),
                axis=axis,
                index=index,
                percentile=99.0,
                buffer_radius=buffer_radius,
                title=f"{title_prefix} truth vs skin {axis}=center",
            )
        )
        if "scanner_ft" in volumes:
            written.append(
                viz.save_ridge_overlay_slice(
                    figures_dir / f"truth_vs_ft_scan_overlay_{axis}_center.png",
                    reference=volumes["truth_fault_mask"],
                    candidate=volumes["scanner_ft"],
                    axis=axis,
                    index=index,
                    percentile=99.0,
                    buffer_radius=buffer_radius,
                    title=f"{title_prefix} truth vs ft_scan {axis}=center",
                )
            )
            written.append(
                viz.save_ridge_overlay_slice(
                    figures_dir / f"truth_vs_ft_used_overlay_{axis}_center.png",
                    reference=volumes["truth_fault_mask"],
                    candidate=volumes["scanner_fet"],
                    axis=axis,
                    index=index,
                    percentile=99.0,
                    buffer_radius=buffer_radius,
                    title=f"{title_prefix} truth vs ft_used {axis}=center",
                )
            )
    written.extend(
        _write_thinning_diagnostic_figures(
            volumes,
            figures_dir.parent / "thinning_diagnostic",
            title_prefix=title_prefix,
            buffer_radius=buffer_radius,
        )
    )
    return written


def _write_thinning_diagnostic_figures(
    volumes: Mapping[str, np.ndarray],
    diagnostic_dir: Path,
    *,
    title_prefix: str,
    buffer_radius: float,
) -> list[Path]:
    if "fvt_reference_thinning_diagnostic" not in volumes:
        return []

    from pyosv import viz

    written = []
    indices = viz.select_center_slices(
        np.asarray(volumes["fvt_reference_thinning_diagnostic"]).shape
    )
    for axis in ("i3", "i2", "i1"):
        index = indices[axis]
        for source_name, output_name in (
            ("fvt_reference_thinning_diagnostic", "fvt_reference"),
            ("fvt_normal_thinning_diagnostic", "fvt_normal"),
            ("keep_reference_only_thinning_diagnostic", "keep_reference_only"),
            ("keep_normal_only_thinning_diagnostic", "keep_normal_only"),
        ):
            written.append(
                viz.save_ridge_overlay_slice(
                    diagnostic_dir / f"truth_vs_{output_name}_overlay_{axis}_center.png",
                    reference=volumes["truth_fault_mask"],
                    candidate=volumes[source_name],
                    axis=axis,
                    index=index,
                    percentile=99.0,
                    buffer_radius=buffer_radius,
                    title=f"{title_prefix} truth vs {output_name} {axis}=center",
                )
            )
        reference_slice = viz.slice_2d(volumes["fvt_reference_thinning_diagnostic"], axis, index)
        normal_slice = viz.slice_2d(volumes["fvt_normal_thinning_diagnostic"], axis, index)
        written.append(
            viz.save_slice_panel(
                diagnostic_dir / f"fvt_reference_vs_normal_{axis}_center.png",
                [
                    ("fvt_reference", reference_slice),
                    ("fvt_normal", normal_slice),
                    ("normal - reference", normal_slice - reference_slice),
                ],
                title=f"{title_prefix} fvt reference vs normal {axis}=center",
            )
        )
    return written


def _case_output_dir(output_dir: Path, case_id: str) -> Path:
    relative_case_path = PurePosixPath(case_id)
    if (
        relative_case_path.is_absolute()
        or not relative_case_path.parts
        or any(part in {"", ".", ".."} for part in relative_case_path.parts)
    ):
        raise ValueError(f"case_id must be a relative path inside output_dir: {case_id!r}")
    return output_dir.joinpath(*relative_case_path.parts)


def _variant_output_dir(case_dir: Path, variant: str, is_single_variant: bool) -> Path:
    if is_single_variant:
        return case_dir
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown variant: {variant}")
    return case_dir / variant
