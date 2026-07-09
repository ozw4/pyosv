from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pyosv.f3d_reference import F3D_ENV_VAR


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
RUN_ENV_VAR = "PYOSV_RUN_F3D_MULTICROP_PIPELINE"
REQUIRED_FILES = ("ep.dat", "fv.dat", "fvt.dat")


def _import_multicrop_module(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop("report_3d_f3d_multicrop", None)
    importlib.invalidate_caches()
    return importlib.import_module("report_3d_f3d_multicrop")


def _synthetic_reference_arrays(shape: tuple[int, int, int] = (8, 8, 8)) -> dict[str, np.ndarray]:
    ep = np.zeros(shape, dtype=np.float32)
    fl = np.zeros(shape, dtype=np.float32)
    fv = np.zeros(shape, dtype=np.float32)
    fvt = np.zeros(shape, dtype=np.float32)
    fl[2, 2, 2] = 3.0
    fl[5, 5, 5] = 2.0
    fv[2, 2, 2] = 3.0
    fv[5, 5, 5] = 2.0
    fvt[2, 2, 2] = 3.0
    fvt[5, 5, 5] = 2.0
    return {"ep.dat": ep, "fl.dat": fl, "fv.dat": fv, "fvt.dat": fvt}


def _synthetic_outputs(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    base = np.zeros(shape, dtype=np.float32)
    center = tuple(size // 2 for size in shape)
    base[center] = 1.0
    return {
        "ft_py.dat": base.copy(),
        "pt_py.dat": np.full(shape, 10.0, dtype=np.float32),
        "tt_py.dat": np.full(shape, 70.0, dtype=np.float32),
        "fet_py.dat": base.copy(),
        "fpt_py.dat": np.full(shape, 10.0, dtype=np.float32),
        "ftt_py.dat": np.full(shape, 70.0, dtype=np.float32),
        "fv_py.dat": base.copy(),
        "vp_py.dat": np.full(shape, 10.0, dtype=np.float32),
        "vt_py.dat": np.full(shape, 70.0, dtype=np.float32),
        "fvt_py.dat": base.copy(),
    }


def _gated_data_root() -> Path:
    if os.environ.get(RUN_ENV_VAR) != "1":
        pytest.skip(f"set {RUN_ENV_VAR}=1 to run the multi-crop F3 pipeline")

    root_text = os.environ.get(F3D_ENV_VAR)
    if root_text is None:
        pytest.skip(f"set {F3D_ENV_VAR} to the F3 reference data root")

    root = Path(root_text)
    if not root.is_dir():
        pytest.skip(f"{F3D_ENV_VAR} does not point to an existing directory: {root}")

    missing = [filename for filename in REQUIRED_FILES if not (root / filename).is_file()]
    if missing:
        pytest.skip(f"{F3D_ENV_VAR} is missing required files: {', '.join(missing)}")

    return root


def _assert_finite_or_none(value: object) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_or_none(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_finite_or_none(item)
        return
    if isinstance(value, int | float):
        assert np.isfinite(float(value))


def test_parser_defaults_and_explicit_centers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_multicrop_module(monkeypatch)

    defaults = module.build_parser().parse_args([])
    assert defaults.data_root is None
    assert defaults.output_json is None
    assert defaults.save_volumes is False
    assert defaults.save_figures is False
    assert defaults.figure_percentile == 99.0
    assert defaults.ridge_buffer_radius == 2.0
    assert defaults.write_markdown_index is False
    assert defaults.quality_validation is False
    assert defaults.quality_density_max_ratio == 2.0
    assert defaults.quality_edge_density_max_delta == 0.10
    assert defaults.quality_sparse_distance_max_delta == 5.0
    assert defaults.volume_dir is None
    assert defaults.workflow_mode == "reference"
    assert defaults.compare_workflows is False
    assert defaults.count == 3
    assert defaults.crop_shape == (128, 128, 100)
    assert defaults.interior_margin == 40
    assert defaults.center is None
    assert defaults.scanner_thin_mode == "reference"
    assert defaults.voter_thin_mode == "reference"
    assert defaults.voter_thin_mode_explicit is False
    assert defaults.reference_thin_sigma == 1.0
    assert defaults.remove_scanner_edge_effects is True
    assert defaults.final_normalization_smoothing is None
    assert defaults.surface_support_min_fraction is None
    assert defaults.surface_support_exponent is None
    assert "--final-normalization-smoothing" in module.build_parser().format_help()
    assert "--quality-validation" in module.build_parser().format_help()

    args = module.build_parser().parse_args(
        [
            "--crop-shape",
            "16,14,12",
            "--center",
            "2,3,4",
            "--center",
            "5,6,7",
            "--count",
            "2",
            "--interior-margin",
            "3",
            "--scanner-thin-mode",
            "reference",
            "--voter-thin-mode",
            "reference",
            "--reference-thin-sigma",
            "1.5",
            "--keep-scanner-edge-effects",
            "--final-normalization-smoothing",
            "1.0",
            "--workflow-mode",
            "quality",
            "--compare-workflows",
            "--quality-density-max-ratio",
            "1.5",
            "--quality-edge-density-max-delta",
            "0.2",
            "--quality-sparse-distance-max-delta",
            "4.0",
        ]
    )
    assert args.crop_shape == (16, 14, 12)
    assert args.center == [(2, 3, 4), (5, 6, 7)]
    assert args.count == 2
    assert args.interior_margin == 3
    assert args.scanner_thin_mode == "reference"
    assert args.voter_thin_mode == "reference"
    assert args.reference_thin_sigma == 1.5
    assert args.remove_scanner_edge_effects is False
    assert args.final_normalization_smoothing == 1.0
    assert args.workflow_mode == "quality"
    assert args.compare_workflows is True
    assert args.quality_density_max_ratio == 1.5
    assert args.quality_edge_density_max_delta == 0.2
    assert args.quality_sparse_distance_max_delta == 4.0
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--voter-thin-mode", "bad"])


def test_aggregate_reducer_on_synthetic_metric_dicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    crops = [
        {
            "normalized_correlation": {"interior": {"fv": 0.5, "fvt": 0.25}},
            "buffered_ridge_overlap": {"interior": {"fvt": {"buffered_recall": 1.0}}},
            "sparse_ridge_distance_metrics": {
                "interior": {"fvt": {"candidate_to_reference_median": None}}
            },
        },
        {
            "normalized_correlation": {"interior": {"fv": 1.0, "fvt": 0.75}},
            "buffered_ridge_overlap": {"interior": {"fvt": {"buffered_recall": 0.5}}},
            "sparse_ridge_distance_metrics": {
                "interior": {"fvt": {"candidate_to_reference_median": None}}
            },
        },
    ]

    aggregate = module.aggregate_crop_metrics(crops)

    assert aggregate["crop_count"] == 2
    assert aggregate["per_metric_mean"]["normalized_correlation.interior.fv"] == pytest.approx(0.75)
    assert aggregate["per_metric_median"]["normalized_correlation.interior.fvt"] == pytest.approx(
        0.5
    )
    assert aggregate["per_metric_min"][
        "buffered_ridge_overlap.interior.fvt.buffered_recall"
    ] == pytest.approx(0.5)
    assert aggregate["per_metric_max"][
        "buffered_ridge_overlap.interior.fvt.buffered_recall"
    ] == pytest.approx(1.0)
    assert (
        aggregate["per_metric_mean"][
            "sparse_ridge_distance_metrics.interior.fvt.candidate_to_reference_median"
        ]
        is None
    )


def test_deterministic_center_selection_order(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_multicrop_module(monkeypatch)
    fv = np.zeros((8, 8, 8), dtype=np.float32)
    fv[5, 5, 5] = 2.0
    fv[2, 2, 2] = 3.0
    fv[3, 3, 3] = 3.0

    centers = module.select_centers(
        fv,
        count=3,
        centers=None,
        percentile=0.0,
        min_separation=0.0,
        crop_shape=(4, 4, 4),
    )

    assert centers == [(2, 2, 2), (3, 3, 3), (5, 5, 5)]


def test_run_example_writes_json_and_uses_explicit_centers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_json = tmp_path / "outputs" / "metrics.json"
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )
    monkeypatch.setattr(
        module.crop_validation,
        "run_pipeline",
        lambda ep, **kwargs: _synthetic_outputs(ep.shape),
    )

    report = module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        pretty=True,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2), (5, 5, 5)],
    )

    assert output_json.is_file()
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["format_version"] == 2
    assert loaded["config"]["workflow_mode"] == "reference"
    assert loaded["config"]["crop_selection"]["source"] == "explicit_centers"
    assert loaded["config"]["crop_selection"]["selected_count"] == 2
    assert loaded["config"]["scanner"]["thin_mode"] == "reference"
    assert loaded["config"]["voter"]["thin_mode"] == "reference"
    assert loaded["config"]["scanner"]["reference_thin_sigma"] == 1.0
    assert loaded["config"]["scanner"]["remove_edge_effects"] is True
    assert loaded["config"]["voter"]["reference_thin_sigma"] == 1.0
    assert loaded["config"]["voter"]["final_normalization_smoothing"] == 0.0
    assert loaded["config"]["voter"]["surface_support_min_fraction"] == 0.0
    assert loaded["config"]["voter"]["surface_support_exponent"] == 0.0
    assert loaded["config"]["voter"]["surface_voting_boundary_policy"] == (
        "reference-like-i2-i3-interior"
    )
    assert [crop["crop_center"] for crop in loaded["crops"]] == [[2, 2, 2], [5, 5, 5]]
    assert loaded["aggregate"]["crop_count"] == 2
    consensus = loaded["consensus"]["workflows"]["reference"]
    assert consensus["crop_count"] == 2
    assert consensus["fvt_nonzero_fraction_mean"] == pytest.approx(1.0 / 216.0)
    assert consensus["fvt_nonzero_fraction_cv"] == pytest.approx(0.0)
    assert consensus["fv_nonzero_fraction_mean"] == pytest.approx(1.0 / 216.0)
    assert "fvt_reference_correlation_mean" in consensus
    assert consensus["fvt_edge_density_proxy_mean"] == pytest.approx(0.0)
    assert consensus["finite_failure_count"] == 0
    assert report == loaded
    assert not (data_root / "metrics.json").exists()
    _assert_finite_or_none(loaded)


def test_run_example_records_selected_thinning_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_json = tmp_path / "outputs" / "metrics.json"
    received_kwargs = {}
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )

    def fake_run_pipeline(ep: np.ndarray, **kwargs: object) -> dict[str, np.ndarray]:
        received_kwargs.update(kwargs)
        return _synthetic_outputs(ep.shape)

    monkeypatch.setattr(module.crop_validation, "run_pipeline", fake_run_pipeline)

    report = module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
        scanner_thin_mode="reference",
        voter_thin_mode="reference",
        reference_thin_sigma=1.25,
        remove_scanner_edge_effects=False,
        final_normalization_smoothing=1.0,
    )

    assert report["config"]["scanner"]["thin_mode"] == "reference"
    assert report["config"]["voter"]["thin_mode"] == "reference"
    assert report["config"]["scanner"]["reference_thin_sigma"] == 1.25
    assert report["config"]["scanner"]["remove_edge_effects"] is False
    assert report["config"]["voter"]["reference_thin_sigma"] == 1.25
    assert report["config"]["voter"]["final_normalization_smoothing"] == 1.0
    assert report["config"]["voter"]["surface_voting_boundary_policy"] == (
        "reference-like-i2-i3-interior"
    )
    assert received_kwargs["scanner_thin_mode"] == "reference"
    assert received_kwargs["voter_thin_mode"] == "reference"
    assert received_kwargs["reference_thin_sigma"] == 1.25
    assert received_kwargs["remove_scanner_edge_effects"] is False
    assert received_kwargs["final_normalization_smoothing"] == 1.0


def test_compare_workflows_runs_same_centers_and_reports_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_json = tmp_path / "outputs" / "metrics.json"
    received_kwargs: list[dict[str, object]] = []
    figure_dirs: list[Path] = []
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )

    def fake_run_pipeline(ep: np.ndarray, **kwargs: object) -> dict[str, np.ndarray]:
        received_kwargs.append(dict(kwargs))
        return _synthetic_outputs(ep.shape)

    def fake_write_figures(output_dir: Path, **kwargs: object) -> dict[str, object]:
        figure_dirs.append(Path(output_dir))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {"directory": Path(output_dir).relative_to(output_json.parent).as_posix()}

    monkeypatch.setattr(module.crop_validation, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(module.crop_validation, "require_figure_support", lambda: None)
    monkeypatch.setattr(module.crop_validation, "write_crop_figures", fake_write_figures)

    report = module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        compare_workflows=True,
        save_volumes=True,
        save_figures=True,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
        pretty=True,
    )

    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded == report
    assert set(loaded["workflows"]) == {"reference", "quality"}
    assert set(loaded["consensus"]["workflows"]) == {"reference", "quality"}
    assert loaded["workflows"]["reference"]["crops"][0]["crop_center"] == [2, 2, 2]
    assert loaded["workflows"]["quality"]["crops"][0]["crop_center"] == [2, 2, 2]
    assert loaded["workflows"]["reference"]["config"]["voter"]["thin_mode"] == "reference"
    assert loaded["workflows"]["quality"]["config"]["voter"]["thin_mode"] == "hybrid_v2"
    assert loaded["workflows"]["quality"]["config"]["voter"]["surface_support_min_fraction"] == 0.0
    assert loaded["workflows"]["quality"]["config"]["voter"]["surface_support_exponent"] == 0.0
    assert loaded["workflow_delta"]["quality_vs_reference"]
    delta = loaded["workflow_delta"]["quality_vs_reference"]["per_metric_mean"]
    assert "normalized_correlation.interior.fv" in delta
    assert "normalized_correlation.interior.fvt" in delta
    assert "buffered_ridge_overlap.interior.fvt.buffered_recall" in delta
    assert "sparse_ridge_distance_metrics.interior.fvt.candidate_to_reference_median" in delta
    comparison = loaded["consensus"]["workflow_comparison"]["quality_minus_reference"]
    assert "fvt_nonzero_fraction_delta_mean" in comparison
    assert "fvt_reference_correlation_delta_mean" in comparison
    assert "fvt_edge_density_proxy_delta_mean" in comparison
    assert "fvt_sparse_distance_p95_delta_mean" in comparison
    assert comparison["finite_failure_count_delta"] == 0
    assert loaded["quality_validation"]["role"] == "truthless_external_smoke"
    assert loaded["quality_validation"]["workflow_comparison_available"] is True
    assert loaded["quality_validation"]["passed"] is True
    assert set(loaded["quality_validation"]["checks"]) == {
        "finite_metrics",
        "quality_density_not_exploding",
        "quality_edge_density_not_exploding",
        "quality_sparse_distance_not_worse",
        "crop_to_crop_stability",
    }
    _assert_finite_or_none(loaded["workflow_delta"]["quality_vs_reference"])
    assert received_kwargs[0]["voter_thin_mode"] == "reference"
    assert received_kwargs[1]["voter_thin_mode"] == "hybrid_v2"
    assert (output_json.parent / "volumes" / "reference" / "crop_001").is_dir()
    assert (output_json.parent / "volumes" / "quality" / "crop_001").is_dir()
    assert figure_dirs == [
        output_json.parent / "figures" / "reference" / "crop_001",
        output_json.parent / "figures" / "quality" / "crop_001",
    ]


def test_compare_workflows_honors_explicit_voter_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_json = tmp_path / "outputs" / "metrics.json"
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )
    monkeypatch.setattr(
        module.crop_validation,
        "run_pipeline",
        lambda ep, **kwargs: _synthetic_outputs(ep.shape),
    )

    report = module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        compare_workflows=True,
        voter_thin_mode="reference",
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
    )

    assert report["workflows"]["quality"]["config"]["voter"]["thin_mode"] == "reference"


def test_consensus_summary_handles_single_crop_and_finite_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    crops = [
        {
            "pyosv": {
                "fv": {"nonzero_fraction": 0.0},
                "fvt": {"nonzero_fraction": 0.0},
            },
            "pyosv_interior": {"fvt": {"nonzero_fraction": 0.0}},
            "normalized_correlation": {"interior": {"fvt": 0.25}},
            "buffered_ridge_overlap": {
                "interior": {
                    "fvt": {
                        "buffered_precision": 0.5,
                        "buffered_recall": 0.75,
                    }
                }
            },
            "sparse_ridge_distance_metrics": {
                "interior": {"fvt": {"candidate_to_reference_p95": 3.0}}
            },
            "finite_checks": {
                "pyosv": {
                    "fv_py": {
                        "size": 8,
                        "finite_count": 7,
                        "nan_count": 1,
                        "posinf_count": 0,
                        "neginf_count": 0,
                    }
                }
            },
        }
    ]

    consensus = module.build_consensus_summary(crops)

    assert consensus["crop_count"] == 1
    assert consensus["fvt_nonzero_fraction_mean"] == 0.0
    assert consensus["fvt_nonzero_fraction_std"] == 0.0
    assert consensus["fvt_nonzero_fraction_cv"] == 0.0
    assert consensus["fv_nonzero_fraction_cv"] == 0.0
    assert consensus["fvt_reference_correlation_std"] == 0.0
    assert consensus["fvt_buffered_overlap_precision_mean"] == 0.5
    assert consensus["fvt_buffered_overlap_recall_mean"] == 0.75
    assert consensus["fvt_sparse_distance_p95_mean"] == 3.0
    assert consensus["fvt_edge_density_proxy_mean"] == 0.0
    assert consensus["finite_failure_count"] == 1


def test_quality_validation_summary_passes_within_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    consensus = {
        "workflows": {
            "reference": {
                "crop_count": 3,
                "fvt_nonzero_fraction_mean": 0.10,
                "fvt_nonzero_fraction_cv": 0.20,
                "fv_nonzero_fraction_cv": 0.10,
                "fvt_edge_density_proxy_mean": 0.01,
                "fvt_sparse_distance_p95_mean": 4.0,
                "finite_failure_count": 0,
            },
            "quality": {
                "crop_count": 3,
                "fvt_nonzero_fraction_mean": 0.15,
                "fvt_nonzero_fraction_cv": 0.25,
                "fv_nonzero_fraction_cv": 0.20,
                "fvt_edge_density_proxy_mean": 0.05,
                "fvt_sparse_distance_p95_mean": 6.0,
                "finite_failure_count": 0,
            },
        },
        "workflow_comparison": {
            "quality_minus_reference": {
                "fvt_nonzero_fraction_delta_mean": 0.05,
                "fvt_edge_density_proxy_delta_mean": 0.04,
                "fvt_sparse_distance_p95_delta_mean": 2.0,
                "finite_failure_count_delta": 0,
            }
        },
    }

    validation = module.build_quality_validation_summary(consensus, compare_workflows=True)

    assert validation["role"] == "truthless_external_smoke"
    assert validation["crop_count"] == 3
    assert validation["workflow_comparison_available"] is True
    assert validation["passed"] is True
    assert validation["reasons"] == []
    assert validation["checks"]["finite_metrics"]["failure_count"] == 0
    assert validation["checks"]["quality_density_not_exploding"]["value"] == pytest.approx(1.5)
    assert validation["checks"]["quality_edge_density_not_exploding"]["value"] == pytest.approx(
        0.04
    )
    assert validation["checks"]["quality_sparse_distance_not_worse"]["value"] == pytest.approx(2.0)
    assert validation["checks"]["crop_to_crop_stability"]["value"] == pytest.approx(0.25)


def test_quality_validation_summary_fails_on_finite_failures_and_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    consensus = {
        "workflows": {
            "reference": {
                "crop_count": 3,
                "fvt_nonzero_fraction_mean": 0.10,
                "fvt_nonzero_fraction_cv": 0.20,
                "fv_nonzero_fraction_cv": 0.10,
                "finite_failure_count": 0,
            },
            "quality": {
                "crop_count": 3,
                "fvt_nonzero_fraction_mean": 0.31,
                "fvt_nonzero_fraction_cv": 3.0,
                "fv_nonzero_fraction_cv": 0.20,
                "finite_failure_count": 1,
            },
        },
        "workflow_comparison": {
            "quality_minus_reference": {
                "fvt_edge_density_proxy_delta_mean": 0.11,
                "fvt_sparse_distance_p95_delta_mean": 5.5,
            }
        },
    }

    validation = module.build_quality_validation_summary(consensus, compare_workflows=True)

    assert validation["passed"] is False
    assert validation["checks"]["finite_metrics"]["passed"] is False
    assert validation["checks"]["quality_density_not_exploding"]["passed"] is False
    assert validation["checks"]["quality_edge_density_not_exploding"]["passed"] is False
    assert validation["checks"]["quality_sparse_distance_not_worse"]["passed"] is False
    assert validation["checks"]["crop_to_crop_stability"]["passed"] is False
    assert any("non-finite" in reason for reason in validation["reasons"])
    assert any("density ratio" in reason for reason in validation["reasons"])
    assert any("edge-density" in reason for reason in validation["reasons"])
    assert any("sparse ridge distance" in reason for reason in validation["reasons"])
    assert any("CV" in reason for reason in validation["reasons"])


def test_quality_validation_summary_single_workflow_skips_comparison_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    consensus = {
        "crop_count": 2,
        "fvt_nonzero_fraction_cv": 0.5,
        "fv_nonzero_fraction_cv": 0.25,
        "finite_failure_count": 0,
    }

    validation = module.build_quality_validation_summary(
        consensus,
        workflow_mode="quality",
        enabled=True,
    )

    assert validation["passed"] is True
    assert validation["crop_count"] == 2
    assert validation["workflow_comparison_available"] is False
    assert validation["checks"]["finite_metrics"]["failure_count"] == 0
    assert validation["checks"]["finite_metrics"]["passed"] is True
    assert validation["checks"]["crop_to_crop_stability"]["passed"] is True
    assert validation["checks"]["crop_to_crop_stability"]["value"] == pytest.approx(0.5)
    assert validation["checks"]["quality_density_not_exploding"]["skipped"] is True
    assert validation["checks"]["quality_edge_density_not_exploding"]["skipped"] is True
    assert validation["checks"]["quality_sparse_distance_not_worse"]["skipped"] is True


def test_compare_workflows_honors_explicit_surface_support_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_json = tmp_path / "outputs" / "metrics.json"
    received_kwargs: list[dict[str, object]] = []
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )

    def fake_run_pipeline(ep: np.ndarray, **kwargs: object) -> dict[str, np.ndarray]:
        received_kwargs.append(dict(kwargs))
        return _synthetic_outputs(ep.shape)

    monkeypatch.setattr(module.crop_validation, "run_pipeline", fake_run_pipeline)

    report = module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        compare_workflows=True,
        surface_support_min_fraction=0.25,
        surface_support_exponent=2.0,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
    )

    reference_voter = report["workflows"]["reference"]["config"]["voter"]
    quality_voter = report["workflows"]["quality"]["config"]["voter"]
    assert reference_voter["surface_support_min_fraction"] == 0.25
    assert reference_voter["surface_support_exponent"] == 2.0
    assert quality_voter["surface_support_min_fraction"] == 0.25
    assert quality_voter["surface_support_exponent"] == 2.0
    assert received_kwargs[0]["surface_support_min_fraction"] == 0.25
    assert received_kwargs[0]["surface_support_exponent"] == 2.0
    assert received_kwargs[1]["surface_support_min_fraction"] == 0.25
    assert received_kwargs[1]["surface_support_exponent"] == 2.0


def test_visual_report_markdown_compare_report_includes_workflow_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    report = {
        "data_root": "/tmp/f3_reference",
        "config": {
            "comparison": "scan_vote_thin_fv_fvt_multicrop",
            "compare_workflows": True,
            "crop_selection": {"crop_shape": [6, 6, 6], "selected_count": 1},
        },
        "workflows": {
            "reference": {
                "config": {
                    "workflow_mode": "reference",
                    "scanner": {"thin_mode": "reference", "remove_edge_effects": True},
                    "voter": {
                        "thin_mode": "reference",
                        "surface_support_min_fraction": 0.0,
                        "surface_support_exponent": 0.0,
                    },
                },
                "crops": [
                    {
                        "index": 1,
                        "crop_center": [2, 2, 2],
                        "crop_slices": [{"axis": "i3", "start": 0, "stop": 6}],
                        "normalized_correlation": {"interior": {"fv": 0.9, "fvt": 0.8}},
                        "figures": {
                            "files": {
                                "scanner_fl_vs_ftpy": {
                                    "i3": "figures/reference/crop_001/scanner.png"
                                }
                            }
                        },
                    }
                ],
            },
            "quality": {
                "config": {
                    "workflow_mode": "quality",
                    "scanner": {"thin_mode": "reference", "remove_edge_effects": True},
                    "voter": {
                        "thin_mode": "hybrid_v2",
                        "surface_support_min_fraction": 0.0,
                        "surface_support_exponent": 0.0,
                    },
                },
                "crops": [
                    {
                        "index": 1,
                        "crop_center": [2, 2, 2],
                        "crop_slices": [{"axis": "i3", "start": 0, "stop": 6}],
                        "normalized_correlation": {"interior": {"fv": 0.95, "fvt": 0.85}},
                        "figures": {
                            "files": {"fv_ref_vs_py": {"i3": "figures/quality/crop_001/fv.png"}}
                        },
                    }
                ],
            },
        },
        "workflow_delta": {
            "quality_vs_reference": {
                "per_metric_mean": {"normalized_correlation.interior.fv": 0.05}
            }
        },
        "consensus": {
            "workflows": {
                "reference": {
                    "crop_count": 1,
                    "fvt_nonzero_fraction_mean": 0.1,
                    "fvt_nonzero_fraction_cv": 0.0,
                    "fv_nonzero_fraction_mean": 0.2,
                    "fv_nonzero_fraction_cv": 0.0,
                    "fvt_reference_correlation_mean": 0.8,
                    "fvt_buffered_overlap_precision_mean": 1.0,
                    "fvt_buffered_overlap_recall_mean": 1.0,
                    "fvt_sparse_distance_p95_mean": 0.0,
                    "fvt_edge_density_proxy_mean": 0.0,
                    "finite_failure_count": 0,
                },
                "quality": {
                    "crop_count": 1,
                    "fvt_nonzero_fraction_mean": 0.15,
                    "fvt_nonzero_fraction_cv": 0.0,
                    "fv_nonzero_fraction_mean": 0.25,
                    "fv_nonzero_fraction_cv": 0.0,
                    "fvt_reference_correlation_mean": 0.85,
                    "fvt_buffered_overlap_precision_mean": 1.0,
                    "fvt_buffered_overlap_recall_mean": 1.0,
                    "fvt_sparse_distance_p95_mean": 0.0,
                    "fvt_edge_density_proxy_mean": 0.0,
                    "finite_failure_count": 0,
                },
            },
            "workflow_comparison": {
                "quality_minus_reference": {
                    "fvt_nonzero_fraction_delta_mean": 0.05,
                    "fvt_reference_correlation_delta_mean": 0.05,
                    "fvt_edge_density_proxy_delta_mean": 0.0,
                    "fvt_sparse_distance_p95_delta_mean": 0.0,
                    "finite_failure_count_delta": 0,
                }
            },
        },
        "quality_validation": {
            "role": "truthless_external_smoke",
            "crop_count": 1,
            "workflow_comparison_available": True,
            "checks": {
                "finite_metrics": {"passed": True, "failure_count": 0, "threshold": 0},
                "quality_density_not_exploding": {
                    "passed": True,
                    "value": 1.5,
                    "threshold": 2.0,
                },
                "quality_edge_density_not_exploding": {
                    "passed": True,
                    "value": 0.0,
                    "threshold": 0.10,
                },
                "quality_sparse_distance_not_worse": {
                    "passed": True,
                    "value": 0.0,
                    "threshold": 5.0,
                },
                "crop_to_crop_stability": {
                    "passed": True,
                    "value": 0.0,
                    "threshold": 2.0,
                },
            },
            "passed": True,
            "reasons": [],
        },
    }

    markdown = module.visual_report_markdown(report)

    assert "### reference" in markdown
    assert "### quality" in markdown
    assert "## reference Crop Metrics" in markdown
    assert "## quality Crop Metrics" in markdown
    assert "crop_001" in markdown
    assert "figures/reference/crop_001/scanner.png" in markdown
    assert "figures/quality/crop_001/fv.png" in markdown
    assert "voter_thin_mode: `reference`" in markdown
    assert "voter_thin_mode: `hybrid_v2`" in markdown
    assert "surface_support_min_fraction: `0.0`" in markdown
    assert "## Consensus" in markdown
    assert "quality_minus_reference consensus delta" in markdown
    assert "fvt_edge_density_proxy_delta_mean" in markdown
    assert "## Quality Validation" in markdown
    assert "truthless external smoke" in markdown
    assert "synthetic promotion gate" in markdown
    assert "quality_density_not_exploding" in markdown
    assert "reference Figures" in markdown
    assert "quality Figures" in markdown
    assert "quality_vs_reference per_metric_mean" in markdown
    assert "No PNG figures were written for this run." not in markdown


def test_compare_workflows_writes_markdown_index_with_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_json = tmp_path / "outputs" / "metrics.json"
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )
    monkeypatch.setattr(
        module.crop_validation,
        "run_pipeline",
        lambda ep, **kwargs: _synthetic_outputs(ep.shape),
    )
    monkeypatch.setattr(module.crop_validation, "require_figure_support", lambda: None)

    def fake_write_figures(output_dir: Path, **kwargs: object) -> dict[str, object]:
        relative_dir = Path(output_dir).relative_to(output_json.parent).as_posix()
        return {
            "directory": relative_dir,
            "files": {
                "scanner_fl_vs_ftpy": {"i3": f"{relative_dir}/scanner_fl_vs_ftpy_i3.png"},
                "fv": {"mip": f"{relative_dir}/fv_mip.png"},
            },
        }

    monkeypatch.setattr(module.crop_validation, "write_crop_figures", fake_write_figures)

    module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        compare_workflows=True,
        save_figures=True,
        write_markdown_index=True,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
    )

    markdown_path = output_json.parent / "visual_report.md"
    markdown = markdown_path.read_text(encoding="utf-8")

    assert markdown_path.is_file()
    assert "## reference Crop Metrics" in markdown
    assert "## quality Crop Metrics" in markdown
    assert "## Consensus" in markdown
    assert "quality_minus_reference consensus delta" in markdown
    assert "## Quality Validation" in markdown
    assert "quality_density_not_exploding" in markdown
    assert "## reference Figures" in markdown
    assert "## quality Figures" in markdown
    assert "crop_001" in markdown
    assert "figures/reference/crop_001/scanner_fl_vs_ftpy_i3.png" in markdown
    assert "figures/quality/crop_001/scanner_fl_vs_ftpy_i3.png" in markdown
    assert "No PNG figures were written for this run." not in markdown


def test_save_volumes_writes_crop_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_json = tmp_path / "outputs" / "metrics.json"
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )
    monkeypatch.setattr(
        module.crop_validation,
        "run_pipeline",
        lambda ep, **kwargs: _synthetic_outputs(ep.shape),
    )

    module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        save_volumes=True,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
    )

    crop_dir = output_json.parent / "volumes" / "crop_001"
    for name in module.crop_validation.VOLUME_NAMES:
        assert (crop_dir / name).is_file()


def test_visual_report_writes_markdown_pngs_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    module = _import_multicrop_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_json = tmp_path / "outputs" / "metrics.json"
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )
    monkeypatch.setattr(
        module.crop_validation,
        "run_pipeline",
        lambda ep, **kwargs: _synthetic_outputs(ep.shape),
    )

    report = module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        save_figures=True,
        figure_percentile=99.0,
        ridge_buffer_radius=2,
        write_markdown_index=True,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
        pretty=True,
    )

    markdown_path = output_json.parent / "visual_report.md"
    figures_dir = output_json.parent / "crop_001" / "figures"
    markdown = markdown_path.read_text(encoding="utf-8")
    loaded = json.loads(output_json.read_text(encoding="utf-8"))

    assert markdown_path.is_file()
    assert "crop_001" in markdown
    assert "normalized_correlation" in markdown
    assert "scanner_thin_mode: `reference`" in markdown
    assert "scanner_edge_effect_removal: `True`" in markdown
    assert "voter_thin_mode: `reference`" in markdown
    assert "reference_thin_sigma: `1.0`" in markdown
    assert "surface_voting_boundary_policy: `reference-like-i2-i3-interior`" in markdown
    assert "## Consensus" in markdown
    assert "fvt_edge_density_proxy_mean" in markdown
    assert "](crop_001/figures/" in markdown
    assert ".png)" in markdown
    assert (figures_dir / "scanner_fl_vs_ftpy_i3_3.png").is_file()
    assert (figures_dir / "fv_mip.png").is_file()
    assert loaded["config"]["visualization"] == {
        "figure_percentile": 99.0,
        "figure_slices": "center",
        "markdown_index": "visual_report.md",
        "ridge_buffer_radius": 2.0,
        "save_figures": True,
        "write_markdown_index": True,
    }
    assert loaded["crops"][0]["figures"] == report["crops"][0]["figures"]


def test_save_figures_requires_output_json(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_multicrop_module(monkeypatch)

    with pytest.raises(ValueError, match="requires --output-json"):
        module.run_example(
            data_root_arg="/tmp/f3_reference",
            save_figures=True,
            crop_shape=(6, 6, 6),
            interior_margin=1,
        )


def test_main_reports_viz_extra_when_matplotlib_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _import_multicrop_module(monkeypatch)

    def raise_missing_matplotlib() -> None:
        raise ValueError('matplotlib is required. Install it with `pip install "pyosv[viz]"`.')

    monkeypatch.setattr(module.crop_validation, "require_figure_support", raise_missing_matplotlib)

    exit_code = module.main(
        [
            "--data-root",
            str(tmp_path / "f3_reference"),
            "--output-json",
            str(tmp_path / "outputs" / "metrics.json"),
            "--save-figures",
            "--crop-shape",
            "6,6,6",
            "--interior-margin",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "pyosv[viz]" in captured.err


def test_output_path_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_multicrop_module(monkeypatch)
    data_root = tmp_path / "f3_reference"

    with pytest.raises(ValueError, match="--output-json must not be inside"):
        module.run_example(
            data_root_arg=data_root,
            output_json=data_root / "outputs" / "metrics.json",
            crop_shape=(6, 6, 6),
            interior_margin=1,
        )

    with pytest.raises(ValueError, match="--volume-dir must not be inside"):
        module.run_example(
            data_root_arg=data_root,
            save_volumes=True,
            volume_dir=data_root / "volumes",
            crop_shape=(6, 6, 6),
            interior_margin=1,
        )


def test_save_volumes_requires_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_multicrop_module(monkeypatch)

    with pytest.raises(ValueError, match="requires --volume-dir or --output-json"):
        module.run_example(
            data_root_arg="/tmp/f3_reference",
            save_volumes=True,
            crop_shape=(6, 6, 6),
            interior_margin=1,
        )


def test_json_serialization_converts_nonfinite_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_multicrop_module(monkeypatch)

    loaded = json.loads(
        module.report_to_json(
            {"finite": np.float32(1.25), "nan": float("nan"), "inf": np.float64(np.inf)}
        )
    )

    assert loaded == {"finite": 1.25, "inf": None, "nan": None}


def test_import_does_not_run_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_multicrop_module(monkeypatch)

    assert callable(module.build_parser)
    assert callable(module.main)
    assert callable(module.run_example)


def test_cli_help_succeeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _import_multicrop_module(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])

    assert excinfo.value.code == 0
    assert "--quality-validation" in capsys.readouterr().out


@pytest.mark.f3d_reference
def test_gated_real_data_multicrop_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _gated_data_root()
    module = _import_multicrop_module(monkeypatch)

    report = module.run_example(
        data_root_arg=data_root,
        count=1,
        crop_shape=(48, 48, 48),
        interior_margin=12,
        percentile=99.9,
        min_separation=24.0,
    )

    assert len(report["crops"]) == 1
    assert report["aggregate"]["crop_count"] == 1
    assert "normalized_correlation.interior.fv" in report["aggregate"]["per_metric_mean"]
    _assert_finite_or_none(report)
