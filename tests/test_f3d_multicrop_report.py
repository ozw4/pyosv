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
    assert defaults.volume_dir is None
    assert defaults.count == 3
    assert defaults.crop_shape == (128, 128, 100)
    assert defaults.interior_margin == 40
    assert defaults.center is None

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
        ]
    )
    assert args.crop_shape == (16, 14, 12)
    assert args.center == [(2, 3, 4), (5, 6, 7)]
    assert args.count == 2
    assert args.interior_margin == 3


def test_aggregate_reducer_on_synthetic_metric_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert loaded["format_version"] == 1
    assert loaded["config"]["crop_selection"]["source"] == "explicit_centers"
    assert loaded["config"]["crop_selection"]["selected_count"] == 2
    assert [crop["crop_center"] for crop in loaded["crops"]] == [[2, 2, 2], [5, 5, 5]]
    assert loaded["aggregate"]["crop_count"] == 2
    assert report == loaded
    assert not (data_root / "metrics.json").exists()
    _assert_finite_or_none(loaded)


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


def test_json_serialization_converts_nonfinite_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
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
