from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pyosv.cells import FaultCell
from pyosv.f3d_reference import F3D_ENV_VAR


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
RUN_ENV_VAR = "PYOSV_RUN_F3D_SEED_DIAGNOSTICS"
REQUIRED_FILES = ("ep.dat", "fv.dat", "fvt.dat")


def _import_seed_module(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop("report_3d_f3d_seed_diagnostics", None)
    importlib.invalidate_caches()
    return importlib.import_module("report_3d_f3d_seed_diagnostics")


def _synthetic_reference_arrays(shape: tuple[int, int, int] = (8, 8, 8)) -> dict[str, np.ndarray]:
    ep = np.zeros(shape, dtype=np.float32)
    fv = np.zeros(shape, dtype=np.float32)
    fvt = np.zeros(shape, dtype=np.float32)
    ep[2, 2, 2] = 1.0
    fv[2, 2, 2] = 3.0
    fv[5, 5, 5] = 2.0
    fvt[2, 2, 2] = 3.0
    fvt[5, 5, 5] = 2.0
    return {"ep.dat": ep, "fv.dat": fv, "fvt.dat": fvt}


def _pipeline_outputs(module: object, shape: tuple[int, int, int]) -> dict[str, dict[str, object]]:
    outputs = {}
    for case in module.build_case_definitions(("current",), ("normal", "reference")):
        fet = np.zeros(shape, dtype=np.float32)
        fet[2, 2, 2] = 0.8
        fet[5, 5, 5] = 0.5
        seeds = [
            FaultCell(2, 2, 2, 0.8, 10.0, 70.0),
            FaultCell(5, 5, 5, 0.5, 10.0, 70.0),
        ]
        outputs[case["name"]] = {
            "fet_py.dat": fet,
            "fpt_py.dat": np.full(shape, 10.0, dtype=np.float32),
            "ftt_py.dat": np.full(shape, 70.0, dtype=np.float32),
            "seeds": seeds,
        }
    return outputs


def _gated_data_root() -> Path:
    if os.environ.get(RUN_ENV_VAR) != "1":
        pytest.skip(f"set {RUN_ENV_VAR}=1 to run the F3 seed diagnostics pipeline")

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


def test_seed_mask_construction_from_fault_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_seed_module(monkeypatch)
    seeds = [
        FaultCell(1.0, 2.0, 3.0, 0.7, 10.0, 70.0),
        FaultCell(4.0, 0.0, 1.0, 0.5, 10.0, 70.0),
    ]

    mask = module.seeds_to_mask(seeds, (4, 3, 5))

    assert mask.dtype == np.bool_
    assert int(np.count_nonzero(mask)) == 2
    assert mask[3, 2, 1]
    assert mask[1, 0, 4]
    with pytest.raises(ValueError, match="outside shape"):
        module.seeds_to_mask([FaultCell(9, 0, 0, 1.0, 0.0, 0.0)], (4, 3, 5))


def test_distance_metric_behavior_for_synthetic_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_seed_module(monkeypatch)
    reference = np.zeros((5, 5, 5), dtype=bool)
    seed = np.zeros((5, 5, 5), dtype=bool)
    reference[0, 0, 0] = True
    seed[0, 0, 3] = True

    ref_to_seed = module.mask_distance_summary(
        source_mask=reference,
        target_mask=seed,
        source_name="reference",
        target_name="seed",
    )
    seed_to_ref = module.mask_distance_summary(
        source_mask=seed,
        target_mask=reference,
        source_name="seed",
        target_name="reference",
    )
    empty_to_seed = module.mask_distance_summary(
        source_mask=np.zeros((5, 5, 5), dtype=bool),
        target_mask=seed,
        source_name="reference",
        target_name="seed",
    )

    assert ref_to_seed["reference_count"] == 1
    assert ref_to_seed["seed_count"] == 1
    assert ref_to_seed["mean"] == pytest.approx(3.0)
    assert ref_to_seed["p95"] == pytest.approx(3.0)
    assert seed_to_ref["mean"] == pytest.approx(3.0)
    assert empty_to_seed["reference_count"] == 0
    assert empty_to_seed["seed_count"] == 1
    assert empty_to_seed["mean"] is None
    assert empty_to_seed["median"] is None


def test_cli_parsing_of_backend_mode_combinations(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_seed_module(monkeypatch)

    defaults = module.build_parser().parse_args([])
    assert defaults.scanner_backends == ("current", "reference-like")
    assert defaults.scanner_thin_modes == ("normal", "reference")
    assert defaults.reference_percentile == 99.0
    assert defaults.count == 3
    assert defaults.crop_shape == (64, 64, 64)
    assert defaults.interior_margin == 16

    args = module.build_parser().parse_args(
        [
            "--scanner-backends",
            "current",
            "--scanner-thin-modes",
            "reference,normal",
            "--reference-percentile",
            "98.5",
        ]
    )
    assert args.scanner_backends == ("current",)
    assert args.scanner_thin_modes == ("reference", "normal")
    assert args.reference_percentile == 98.5

    cases = module.build_case_definitions(args.scanner_backends, args.scanner_thin_modes)
    assert [case["name"] for case in cases] == ["current_reference", "current_normal"]
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--scanner-backends", "bad"])


def test_run_example_writes_seed_diagnostic_json_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_seed_module(monkeypatch)
    output_json = tmp_path / "outputs" / "metrics.json"
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )
    monkeypatch.setattr(
        module,
        "run_seed_diagnostic_pipeline",
        lambda ep, **kwargs: _pipeline_outputs(module, ep.shape),
    )

    report = module.run_example(
        data_root_arg=tmp_path / "f3_reference",
        output_json=output_json,
        pretty=True,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
        scanner_backends=("current",),
        scanner_thin_modes=("normal", "reference"),
        reference_percentile=100.0,
    )

    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    cases = loaded["crops"][0]["cases"]
    diagnostics = cases["current_normal"]["seed_diagnostics"]
    assert report == loaded
    assert loaded["format_version"] == 1
    assert loaded["config"]["comparison"] == "f3d_seed_diagnostics"
    assert [case["name"] for case in loaded["config"]["cases"]] == [
        "current_normal",
        "current_reference",
    ]
    assert set(cases) == {"current_normal", "current_reference"}
    assert diagnostics["seed_count"] == 2
    assert diagnostics["seed_density"] == pytest.approx(2 / 216)
    assert diagnostics["seed_likelihood_percentiles"]["p50"] is not None
    assert diagnostics["reference_high_counts"]["fv"] == 1
    assert diagnostics["distance"]["reference_high_fv_to_seed"]["reference_count"] == 1
    assert diagnostics["distance"]["reference_high_fv_to_seed"]["seed_count"] == 2
    assert (
        "seed_diagnostics.seed_count"
        in loaded["aggregate"]["cases"]["current_normal"]["metric_paths"]
    )


def test_current_backend_works_without_reference_like(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_seed_module(monkeypatch)
    ep = np.zeros((4, 4, 4), dtype=np.float32)
    ep[2, 2, 2] = 1.0

    outputs = module.run_seed_diagnostic_pipeline(
        ep,
        scanner_backends=("current",),
        scanner_thin_modes=("normal",),
        sigma1=2.0,
        sigma2=2.0,
        phi_min=0.0,
        phi_max=0.0,
        theta_min=70.0,
        theta_max=70.0,
        ru=2,
        rv=2,
        rw=2,
        d=1,
        fm=0.0,
        reference_thin_sigma=1.0,
    )

    assert set(outputs) == {"current_normal"}
    assert isinstance(outputs["current_normal"]["seeds"], list)


def test_reference_like_backend_unavailable_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_seed_module(monkeypatch)

    class ScannerWithoutReferenceLike:
        def scan(self, *args: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            shape = np.asarray(args[-1]).shape
            return (
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
            )

    with pytest.raises(ValueError, match="reference-like scanner backend is unavailable"):
        module._scan_backend(
            ScannerWithoutReferenceLike(),
            backend="reference-like",
            phi_min=0.0,
            phi_max=0.0,
            theta_min=70.0,
            theta_max=70.0,
            ep=np.zeros((4, 4, 4), dtype=np.float32),
        )


def test_output_path_safety_rejects_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_seed_module(monkeypatch)
    data_root = tmp_path / "f3_reference"

    with pytest.raises(ValueError, match="--output-json must not be inside"):
        module.run_example(
            data_root_arg=data_root,
            output_json=data_root / "outputs" / "metrics.json",
            crop_shape=(6, 6, 6),
            interior_margin=1,
            scanner_backends=("current",),
            scanner_thin_modes=("normal",),
        )

    assert not (data_root / "outputs" / "metrics.json").exists()


def test_visual_report_writes_markdown_and_seed_overlay_pngs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    module = _import_seed_module(monkeypatch)
    output_json = tmp_path / "outputs" / "metrics.json"
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )
    monkeypatch.setattr(
        module,
        "run_seed_diagnostic_pipeline",
        lambda ep, **kwargs: _pipeline_outputs(module, ep.shape),
    )

    module.run_example(
        data_root_arg=tmp_path / "f3_reference",
        output_json=output_json,
        save_figures=True,
        write_markdown_index=True,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
        scanner_backends=("current",),
        scanner_thin_modes=("normal",),
    )

    figures_dir = output_json.parent / "crop_001" / "current_normal" / "figures"
    markdown = (output_json.parent / "visual_report.md").read_text(encoding="utf-8")
    assert (figures_dir / "fet_seed_overlay_i3_3.png").is_file()
    assert (figures_dir / "reference_fv_high_seed_overlay_i3_3.png").is_file()
    assert (figures_dir / "reference_fvt_high_seed_overlay_i3_3.png").is_file()
    assert "current_normal" in markdown
    assert "crop_001/current_normal/figures/fet_seed_overlay_i3_3.png" in markdown


@pytest.mark.f3d_reference
def test_gated_real_data_seed_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _gated_data_root()
    module = _import_seed_module(monkeypatch)

    report = module.run_example(
        data_root_arg=data_root,
        count=1,
        crop_shape=(32, 32, 32),
        interior_margin=8,
        percentile=99.9,
        min_separation=16.0,
        scanner_backends=("current",),
        scanner_thin_modes=("normal",),
    )

    assert len(report["crops"]) == 1
    assert set(report["crops"][0]["cases"]) == {"current_normal"}
    assert report["aggregate"]["crop_count"] == 1
