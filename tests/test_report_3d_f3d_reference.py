from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pyosv.f3d_reference import F3D_ENV_VAR, F3D_FILENAMES


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _import_report_module(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop("report_3d_f3d_reference", None)
    importlib.invalidate_caches()
    return importlib.import_module("report_3d_f3d_reference")


def _synthetic_arrays() -> dict[str, np.ndarray]:
    base = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    return {file_name: base + index for index, file_name in enumerate(F3D_FILENAMES)}


def test_summarize_array_reports_expected_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_report_module(monkeypatch)
    values = np.arange(6, dtype=np.float32).reshape(2, 3)

    summary = module.summarize_array("small.dat", values)

    assert summary["file_name"] == "small.dat"
    assert summary["shape"] == [2, 3]
    assert summary["dtype"] == "float32"
    assert summary["finite_count"] == 6
    assert summary["min"] == 0.0
    assert summary["max"] == 5.0
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["std"] == pytest.approx(float(np.std(values.astype(np.float64))))
    assert summary["nonzero_fraction"] == pytest.approx(5.0 / 6.0)
    assert summary["percentiles"]["50"] == pytest.approx(2.5)
    assert summary["percentiles"]["99.9"] == pytest.approx(float(np.percentile(values, 99.9)))


def test_build_report_uses_synthetic_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_report_module(monkeypatch)
    arrays = _synthetic_arrays()
    arrays["fvt.dat"] = arrays["fv.dat"].copy()

    report = module.build_report(arrays, data_root="/tmp/f3")

    assert report["format_version"] == 1
    assert report["data_root"] == "/tmp/f3"
    assert [item["file_name"] for item in report["files"]] == list(F3D_FILENAMES)
    comparison = report["comparisons"]["fv_fvt"]
    assert comparison["normalized_correlation"] == pytest.approx(1.0)
    assert comparison["top_percentile_overlap"]["95"]["jaccard"] == pytest.approx(1.0)


def test_write_report_json_creates_parent_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    report = module.build_report(_synthetic_arrays())
    output_path = tmp_path / "outputs" / "3d" / "f3d" / "reference_baseline.json"

    written_path = module.write_report_json(report, output_path, pretty=True)

    assert written_path == output_path
    with output_path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    assert loaded["format_version"] == 1
    assert loaded["files"][0]["file_name"] == F3D_FILENAMES[0]


def test_run_example_uses_data_root_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    monkeypatch.setenv(F3D_ENV_VAR, str(data_root))

    seen_roots = []

    def fake_read_reference_arrays(root: Path) -> dict[str, np.ndarray]:
        seen_roots.append(root)
        arrays = _synthetic_arrays()
        arrays["fvt.dat"] = arrays["fv.dat"].copy()
        return arrays

    monkeypatch.setattr(module, "read_reference_arrays", fake_read_reference_arrays)

    report = module.run_example(data_root_arg=None)

    assert seen_roots == [data_root]
    assert report["data_root"] == str(data_root)


def test_run_example_rejects_output_inside_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    data_root = tmp_path / "f3_reference"

    with pytest.raises(ValueError, match="inside the F3 data root"):
        module.run_example(
            data_root_arg=data_root,
            output_json=data_root / "reports" / "reference_baseline.json",
        )
