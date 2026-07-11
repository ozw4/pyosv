from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pyosv.evaluation.synthetic_quality.config import (
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
)
from pyosv.evaluation.synthetic_quality.diagnostics import (
    _run_voter_thinning_diagnostic,
    _thinning_keep_mask_comparison,
)
from pyosv.synthetic3d import make_curved_surface_case, make_single_vertical_plane_case
from pyosv.voting3d import OptimalSurfaceVoter


DIAGNOSTIC_VOLUME_KEYS = {
    "fvt_reference_thinning_diagnostic",
    "fvt_normal_thinning_diagnostic",
    "keep_reference_thinning_diagnostic",
    "keep_normal_thinning_diagnostic",
    "keep_both_thinning_diagnostic",
    "keep_reference_only_thinning_diagnostic",
    "keep_normal_only_thinning_diagnostic",
}


def _load_report_module():
    script = Path(__file__).resolve().parents[3] / "examples" / "report_3d_synthetic_quality.py"
    spec = importlib.util.spec_from_file_location("report_3d_synthetic_quality", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "case_factory", (make_single_vertical_plane_case, make_curved_surface_case)
)
def test_thinning_diagnostic_is_finite_and_does_not_modify_primary_arrays(case_factory) -> None:
    case = case_factory((9, 9, 9))
    fv = np.array(case.ft_oracle, dtype=np.float32, copy=True)
    vp = np.array(case.pt_oracle, dtype=np.float32, copy=True)
    vt = np.array(case.tt_oracle, dtype=np.float32, copy=True)
    before = tuple(array.copy() for array in (fv, vp, vt))

    report, volumes = _run_voter_thinning_diagnostic(
        case=case,
        voter=OptimalSurfaceVoter(ru=2, rv=2, rw=2),
        fv=fv,
        vp=vp,
        vt=vt,
        reference_sigma=1.0,
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(),
    )

    for actual, expected in zip((fv, vp, vt), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert volumes.keys() == DIAGNOSTIC_VOLUME_KEYS
    assert all(np.all(np.isfinite(volume)) for volume in volumes.values())
    assert np.isfinite(report["reference"]["pyosv"]["fvt"]["mean"])
    assert np.isfinite(report["normal"]["pyosv"]["fvt"]["mean"])


def test_thinning_diagnostic_opt_out_does_not_invoke_extra_thinning(monkeypatch) -> None:
    module = _load_report_module()
    case = make_single_vertical_plane_case((9, 9, 9))
    calls = 0

    def fail_if_called(**kwargs):
        del kwargs
        nonlocal calls
        calls += 1
        raise AssertionError("diagnostic thinning must remain opt-in")

    monkeypatch.setattr(module, "_run_voter_thinning_diagnostic", fail_if_called)
    module._run_voting_from_attributes(
        case,
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
        voting_config=module.SyntheticVotingConfig(ru=2, rv=2, rw=2),
        truth_metric_config=module.SyntheticTruthMetricConfig(),
        skinning_config=module.SyntheticSkinningConfig(enabled=False),
        variant_spec=module.get_variant_spec("current_default"),
        include_thinning_diagnostic=False,
    )

    assert calls == 0


def test_scanner_diagnostics_are_opt_in_and_leave_primary_volumes_equal(monkeypatch) -> None:
    module = _load_report_module()
    case = make_single_vertical_plane_case((9, 9, 9))
    calls = {"downstream": 0, "stage_loss": 0}

    def downstream(**kwargs):
        del kwargs
        calls["downstream"] += 1
        return {"called": True}

    def stage_loss(**kwargs):
        del kwargs
        calls["stage_loss"] += 1
        return {"called": True}

    monkeypatch.setattr(module, "_scanner_downstream_diagnostics", downstream)
    monkeypatch.setattr(module, "_scanner_stage_loss_diagnostics", stage_loss)
    common = {
        "voting_config": module.SyntheticVotingConfig(ru=2, rv=2, rw=2),
        "scanner_config": module.SyntheticScannerConfig(backend="fast", scanner_thin_mode="none"),
        "truth_metric_config": module.SyntheticTruthMetricConfig(),
        "skinning_config": module.SyntheticSkinningConfig(enabled=False),
        "variant_spec": module.get_variant_spec("current_default"),
        "scanner_backend_matrix": False,
        "include_thinning_diagnostic": False,
    }

    plain_report, plain_volumes, _ = module._run_scanner_pipeline(
        case, include_scanner_downstream_diagnostics=False, **common
    )
    diagnostic_report, diagnostic_volumes, _ = module._run_scanner_pipeline(
        case, include_scanner_downstream_diagnostics=True, **common
    )

    assert calls == {"downstream": 1, "stage_loss": 1}
    assert "scanner_downstream" not in plain_report
    assert "scanner_stage_loss" not in plain_report
    assert diagnostic_report["scanner_downstream"] == {"called": True}
    assert diagnostic_report["scanner_stage_loss"] == {"called": True}
    assert plain_volumes.keys() == diagnostic_volumes.keys()
    for name in plain_volumes:
        np.testing.assert_array_equal(plain_volumes[name], diagnostic_volumes[name])


def test_keep_mask_comparison_rejects_mismatched_shapes_without_modifying_inputs() -> None:
    reference = np.zeros((2, 2, 2), dtype=bool)
    normal = np.zeros((2, 2, 3), dtype=bool)
    reference_before = reference.copy()
    normal_before = normal.copy()

    with pytest.raises(ValueError, match="keep mask shapes must match"):
        _thinning_keep_mask_comparison(
            reference,
            normal,
            truth_fault_mask=reference,
            buffer_radius=2.0,
        )

    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(normal, normal_before)
