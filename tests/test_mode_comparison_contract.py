from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyosv.evaluation.synthetic_quality import (
    SyntheticScannerConfig,
    resolve_workflow_settings,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
SYNTHETIC_CONDITIONS = (
    ("RL-REF", "reference-like", "reference"),
    ("RL-QUAL", "reference-like", "quality"),
    ("Q-REF", "quality", "reference"),
    ("Q-QUAL", "quality", "quality"),
)


def _import_example(monkeypatch: pytest.MonkeyPatch, module_name: str) -> object:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    return importlib.import_module(module_name)


def _synthetic_effective_config(
    *,
    scanner_backend: str,
    workflow_mode: str,
) -> dict[str, Any]:
    scanner_config = SyntheticScannerConfig(backend=scanner_backend)
    workflow = resolve_workflow_settings(workflow_mode=workflow_mode)
    scanner_report = scanner_config.as_report_dict()
    voting_report = workflow.voting_config.as_report_dict()
    skinning_report = workflow.skinning_config.as_report_dict()
    return {
        "scanner_backend": scanner_report["backend"],
        "workflow_mode": workflow.workflow_mode,
        "scanner": scanner_report,
        "voting": voting_report,
        "skinning": skinning_report,
    }


def _synthetic_contract_fields(config: dict[str, Any]) -> dict[str, Any]:
    scanner = config["scanner"]
    voting = config["voting"]
    skinning = config["skinning"]
    return {
        "scanner_backend": config["scanner_backend"],
        "workflow_mode": config["workflow_mode"],
        "scanner_refinement_factor": scanner["refinement_factor"],
        "scanner_thin_mode": scanner["scanner_thin_mode"],
        "scanner_edge_cleanup": scanner["remove_edge_effects"],
        "voter_thin_mode": voting["voter_thin_mode"],
        "surface_support_min_fraction": voting["surface_support_min_fraction"],
        "surface_support_exponent": voting["surface_support_exponent"],
        "skinner_method": skinning["method"],
        "min_likelihood": skinning["min_likelihood"],
        "adaptive_min_likelihood": skinning["adaptive_min_likelihood"],
        "seed_minimum_planarity": skinning["seed_min_ep"],
        "growth_source": skinning["growth_source"],
        "accepted_occupancy_radius": skinning["accepted_occupancy_radius"],
        "effective_accepted_occupancy_radius": skinning["effective_accepted_occupancy_radius"],
        "boundary_skinner_fallback": skinning["boundary_skinner_fallback"],
        "boundary_skinner_fallback_policy": skinning["boundary_skinner_fallback_policy"],
    }


@pytest.mark.parametrize(
    ("condition_id", "scanner_backend", "workflow_mode"),
    SYNTHETIC_CONDITIONS,
)
def test_synthetic_public_condition_effective_configuration(
    condition_id: str,
    scanner_backend: str,
    workflow_mode: str,
) -> None:
    config = _synthetic_effective_config(
        scanner_backend=scanner_backend,
        workflow_mode=workflow_mode,
    )

    expected = {
        "RL-REF": {
            "scanner_backend": "reference-like",
            "workflow_mode": "reference",
            "scanner_refinement_factor": 2,
            "scanner_thin_mode": "reference",
            "scanner_edge_cleanup": True,
            "voter_thin_mode": "reference",
            "surface_support_min_fraction": 0.0,
            "surface_support_exponent": 0.0,
            "skinner_method": "reference",
            "min_likelihood": 0.5,
            "adaptive_min_likelihood": False,
            "seed_minimum_planarity": 0.8,
            "growth_source": "thinned",
            "accepted_occupancy_radius": None,
            "effective_accepted_occupancy_radius": 5,
            "boundary_skinner_fallback": False,
            "boundary_skinner_fallback_policy": "empty_primary",
        },
        "RL-QUAL": {
            "scanner_backend": "reference-like",
            "workflow_mode": "quality",
            "scanner_refinement_factor": 2,
            "scanner_thin_mode": "reference",
            "scanner_edge_cleanup": True,
            "voter_thin_mode": "hybrid_v2",
            "surface_support_min_fraction": 0.0,
            "surface_support_exponent": 0.0,
            "skinner_method": "quality",
            "min_likelihood": None,
            "adaptive_min_likelihood": True,
            "seed_minimum_planarity": 0.5,
            "growth_source": "pre_thin",
            "accepted_occupancy_radius": 1,
            "effective_accepted_occupancy_radius": 1,
            "boundary_skinner_fallback": True,
            "boundary_skinner_fallback_policy": "empty_primary",
        },
        "Q-REF": {
            "scanner_backend": "quality",
            "workflow_mode": "reference",
            "scanner_refinement_factor": 2,
            "scanner_thin_mode": "reference",
            "scanner_edge_cleanup": True,
            "voter_thin_mode": "reference",
            "surface_support_min_fraction": 0.0,
            "surface_support_exponent": 0.0,
            "skinner_method": "reference",
            "min_likelihood": 0.5,
            "adaptive_min_likelihood": False,
            "seed_minimum_planarity": 0.8,
            "growth_source": "thinned",
            "accepted_occupancy_radius": None,
            "effective_accepted_occupancy_radius": 5,
            "boundary_skinner_fallback": False,
            "boundary_skinner_fallback_policy": "empty_primary",
        },
        "Q-QUAL": {
            "scanner_backend": "quality",
            "workflow_mode": "quality",
            "scanner_refinement_factor": 2,
            "scanner_thin_mode": "reference",
            "scanner_edge_cleanup": True,
            "voter_thin_mode": "hybrid_v2",
            "surface_support_min_fraction": 0.0,
            "surface_support_exponent": 0.0,
            "skinner_method": "quality",
            "min_likelihood": None,
            "adaptive_min_likelihood": True,
            "seed_minimum_planarity": 0.5,
            "growth_source": "pre_thin",
            "accepted_occupancy_radius": 1,
            "effective_accepted_occupancy_radius": 1,
            "boundary_skinner_fallback": True,
            "boundary_skinner_fallback_policy": "empty_primary",
        },
    }

    assert _synthetic_contract_fields(config) == expected[condition_id]


def test_synthetic_condition_axes_are_independent() -> None:
    conditions = {
        condition_id: _synthetic_effective_config(
            scanner_backend=scanner_backend,
            workflow_mode=workflow_mode,
        )
        for condition_id, scanner_backend, workflow_mode in SYNTHETIC_CONDITIONS
    }
    workflow_fields = (
        "voter_thin_mode",
        "surface_support_min_fraction",
        "surface_support_exponent",
        "skinner_method",
        "min_likelihood",
        "adaptive_min_likelihood",
        "seed_minimum_planarity",
        "growth_source",
        "accepted_occupancy_radius",
        "effective_accepted_occupancy_radius",
        "boundary_skinner_fallback",
        "boundary_skinner_fallback_policy",
    )
    scanner_fields = (
        "scanner_refinement_factor",
        "scanner_thin_mode",
        "scanner_edge_cleanup",
    )

    def selected(fields: tuple[str, ...], condition_id: str) -> tuple[Any, ...]:
        return tuple(
            _synthetic_contract_fields(conditions[condition_id])[field] for field in fields
        )

    assert selected(workflow_fields, "RL-REF") == selected(workflow_fields, "Q-REF")
    assert selected(workflow_fields, "RL-QUAL") == selected(workflow_fields, "Q-QUAL")
    assert selected(scanner_fields, "RL-REF") == selected(scanner_fields, "RL-QUAL")
    assert selected(scanner_fields, "Q-REF") == selected(scanner_fields, "Q-QUAL")
    assert selected(workflow_fields, "RL-REF") != selected(workflow_fields, "RL-QUAL")
    assert selected(workflow_fields, "Q-REF") != selected(workflow_fields, "Q-QUAL")
    assert selected(scanner_fields, "RL-REF") == selected(scanner_fields, "Q-REF")
    assert selected(scanner_fields, "RL-QUAL") == selected(scanner_fields, "Q-QUAL")

    assert SyntheticScannerConfig(backend="quality").as_report_dict()["refinement_factor"] == 2
    assert (
        SyntheticScannerConfig(
            backend="reference-like", scanner_thin_mode="normal"
        ).as_report_dict()["scanner_thin_mode"]
        == "normal"
    )
    assert (
        SyntheticScannerConfig(backend="quality", scanner_thin_mode="normal").as_report_dict()[
            "scanner_thin_mode"
        ]
        == "normal"
    )


def test_f3_workflow_resolution_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_example(monkeypatch, "run_3d_f3d_crop_validation")

    reference = module.resolve_workflow_options(
        workflow_mode="reference",
        voter_thin_mode=None,
        surface_support_min_fraction=None,
        surface_support_exponent=None,
    )
    quality = module.resolve_workflow_options(
        workflow_mode="quality",
        voter_thin_mode=None,
        surface_support_min_fraction=None,
        surface_support_exponent=None,
    )
    assert reference == {
        "workflow_mode": "reference",
        "voter_thin_mode": "reference",
        "surface_support_min_fraction": 0.0,
        "surface_support_exponent": 0.0,
    }
    assert quality == {
        "workflow_mode": "quality",
        "voter_thin_mode": "hybrid_v2",
        "surface_support_min_fraction": 0.0,
        "surface_support_exponent": 0.0,
    }

    overridden = module.resolve_workflow_options(
        workflow_mode="quality",
        voter_thin_mode="reference",
        surface_support_min_fraction=0.25,
        surface_support_exponent=2.0,
    )
    assert overridden == {
        "workflow_mode": "quality",
        "voter_thin_mode": "reference",
        "surface_support_min_fraction": 0.25,
        "surface_support_exponent": 2.0,
    }


def _synthetic_f3_reference_arrays(
    shape: tuple[int, int, int] = (8, 8, 8),
) -> dict[str, np.ndarray]:
    arrays = {
        name: np.zeros(shape, dtype=np.float32)
        for name in ("ep.dat", "fl.dat", "fv.dat", "fvt.dat")
    }
    arrays["fv.dat"][3, 3, 3] = 1.0
    arrays["fvt.dat"][3, 3, 3] = 1.0
    arrays["fl.dat"][3, 3, 3] = 1.0
    return arrays


def _synthetic_f3_outputs(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    base = np.zeros(shape, dtype=np.float32)
    base[tuple(size // 2 for size in shape)] = 1.0
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


def test_f3_compare_workflow_branches_keep_scanner_configuration_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_example(monkeypatch, "report_3d_f3d_multicrop")
    received_kwargs: list[dict[str, object]] = []
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_f3_reference_arrays(),
    )

    def fake_run_pipeline(ep: np.ndarray, **kwargs: object) -> dict[str, np.ndarray]:
        received_kwargs.append(dict(kwargs))
        return _synthetic_f3_outputs(ep.shape)

    monkeypatch.setattr(module.crop_validation, "run_pipeline", fake_run_pipeline)

    report = module.run_example(
        data_root_arg=tmp_path / "f3_reference",
        compare_workflows=True,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(3, 3, 3)],
    )

    assert set(report["workflows"]) == {"reference", "quality"}
    assert len(received_kwargs) == 2
    scanner_keys = (
        "sigma1",
        "sigma2",
        "phi_min",
        "phi_max",
        "theta_min",
        "theta_max",
        "scanner_thin_mode",
        "reference_thin_sigma",
        "remove_scanner_edge_effects",
    )
    for key in scanner_keys:
        assert received_kwargs[0][key] == received_kwargs[1][key]

    assert received_kwargs[0]["scanner_thin_mode"] == "reference"
    assert received_kwargs[0]["reference_thin_sigma"] == 1.0
    assert received_kwargs[0]["remove_scanner_edge_effects"] is True
    assert received_kwargs[0]["voter_thin_mode"] == "reference"
    assert received_kwargs[1]["voter_thin_mode"] == "hybrid_v2"
    assert received_kwargs[0]["surface_support_min_fraction"] == 0.0
    assert received_kwargs[1]["surface_support_min_fraction"] == 0.0
    assert received_kwargs[0]["surface_support_exponent"] == 0.0
    assert received_kwargs[1]["surface_support_exponent"] == 0.0

    differing_keys = {
        key for key in received_kwargs[0] if received_kwargs[0][key] != received_kwargs[1][key]
    }
    assert differing_keys == {"voter_thin_mode"}
