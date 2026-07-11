from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from pyosv.evaluation.synthetic_quality.config import SyntheticVotingConfig
from pyosv.experimental.boundary_seed_selection import select_boundary_seed_retention_v1


def _coordinates(seeds):
    return [(seed.i1, seed.i2, seed.i3) for seed in seeds]


def test_boundary_seed_selection_is_deterministic_and_reports_target_source() -> None:
    config = SyntheticVotingConfig(seed_distance=1, seed_threshold=0.5)
    ft = np.zeros((5, 5, 5), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    target = np.zeros_like(ft)
    ft[2, 2, 2] = 0.9
    ft[0, 1, 1] = ft[0, 3, 3] = 0.4
    target[0, 1, 1] = target[0, 3, 3] = 1.0

    first = select_boundary_seed_retention_v1(
        voting_config=config,
        ft=ft,
        pt=pt,
        tt=tt,
        target=target,
        target_source="scanner_fet",
        edge_margin=1,
    )
    second = select_boundary_seed_retention_v1(
        voting_config=config,
        ft=ft,
        pt=pt,
        tt=tt,
        target=target,
        target_source="scanner_fet",
        edge_margin=1,
    )

    assert _coordinates(first.selected_seeds) == _coordinates(second.selected_seeds)
    assert _coordinates(first.selected_seeds) == [(2, 2, 2), (1, 1, 0), (3, 3, 0)]
    assert first.diagnostics["target_source"] == "scanner_fet"


def test_report_seed_wrapper_matches_experimental_result() -> None:
    path = Path(__file__).parents[2] / "examples" / "report_3d_synthetic_quality.py"
    spec = importlib.util.spec_from_file_location("seed_wrapper_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SyntheticVotingConfig(seed_distance=2, seed_threshold=0.5)
    ft = np.zeros((3, 3, 3), dtype=np.float32)
    ft[0, 1, 1] = 0.4
    kwargs = {
        "voting_config": config,
        "ft": ft,
        "pt": np.zeros_like(ft),
        "tt": np.full_like(ft, 90.0),
        "target": (ft > 0.0).astype(np.float32),
        "target_source": "scanner_fet",
        "edge_margin": 1,
    }

    result = select_boundary_seed_retention_v1(**kwargs)
    default, selected, diagnostics = module._boundary_seed_retention_v1_seeds(**kwargs)

    assert _coordinates(default) == _coordinates(result.default_seeds)
    assert _coordinates(selected) == _coordinates(result.selected_seeds)
    assert diagnostics == result.diagnostics
