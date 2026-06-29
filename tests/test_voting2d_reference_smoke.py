from pathlib import Path

import numpy as np
import pytest

from pyosv.io import read_dat
from pyosv.reference import REFERENCE_DATASETS_2D, reference_root, resolve_reference_file
from pyosv.voting2d import OptimalPathVoter


def require_reference_root() -> Path:
    root = reference_root()
    if not root.exists():
        pytest.skip(f"reference_osv mount not available: {root}")
    return root


def test_f3d2d_seed_picking_smoke() -> None:
    root = require_reference_root()
    dataset = REFERENCE_DATASETS_2D["f3d2d"]
    ft_path = resolve_reference_file(dataset, "ft.dat", root=root)
    pt_path = resolve_reference_file(dataset, "pt.dat", root=root)
    if not ft_path.exists():
        pytest.skip(f"reference fixture not available: {ft_path}")
    if not pt_path.exists():
        pytest.skip(f"reference fixture not available: {pt_path}")

    ft = read_dat(ft_path, dataset.shape, endian=dataset.endian)
    pt = read_dat(pt_path, dataset.shape, endian=dataset.endian)
    fm = 0.3

    seeds = OptimalPathVoter(ru=15, rv=30).pick_seeds(d=4, fm=fm, ft=ft, pt=pt)

    assert len(seeds) > 0

    n2, n1 = dataset.shape
    i1 = np.array([seed.i1 for seed in seeds])
    i2 = np.array([seed.i2 for seed in seeds])
    fl = np.array([seed.fl for seed in seeds], dtype=np.float32)
    seed_values = np.array(
        [(seed.i1, seed.i2, seed.fl, seed.fp) for seed in seeds],
        dtype=np.float32,
    )

    assert ((0 <= i1) & (i1 < n1)).all()
    assert ((0 <= i2) & (i2 < n2)).all()
    assert (fl > fm).all()
    assert np.isfinite(seed_values).all()
