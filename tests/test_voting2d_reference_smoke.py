import numpy as np

from pyosv.reference import REFERENCE_DATASETS_2D
from pyosv.voting2d import OptimalPathVoter

from reference_fixtures import load_reference_2d_array


def test_f3d2d_seed_picking_smoke() -> None:
    dataset = REFERENCE_DATASETS_2D["f3d2d"]
    ft = load_reference_2d_array("f3d2d", "ft.dat")
    pt = load_reference_2d_array("f3d2d", "pt.dat")
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
