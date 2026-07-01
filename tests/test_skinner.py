import numpy as np
import pytest

from pyosv.cells import FaultCell
from pyosv.skin import FaultSkin
from pyosv.skinner import (
    ConnectedComponentSkinner,
    FaultSkinner,
    _find_reference_seeds,
    find_skins,
)
from pyosv.voting3d import OptimalSurfaceVoter


def test_constructor_stores_configuration_for_later_grouping() -> None:
    skinner = FaultSkinner(min_likelihood=np.float32(0.35), min_skin_size=np.int32(4))

    assert skinner.min_likelihood == pytest.approx(0.35)
    assert skinner.min_skin_size == 4
    assert skinner.connectivity == "corner"


def test_connected_component_skinner_stores_fallback_configuration() -> None:
    skinner = ConnectedComponentSkinner(
        min_likelihood=np.float32(0.35),
        min_skin_size=np.int32(4),
        connectivity="face",
    )

    assert skinner.min_likelihood == pytest.approx(0.35)
    assert skinner.min_skin_size == 4
    assert skinner.connectivity == "face"


def test_fault_skinner_configuration_remains_mutable() -> None:
    skinner = FaultSkinner()

    skinner.min_likelihood = np.float32(0.35)
    skinner.min_skin_size = np.int32(4)
    skinner.connectivity = "face"

    assert skinner.min_likelihood == pytest.approx(0.35)
    assert skinner.min_skin_size == 4
    assert skinner.connectivity == "face"


@pytest.mark.parametrize("min_likelihood", [-0.1, np.nan, np.inf, True, "0.5"])
def test_constructor_rejects_invalid_min_likelihood(min_likelihood: object) -> None:
    with pytest.raises(ValueError, match="min_likelihood"):
        FaultSkinner(min_likelihood=min_likelihood)  # type: ignore[arg-type]


@pytest.mark.parametrize("min_skin_size", [-1, 1.5, True, "1"])
def test_constructor_rejects_invalid_min_skin_size(min_skin_size: object) -> None:
    with pytest.raises(ValueError, match="min_skin_size"):
        FaultSkinner(min_skin_size=min_skin_size)  # type: ignore[arg-type]


def test_constructor_rejects_unknown_connectivity() -> None:
    with pytest.raises(ValueError, match="connectivity"):
        FaultSkinner(connectivity="diagonal")


def test_cells_from_votes_extracts_thresholded_fault_cells_in_volume_order() -> None:
    fv = np.zeros((2, 2, 3), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)
    fv[1, 0, 2] = 0.75
    fv[0, 1, 0] = 0.5
    fv[1, 1, 1] = 0.9
    vp[1, 0, 2] = 30.0
    vt[1, 0, 2] = 60.0
    vp[0, 1, 0] = 20.0
    vt[0, 1, 0] = 50.0
    vp[1, 1, 1] = 40.0
    vt[1, 1, 1] = 70.0

    cells = FaultSkinner().cells_from_votes(fv, vp, vt, min_likelihood=0.5)

    assert [cell.index for cell in cells] == [(0, 1, 0), (2, 0, 1), (1, 1, 1)]
    assert [cell.fl for cell in cells] == pytest.approx([0.5, 0.75, 0.9])
    assert [cell.fp for cell in cells] == pytest.approx([20.0, 30.0, 40.0])
    assert [cell.ft for cell in cells] == pytest.approx([50.0, 60.0, 70.0])
    assert all(isinstance(cell, FaultCell) for cell in cells)


def test_cells_from_votes_uses_configured_threshold_by_default() -> None:
    fv = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    vp = np.full_like(fv, 10.0)
    vt = np.full_like(fv, 20.0)

    cells = FaultSkinner(min_likelihood=0.5).cells_from_votes(fv, vp, vt)

    assert [cell.index for cell in cells] == [(1, 0, 0), (2, 0, 0)]


def test_cells_from_votes_accepts_call_threshold_override() -> None:
    fv = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    vp = np.full_like(fv, 10.0)
    vt = np.full_like(fv, 20.0)

    cells = FaultSkinner(min_likelihood=0.7).cells_from_votes(fv, vp, vt, min_likelihood=0.5)

    assert [cell.index for cell in cells] == [(1, 0, 0), (2, 0, 0)]


def test_cells_from_votes_returns_empty_list_without_candidates() -> None:
    fv = np.zeros((1, 2, 3), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)

    assert FaultSkinner().cells_from_votes(fv, vp, vt, min_likelihood=0.1) == []


def test_cells_from_votes_excludes_zero_background_by_default() -> None:
    fv = np.zeros((2, 3, 4), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)

    assert FaultSkinner().cells_from_votes(fv, vp, vt) == []


@pytest.mark.parametrize(
    ("fv", "vp", "vt", "match"),
    [
        (
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((1, 2, 2), dtype=np.float32),
            np.zeros((1, 2, 3), dtype=np.float32),
            "shapes must match",
        ),
        (
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            "3D array",
        ),
    ],
)
def test_cells_from_votes_rejects_invalid_shapes(
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        FaultSkinner().cells_from_votes(fv, vp, vt)


@pytest.mark.parametrize("name", ["fv", "vp", "vt"])
def test_cells_from_votes_rejects_non_finite_inputs(name: str) -> None:
    arrays = {
        "fv": np.zeros((1, 2, 3), dtype=np.float32),
        "vp": np.zeros((1, 2, 3), dtype=np.float32),
        "vt": np.zeros((1, 2, 3), dtype=np.float32),
    }
    arrays[name][0, 0, 0] = np.nan

    with pytest.raises(ValueError, match=f"{name} must contain only finite values"):
        FaultSkinner().cells_from_votes(arrays["fv"], arrays["vp"], arrays["vt"])


@pytest.mark.parametrize("min_likelihood", [-0.1, np.nan, np.inf, True, "0.5"])
def test_cells_from_votes_rejects_invalid_threshold_override(min_likelihood: object) -> None:
    fv = np.zeros((1, 1, 1), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)

    with pytest.raises(ValueError, match="min_likelihood"):
        FaultSkinner().cells_from_votes(
            fv,
            vp,
            vt,
            min_likelihood=min_likelihood,  # type: ignore[arg-type]
        )


def test_cells_from_votes_includes_samples_equal_to_positive_threshold() -> None:
    fv = np.array([[[0.0, 0.49, 0.5, 0.75]]], dtype=np.float32)
    vp = np.full_like(fv, 10.0)
    vt = np.full_like(fv, 20.0)

    cells = FaultSkinner(min_likelihood=0.5).cells_from_votes(fv, vp, vt)

    assert [cell.index for cell in cells] == [(2, 0, 0), (3, 0, 0)]
    assert [cell.fl for cell in cells] == pytest.approx([0.5, 0.75])


def test_find_reference_seeds_uses_planarity_and_thinned_likelihood_thresholds() -> None:
    ep = np.array([[[0.81, 0.80, 0.90, 0.95]]], dtype=np.float32)
    ft = np.array([[[0.60, 0.70, 0.50, 0.51]]], dtype=np.float32)
    pt = np.array([[[10.0, 20.0, 30.0, 40.0]]], dtype=np.float32)
    tt = np.array([[[50.0, 60.0, 70.0, 80.0]]], dtype=np.float32)

    seeds = _find_reference_seeds(d=0, fm=0.5, ep=ep, ft=ft, pt=pt, tt=tt)

    assert [seed.index for seed in seeds] == [(0, 0, 0), (3, 0, 0)]
    assert [seed.fl for seed in seeds] == pytest.approx([0.60, 0.51])
    assert [seed.fp for seed in seeds] == pytest.approx([10.0, 40.0])
    assert [seed.ft for seed in seeds] == pytest.approx([50.0, 80.0])


def test_find_reference_seeds_orders_candidates_by_likelihood_with_deterministic_ties() -> None:
    ep = np.ones((2, 2, 3), dtype=np.float32)
    ft = np.zeros_like(ep)
    pt = np.zeros_like(ep)
    tt = np.zeros_like(ep)
    ft[1, 1, 2] = 0.8
    ft[0, 1, 0] = 0.7
    ft[0, 0, 1] = 0.7

    seeds = _find_reference_seeds(d=0, fm=0.5, ep=ep, ft=ft, pt=pt, tt=tt)

    assert [seed.index for seed in seeds] == [(2, 1, 1), (1, 0, 0), (0, 1, 0)]
    assert [seed.fl for seed in seeds] == pytest.approx([0.8, 0.7, 0.7])


def test_find_reference_seeds_excludes_candidates_within_marked_box() -> None:
    ep = np.ones((5, 5, 5), dtype=np.float32)
    ft = np.zeros_like(ep)
    pt = np.zeros_like(ep)
    tt = np.zeros_like(ep)
    ft[2, 2, 2] = 0.9
    ft[2, 3, 2] = 0.8
    ft[4, 2, 2] = 0.7

    seeds = _find_reference_seeds(d=1, fm=0.5, ep=ep, ft=ft, pt=pt, tt=tt)

    assert [seed.index for seed in seeds] == [(2, 2, 2), (2, 2, 4)]


def test_fault_skinner_find_seeds_returns_public_fault_cells() -> None:
    ep = np.ones((1, 1, 1), dtype=np.float32)
    ft = np.array([[[0.9]]], dtype=np.float32)
    pt = np.array([[[20.0]]], dtype=np.float32)
    tt = np.array([[[45.0]]], dtype=np.float32)

    seeds = FaultSkinner().find_seeds(d=0, fm=0.5, ep=ep, ft=ft, pt=pt, tt=tt)

    assert len(seeds) == 1
    assert isinstance(seeds[0], FaultCell)
    assert seeds[0].index == (0, 0, 0)
    assert seeds[0].fl == pytest.approx(0.9)
    assert seeds[0].fp == pytest.approx(20.0)
    assert seeds[0].ft == pytest.approx(45.0)


def test_find_reference_seeds_rejects_mismatched_shapes() -> None:
    ep = np.zeros((2, 3, 4), dtype=np.float32)
    ft = np.zeros((2, 4, 3), dtype=np.float32)
    pt = np.zeros_like(ep)
    tt = np.zeros_like(ep)

    with pytest.raises(ValueError, match="shapes must match"):
        _find_reference_seeds(d=1, fm=0.5, ep=ep, ft=ft, pt=pt, tt=tt)


@pytest.mark.parametrize("name", ["ep", "ft", "pt", "tt"])
def test_find_reference_seeds_rejects_non_finite_inputs(name: str) -> None:
    arrays = {
        "ep": np.zeros((1, 2, 3), dtype=np.float32),
        "ft": np.zeros((1, 2, 3), dtype=np.float32),
        "pt": np.zeros((1, 2, 3), dtype=np.float32),
        "tt": np.zeros((1, 2, 3), dtype=np.float32),
    }
    arrays[name][0, 0, 0] = np.nan

    with pytest.raises(ValueError, match=f"{name} must contain only finite values"):
        _find_reference_seeds(
            d=1,
            fm=0.5,
            ep=arrays["ep"],
            ft=arrays["ft"],
            pt=arrays["pt"],
            tt=arrays["tt"],
        )


@pytest.mark.parametrize(
    ("d", "fm", "match"),
    [(-1, 0.5, "d"), (1, np.nan, "fm"), (1, True, "fm")],
)
def test_find_reference_seeds_rejects_invalid_parameters(
    d: object,
    fm: object,
    match: str,
) -> None:
    ep = np.zeros((1, 1, 1), dtype=np.float32)
    ft = np.zeros_like(ep)
    pt = np.zeros_like(ep)
    tt = np.zeros_like(ep)

    with pytest.raises(ValueError, match=match):
        _find_reference_seeds(
            d=d,  # type: ignore[arg-type]
            fm=fm,  # type: ignore[arg-type]
            ep=ep,
            ft=ft,
            pt=pt,
            tt=tt,
        )


def test_find_skins_default_groups_only_sparse_positive_samples() -> None:
    fv = np.zeros((2, 3, 4), dtype=np.float32)
    vp = np.full_like(fv, 25.0)
    vt = np.full_like(fv, 65.0)
    fv[0, 0, 0] = 0.2
    fv[0, 0, 1] = 0.3
    fv[1, 2, 3] = 0.4

    skins = find_skins(fv, vp, vt)

    assert [len(skin) for skin in skins] == [2, 1]
    assert [[cell.index for cell in skin] for skin in skins] == [
        [(0, 0, 0), (1, 0, 0)],
        [(3, 2, 1)],
    ]


def test_find_skins_returns_separated_planar_patches_as_two_skins() -> None:
    fv = np.zeros((3, 5, 6), dtype=np.float32)
    vp = np.full_like(fv, 30.0)
    vt = np.full_like(fv, 60.0)
    fv[0, 0:2, 0:2] = 0.8
    fv[2, 3:5, 4:6] = 0.9

    skins = find_skins(fv, vp, vt, min_likelihood=0.5)

    assert [len(skin) for skin in skins] == [4, 4]
    assert [skin.cells[0].index for skin in skins] == [(0, 0, 0), (4, 3, 2)]
    assert all(isinstance(skin, FaultSkin) for skin in skins)
    assert all(np.isfinite([cell.fl, cell.fp, cell.ft]).all() for skin in skins for cell in skin)


def test_find_skins_groups_adjacent_voxels_with_configured_connectivity() -> None:
    fv = np.zeros((2, 2, 2), dtype=np.float32)
    vp = np.full_like(fv, 15.0)
    vt = np.full_like(fv, 75.0)
    fv[0, 0, 0] = 0.7
    fv[1, 1, 1] = 0.8

    face_skins = FaultSkinner(connectivity="face").find_skins(fv, vp, vt, min_likelihood=0.5)
    corner_skins = FaultSkinner(connectivity="corner").find_skins(fv, vp, vt, min_likelihood=0.5)

    assert [len(skin) for skin in face_skins] == [1, 1]
    assert [skin.cells[0].index for skin in face_skins] == [(0, 0, 0), (1, 1, 1)]
    assert [len(skin) for skin in corner_skins] == [2]
    assert [cell.index for cell in corner_skins[0]] == [(0, 0, 0), (1, 1, 1)]


def test_find_skins_uses_edge_connectivity_for_edge_adjacent_voxels() -> None:
    fv = np.zeros((1, 2, 2), dtype=np.float32)
    vp = np.full_like(fv, 20.0)
    vt = np.full_like(fv, 50.0)
    fv[0, 0, 0] = 0.7
    fv[0, 1, 1] = 0.8

    face_skins = FaultSkinner(connectivity="face").find_skins(fv, vp, vt, min_likelihood=0.5)
    edge_skins = FaultSkinner(connectivity="edge").find_skins(fv, vp, vt, min_likelihood=0.5)

    assert [len(skin) for skin in face_skins] == [1, 1]
    assert [len(skin) for skin in edge_skins] == [2]


def test_fault_skinner_matches_connected_component_fallback() -> None:
    fv = np.zeros((1, 3, 4), dtype=np.float32)
    vp = np.full_like(fv, 20.0)
    vt = np.full_like(fv, 50.0)
    fv[0, 0, 0] = 0.7
    fv[0, 1, 1] = 0.8
    fv[0, 2, 3] = 0.9

    fallback = ConnectedComponentSkinner(connectivity="edge").find_skins(
        fv,
        vp,
        vt,
        min_likelihood=0.5,
    )
    default = FaultSkinner(connectivity="edge").find_skins(fv, vp, vt, min_likelihood=0.5)

    assert [[cell.index for cell in skin] for skin in default] == [
        [cell.index for cell in skin] for skin in fallback
    ]


def test_find_skins_filters_small_components_and_orders_remaining_skins() -> None:
    fv = np.zeros((1, 4, 7), dtype=np.float32)
    vp = np.full_like(fv, 35.0)
    vt = np.full_like(fv, 65.0)
    fv[0, 0, 6] = 0.9
    fv[0, 1, 0:2] = 0.8
    fv[0, 3, 4:7] = 0.7

    skins = FaultSkinner(min_skin_size=2, connectivity="face").find_skins(
        fv,
        vp,
        vt,
        min_likelihood=0.5,
    )

    assert [len(skin) for skin in skins] == [3, 2]
    assert [skin.cells[0].index for skin in skins] == [(4, 3, 0), (0, 1, 0)]


def test_find_skins_groups_thinned_apply_voting_plane_as_one_dominant_skin() -> None:
    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    ft[3:8, 5, 3:8] = 0.9

    fv, vp, vt = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)
    fvt = voter.thin(fv, vp, vt)
    skins = FaultSkinner(
        min_likelihood=0.7,
        min_skin_size=20,
        connectivity="corner",
    ).find_skins(fvt, vp, vt)

    assert len(skins) == 1
    skin = skins[0]
    indices = skin.indices()
    assert len(skin) == 105
    assert indices.dtype == np.int32
    assert np.count_nonzero(indices[:, 1] == 5) > 0.75 * len(skin)
    assert skin.likelihoods().min() >= 0.7
