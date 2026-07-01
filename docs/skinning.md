# Skinning

`pyosv.skin` and `pyosv.skinner` provide a minimal Python skinning layer for
3D voting outputs. `ConnectedComponentSkinner` is the explicit fallback
connected-component implementation. `FaultSkinner.find_skins` delegates to
that fallback so the default grouping behavior is unchanged, while
`FaultSkinner.find_seeds` and `FaultSkinner.find_skin` expose opt-in
reference-like seed selection and single-seed growth helpers.

## Scope

`FaultSkin` is a small container for grouped `FaultCell` objects. It preserves
cell order, supports iteration and `len()`, and exposes helper arrays:

- `indices()` returns an `(n, 3)` `int32` array in `(i1, i2, i3)` order.
- `likelihoods()` returns an `(n,)` `float32` array of fault likelihood values.

`ConnectedComponentSkinner.cells_from_votes` extracts one `FaultCell` for
positive `fv` samples where `fv >= min_likelihood`. Zero-valued background
samples are excluded even when `min_likelihood=0.0`. `min_likelihood` must be
finite and nonnegative. Input arrays are global 3D volumes with shape
`(n3, n2, n1)`, and `vp` and `vt` must match `fv`.

`ConnectedComponentSkinner.find_skins` groups extracted cells by voxel
connected components. Connectivity is configured as:

- `face`: 6-connected adjacency.
- `edge`: 18-connected adjacency.
- `corner`: 26-connected adjacency.

Small components can be filtered with `min_skin_size`. Returned skins are
ordered by descending size, then by the first cell index in each component.
`FaultSkinner.find_skins` currently preserves the same public behavior by
delegating to `ConnectedComponentSkinner`.

`FaultSkinner.find_seeds(d, fm, ep, ft, pt, tt)` selects starting cells from
thinned 3D volumes. All arrays use shape `(n3, n2, n1)` and must have matching
finite values. Candidates must satisfy `ep > 0.8` and `ft > fm`; they are
processed by descending `ft`, with deterministic index ordering for ties. A
candidate is skipped when an already accepted seed falls inside its `d`-sample
axis-aligned exclusion box. Returned seeds are public `FaultCell` objects.

`FaultSkinner.find_skin(seed, fv, vp, vt, ...)` grows one opt-in,
reference-like `FaultSkin` from a seed without changing `find_skins`. The
grower builds a seed-local `(u, v, w)` coordinate frame where `u` follows the
fault normal, `v` follows the dip vector, and `w` follows the strike vector.
It samples candidate slices with Java-style nearest rounding, uses a priority
queue ordered by likelihood, explores above/below and left/right local
directions, and applies deterministic geometry gates such as minimum
likelihood, local `u` continuity, interior bounds, and accepted-cell collision
avoidance.

## Minimal Usage

```python
from pyosv.skinner import ConnectedComponentSkinner
from pyosv.voting3d import OptimalSurfaceVoter

voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
fv, vp, vt = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)
fvt = voter.thin(fv, vp, vt)

skinner = ConnectedComponentSkinner(min_likelihood=0.7, min_skin_size=20)
skins = skinner.find_skins(fvt, vp, vt)
```

The self-contained example can be run without external data:

```bash
python examples/run_3d_synthetic_skinning.py
```

Pass `--output-dir` only when generated DAT outputs and a small text skin
summary should be written.

## Limitations

This implementation does not reproduce the full Java skinning workflow,
including reskin smoothing, throw/slip estimation, or real-data workflow
helpers. `find_skin` is a focused single-seed, reference-like grower with
simplified candidate picking and geometry gates; `find_skins` remains the
connected-component fallback. `ConnectedComponentSkinner` remains available for
explicit fallback use.
