# Minimal Skinning

`pyosv.skin` and `pyosv.skinner` provide a minimal Python skinning layer for
3D voting outputs. This is connected-component skinning, not the full Java
`FaultSkinner` algorithm.

## Scope

`FaultSkin` is a small container for grouped `FaultCell` objects. It preserves
cell order, supports iteration and `len()`, and exposes helper arrays:

- `indices()` returns an `(n, 3)` `int32` array in `(i1, i2, i3)` order.
- `likelihoods()` returns an `(n,)` `float32` array of fault likelihood values.

`FaultSkinner.cells_from_votes` extracts one `FaultCell` at every sample where
`fv >= min_likelihood`. Input arrays are global 3D volumes with shape
`(n3, n2, n1)`, and `vp` and `vt` must match `fv`.

`FaultSkinner.find_skins` groups extracted cells by voxel connected components.
Connectivity is configured as:

- `face`: 6-connected adjacency.
- `edge`: 18-connected adjacency.
- `corner`: 26-connected adjacency.

Small components can be filtered with `min_skin_size`. Returned skins are
ordered by descending size, then by the first cell index in each component.

## Minimal Usage

```python
from pyosv.skinner import FaultSkinner
from pyosv.voting3d import OptimalSurfaceVoter

voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
fv, vp, vt = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)
fvt = voter.thin(fv, vp, vt)

skinner = FaultSkinner(min_likelihood=0.7, min_skin_size=20)
skins = skinner.find_skins(fvt, vp, vt)
```

The self-contained example can be run without external data:

```bash
python examples/run_3d_synthetic_skinning.py
```

Pass `--output-dir` only when generated DAT outputs and a small text skin
summary should be written.

## Limitations

This implementation does not reproduce the Java linked-cell topology,
neighbor-link growth rules, skin smoothing, or real-data workflow helpers. It
is intended as a practical MVP for extracting connected components from
thinned 3D vote volumes.
