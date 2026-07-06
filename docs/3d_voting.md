# 3D Voting Conventions

`pyosv` stores global 3D image volumes with shape `(n3, n2, n1)`. Sample
indices are addressed as `(i3, i2, i1)` in array indexing order, while vector
components follow the OSV component order `(x1, x2, x3)`.

Local 3D voting samples use a seed-centered `(w, v, u)` coordinate system with
array shape `(nw, nv, nu)`. The local axes are:

- `u`: fault-normal lag direction.
- `v`: fault-dip direction.
- `w`: fault-strike direction.

For orientation vectors, `u` is the fault normal, `v` is the dip vector, and
`w` is the strike vector. The component arrays for these vectors use
`(x1, x2, x3)` order even when they are sampled from global volumes stored as
`(n3, n2, n1)`.

`reference_osv/` is a read-only reference implementation. It is not a runtime
dependency of `pyosv`, and generated outputs should not be written under that
directory.

## Minimal Usage

`OptimalSurfaceVoter.apply_voting` runs the current 3D voting MVP on fault
likelihood, strike, and dip volumes. These volumes may come from
`FaultOrientScanner3` or from another workflow that follows the same shape and
angle conventions:

```python
import numpy as np

from pyosv.voting3d import OptimalSurfaceVoter

ft = np.zeros((64, 96, 128), dtype=np.float32)
pt = np.zeros_like(ft)
tt = np.full_like(ft, 90.0)

voter = OptimalSurfaceVoter(ru=6, rv=8, rw=8)
voter.set_strain_max(0.25, 0.25)
fv, vp, vt = voter.apply_voting(d=4, fm=0.3, ft=ft, pt=pt, tt=tt)
fvt = voter.thin(fv, vp, vt)
```

The returned `fv`, `vp`, and `vt` arrays have the same `(n3, n2, n1)` shape as
the inputs. `fv` is a normalized `float32` vote volume in `[0, 1]`; `vp` and
`vt` store the strike and dip angles associated with the strongest local vote at
each sample.

Final vote-map normalization follows the Java reference default. After all
votes are accumulated, `apply_voting` subtracts the global minimum from the vote
evidence, divides by the global maximum when it is nonzero, and applies the
`1 - (1 - x) ** 8` power transform. This final normalization step does not
smooth the vote map by default. The older practical behavior remains available
as an explicit opt-in:

```python
voter.set_final_normalization_smoothing(1.0)
```

This setting is only for the final `fe -> fv` vote-map normalization before the
power transform. It is separate from these other smoothing stages:

- `_smooth_fault_likelihood_3d(ft, sigma=1.0)`: input fault-likelihood
  smoothing before seeds and local voting are built.
- `surface_smoothing1` and `surface_smoothing2`: dynamic-programming surface
  extraction smoothing.
- `surface_orientation_smoothing=max(rv, rw)`: extracted-surface smoothing used
  only before vote strike/dip are re-estimated.

Reference-first workflows should not call
`set_final_normalization_smoothing(...)`. Use
`set_final_normalization_smoothing(1.0)` only when comparing with older pyosv
runs that smoothed the final vote map. Negative, nonfinite, boolean, and
nonnumeric values are rejected.

Surface orientation is re-estimated from each extracted local surface before
votes are accumulated. By default, `OptimalSurfaceVoter` smooths that surface
with `surface_orientation_smoothing=max(rv, rw)` before computing the
center-difference strike and dip. This corresponds to the reference
`surfaceStrikeAndDip` path, where `RecursiveGaussianFilter(max(rv,rw))` smooths
the picked surface before orientation is recomputed. In `pyosv`, the filter is a
SciPy-backed approximation through `smooth_surface_2d`, not a bit-exact Mines
JTK clone. This setting is separate from `surface_smoothing1` and
`surface_smoothing2`, which affect dynamic-programming surface extraction. Use
`voter.set_surface_orientation_smoothing(0.0)` to disable this orientation-only
smoothing for diagnostics that need raw-surface behavior; negative, nonfinite,
boolean, and nonnumeric values are rejected.

Surface-voting source samples follow the Java reference boundary rule: `i1`
may lie on an image face, but `i2` and `i3` must be interior before the sample
contributes to the average or vote accumulation. Crop-edge votes near `i2` and
`i3` faces can therefore be weaker than older `pyosv` results that accepted all
in-bounds boundary source samples.

`OptimalSurfaceVoter.thin` keeps local maxima from `fv` and returns a thinned
`float32` vote volume with the same shape. The default is the reference-like
strike-bin mode: it uses SciPy smoothing before strike-binned comparison in
the `i2-i3` plane. The legacy pyosv path remains available with
`mode="normal"` and uses SciPy interpolation along fault normals derived from
`vp` and `vt`. Voter reference-like thinning may reinforce retained
near-vertical strike samples, and it does not apply scanner edge-effect cleanup.
Neither mode is a bit-exact Mines JTK implementation.

For F3 diagnostics and other reference-style comparisons, the default call is
the reference-like strike-bin path:

```python
fvt = voter.thin(fv, vp, vt, reference_sigma=1.0)
```

Reference-like thinning uses `fv` for values and `vp` for strike-angle bins,
compares local maxima in the `i2-i3` plane, and writes the smoothed comparison
values to retained samples. `reference_sigma` controls smoothing inside the
comparison helper. Code that needs the old fault-normal voter thinning should
pass `mode="normal"` explicitly.

```python
fvt = voter.thin(fv, vp, vt, mode="normal")
```

## MVP Limitations

This is the 3D voting MVP, not the complete Java 3D fault interpretation
pipeline. The implementation is sequential and currently covered by synthetic
regression tests only.

See `docs/orient3d.md` for the approximate 3D orientation scanner. See
`docs/skinning.md` for the reference-like 3D skinning layer and explicit
connected-component fallback.
