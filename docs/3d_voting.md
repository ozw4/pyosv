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

The orientation smoothing backend defaults to `"full_surface"`, which retains
the existing full-surface SciPy filtering path and its numerical output. A
center-only separable backend is available as an explicit opt-in:

```python
voter.set_surface_orientation_backend("center_separable")
```

The opt-in backend reproduces SciPy's Gaussian kernel truncation and nearest-edge
behavior while computing only the center patch needed for strike/dip. Tiny
floating-point differences remain possible. It is not selected by default or by
the default synthetic-quality variants. Only `"full_surface"` and
`"center_separable"` are accepted.

Surface voting defaults to `surface_voting_boundary_policy="reference"`.
This keeps the Java-reference-oriented boundary behavior unchanged:
`samples_in_uvw_box` rounds with Java-style `floor(x + 0.5)` and clamps sampled
image coordinates to the volume, while vote averaging and accumulation permit
an `i1` face sample but require `i2` and `i3` source samples to be interior.
Crop-edge votes near `i2` and `i3` faces can therefore be weaker than older
`pyosv` results that accepted all in-bounds boundary source samples.

## Boundary-aware Surface Voting

`masked_in_bounds` is an explicit quality experiment for boundary-aware UVW
surface voting:

```python
voter.set_surface_voting_boundary_policy("masked_in_bounds")
```

Only `"reference"` and `"masked_in_bounds"` are accepted. Selecting the
masked policy does not change the public `samples_in_uvw_box(...)` API or its
reference clamping behavior. Instead, the surface-voting path uses an internal
sampler that returns `float32` costs and a boolean lag mask. A lag is valid only
when it is admitted by `lmins` / `lmaxs` and its Java-rounded global sample is
inside the volume. Out-of-bounds lags are not clamped, are not treated as
evidence, and cannot be selected by dynamic programming; no assumption is made
that fault-likelihood values lie in `[0, 1]`.

For each `(w, v)` column, the masked path marks the column supported when at
least one lag is valid. It extracts the largest all-supported, axis-aligned
rectangle containing the local origin. Equal-area rectangles are ordered first
by lower origin asymmetry and then lexicographically by
`(w_start, v_start, w_stop, v_stop)`. The selected rectangle carries explicit
offsets into the full UVW box, so a cropped surface is mapped back to volume
coordinates without assuming `kw-rw` or `kv-rv`.

Surface extraction then uses a private mask-aware DP path. Invalid states stay
out of attribute smoothing, forward/reverse accumulation, and backtracking. A
seed is skipped when no strain-feasible surface exists. After optional surface
smoothing, the raw smoothed surface is revalidated against the mask and the
strain limits in both tangential directions. If it is not jointly feasible, a
deterministic global recovery is attempted over the full selected rectangle;
independent nearest-column projection is not sufficient. Recovery does not move
a lag to an integer center merely because it is fractional: a value already in
a valid Java-rounding cell is retained when the mask and both strain directions
permit it. If no global mask-and-strain-feasible surface can be recovered, the
seed is skipped with `skip_reason="no_feasible_surface"`.

`surface_projection_count` compares the raw smoothed surface with the final
mask-and-strain-feasible surface. It counts the `(w, v)` columns whose value
changed, with each changed column counted once even if global recovery adjusted
it more than once. When surface smoothing is disabled, the existing diagnostic
contract reports zero smoothing projections.

Masked vote scoring uses only valid selected volume samples. Its support
fraction is the number of valid selected columns divided by the full
`(2*rw+1)*(2*rv+1)` tangential area, not the cropped area; the existing support
minimum and exponent apply to that value. Center votes may be written on all
six volume faces, while reinforcement writes retain defensive bounds checks.
Any selected invalid sample causes the seed vote to be skipped. A full-box
surface keeps `_surface_strike_and_dip`; an asymmetric/cropped rectangle uses
the seed `cell.fp` / `cell.ft` orientation and records
`orientation_source="seed_boundary_fallback"`. This deliberately avoids adding
a new local-normal model in the boundary experiment.

Each `apply_voting_from_seeds` run replaces the previous immutable per-seed
diagnostics. They record the seed index, policy, full and selected tangential
column counts, admissible and in-bounds lag counts, support fraction, center
lag, smoothing projections, invalid selected samples, center/face vote counts,
orientation source, and any skip reason. Stable skip reasons include
`no_supported_origin`, `no_feasible_surface`, `no_valid_surface_samples`,
`support_below_min_fraction`, and `invalid_selected_sample`. The summary exposes
only JSON-serializable counts and support statistics; it does not retain
volumes or per-voxel masks.

`masked_in_bounds` is not a reference-equivalence mode and is not enabled by the
default `reference` or `quality` workflows or by the `current_default` report
variant. It must be selected explicitly, including through the synthetic-report
`boundary_aware_voter_v1` variant.

`OptimalSurfaceVoter.thin` keeps local maxima from `fv` and returns a thinned
`float32` vote volume with the same shape. The default is the reference-like
strike-bin mode: it uses SciPy smoothing before strike-binned comparison in
the `i2-i3` plane. The legacy pyosv path remains available with
`mode="normal"` and uses SciPy interpolation along fault normals derived from
`vp` and `vt`. `mode="normal_plateau"` is a diagnostic fault-normal variant
that collapses contiguous plateau runs along each sample's dominant normal axis
and can use `plateau_tie_breaker` to choose the retained layer. `mode="hybrid"`
keeps reference-like thinning in stable-orientation regions and uses
fault-normal thinning where local orientation changes rapidly. `mode="hybrid_v2"`
is a diagnostic variant that starts from reference-like thinning, adopts
positive fault-normal candidates in rough-orientation regions, and applies an
edge-region plateau fallback. Voter reference-like thinning may reinforce
retained near-vertical strike samples, and it does not apply scanner
edge-effect cleanup. These modes are not bit-exact Mines JTK implementations.

For F3 diagnostics and other reference-style comparisons, the default call is
the reference-like strike-bin path:

```python
fvt = voter.thin(fv, vp, vt, reference_sigma=1.0)
```

Reference-like thinning uses `fv` for values and `vp` for strike-angle bins,
compares local maxima in the `i2-i3` plane, and writes the smoothed comparison
values to retained samples. `reference_sigma` controls smoothing inside the
comparison helper. Code that needs the old fault-normal voter thinning should
pass `mode="normal"` explicitly. Boundary diagnostics that need plateau-aware
fault-normal thinning can use `mode="normal_plateau"` or the hybrid
`mode="hybrid_v2"` fallback path; when `plateau_tie_breaker` is omitted, `fv`
is used for tie-breaking.

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
