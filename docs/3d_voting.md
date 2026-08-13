# 3D Voting Conventions

`pyosv.voting3d.OptimalSurfaceVoter` converts 3D fault-likelihood, strike, and
dip volumes into a normalized vote-likelihood volume and associated vote
orientations. It also provides voter-thinning policies for producing a sparse
fault-surface candidate volume.

The implementation follows the processing structure of
`reference_osv/src/osv/OptimalSurfaceVoter.java` where practical, but it is not
a bit-exact Mines JTK port. PyOSV uses NumPy, SciPy, and optional Numba kernels
and has no JVM, Jython, Gradle, or Mines JTK runtime dependency.

## Array and coordinate contract

Global image volumes use shape `(n3, n2, n1)` and array indexing
`array[i3, i2, i1]`. Geometric vectors use component order `(x1, x2, x3)`.

Seed-local voting uses a `(w, v, u)` coordinate system:

```text
u = fault-normal lag direction
v = fault-dip direction
w = fault-strike direction
```

A local cost volume has shape `(nw, nv, nu)` and indexing
`cost[kw, kv, ku]`. For integer local offsets `iw`, `iv`, and lag `iu`, the
corresponding global point is:

```text
seed + iw * strike + iv * dip + iu * normal
```

The local array indices are related to the configured radii by:

```text
kw = iw + rw
kv = iv + rv
ku = iu + ru
```

Public voting inputs must be finite numeric 3D arrays with matching shapes.
They are interpreted as:

| Input | Contract |
| --- | --- |
| `ft` | Fault likelihood used for seed selection and voting evidence. |
| `pt` | Strike in degrees. |
| `tt` | Dip in degrees. |

The returned tuple is `(fv, vp, vt)`:

| Output | Contract |
| --- | --- |
| `fv` | Normalized `float32` vote likelihood in `[0, 1]`. |
| `vp` | Strike associated with the strongest individual local vote at each sample. |
| `vt` | Dip associated with the strongest individual local vote at each sample. |

All three outputs have the same `(n3, n2, n1)` shape as the inputs. When no
seed is selected, all three outputs are zero.

## Constructor and defaults

```python
from pyosv.voting3d import OptimalSurfaceVoter

voter = OptimalSurfaceVoter(ru=6, rv=8, rw=8)
```

`ru`, `rv`, and `rw` must be nonnegative integers. The constructor establishes
this configuration:

| Setting | Default |
| --- | --- |
| normal lag bounds | `lmin=-ru`, `lmax=ru`, `nl=2*ru+1` |
| reciprocal strain spacing | `bstrain1=4`, `bstrain2=4` |
| attribute-smoothing passes | `1` |
| extracted-surface smoothing | `surface_smoothing1=2.0`, `surface_smoothing2=2.0` |
| orientation-only surface smoothing | `float(max(rv, rw))` |
| surface-orientation backend | `"full_surface"` |
| final vote-map smoothing | `0.0` |
| support minimum | `0.0` |
| support exponent | `0.0` |
| surface-voting boundary policy | `"reference"` |

The normal-lag bounds for each tangential position are stored in `lmins` and
`lmaxs`, each with shape `(2*rw+1, 2*rv+1)`. For tangential radius
`sqrt(iw**2 + iv**2)`:

```text
radius <= 2:
    lmin = lmax = 0
radius > 2:
    shift = min(floor(radius + 0.5), ru)
    lmin = -shift
    lmax =  shift
```

`set_strain_max(strain_max1, strain_max2)` converts each maximum strain to a
positive integer spacing with `ceil(1 / strain_max)`. Each value must satisfy
`0 < strain_max <= 1`.

## Basic voting workflow

```python
import numpy as np

from pyosv.voting3d import OptimalSurfaceVoter

ft = np.zeros((64, 96, 128), dtype=np.float32)
pt = np.zeros_like(ft)
tt = np.full_like(ft, 90.0)

voter = OptimalSurfaceVoter(ru=6, rv=8, rw=8)
voter.set_strain_max(0.25, 0.25)

fv, vp, vt = voter.apply_voting(
    d=4,
    fm=0.3,
    ft=ft,
    pt=pt,
    tt=tt,
)
fvt = voter.thin(fv, vp, vt)
```

`apply_voting(...)` executes this stage order:

```text
seed selection from input ft/pt/tt
  -> fault-likelihood smoothing and unit-range normalization
  -> per-seed UVW cost sampling
  -> dynamic-programming surface extraction
  -> surface scoring and support policy
  -> surface-orientation estimation
  -> vote and orientation accumulation
  -> final vote-map normalization
```

`apply_voting_from_seeds(...)` accepts an explicit ordered sequence of
`FaultCell` objects and executes the same stages after seed selection. Seeds
must lie inside the shared input volume. Diagnostics follow the supplied seed
order.

## Seed selection

`pick_seeds(d, fm, ft, pt, tt)` uses the following deterministic contract:

1. A sample is a candidate only when `ft > fm`.
2. Candidates are ordered by descending likelihood.
3. Equal likelihoods use descending C-order flat index.
4. A candidate is accepted when the Chebyshev box of radius `d` around it
   contains no previously accepted seed center.
5. Each accepted sample becomes a `FaultCell` containing `(i1, i2, i3)`,
   likelihood, strike, and dip.

The suppression radius is bounded internally by the largest distance that can
be distinguished for the input shape. `d=0` accepts every threshold candidate
in deterministic score order.

`get_seeds(c1, c2, c3, ft, pt, tt)` returns the one `FaultCell` at the requested
sample and does not apply a likelihood threshold.

Seed selection uses the original input attributes. The likelihood smoothing
used for local voting occurs after the seed sequence has been determined.

## Distinct smoothing stages

The voter contains several independent smoothing controls:

| Stage | Control | Contract |
| --- | --- | --- |
| voting evidence | fixed sigma `1.0` | Smooths `ft`, subtracts its minimum, and divides by its maximum when positive before UVW costs are built. |
| DP attribute smoothing | `attribute_smoothing` | Repeats nonlinear forward/reverse cost accumulation before surface extraction. |
| extracted surface | `surface_smoothing1`, `surface_smoothing2` | Smooths the selected `u(w,v)` surface; setting both to zero disables this stage. |
| surface orientation | `surface_orientation_smoothing` | Smooths only for strike/dip re-estimation and does not change the extracted voting surface. |
| final vote map | `final_normalization_smoothing` | Optionally smooths accumulated vote evidence immediately before final normalization. |

For a surface with shape `(nw, nv)`, `surface_smoothing1` acts along the `v`
axis and `surface_smoothing2` acts along the `w` axis.

## Dynamic-programming surface contract

`pyosv.dp.find_surface_3d` consumes a finite local cost volume and returns a
lag surface:

```text
cost shape    = (nw, nv, nu)
surface shape = (nw, nv)
surface value = selected u lag
```

The unmasked extractor performs the configured number of 3D attribute-smoothing
passes, solves one optimal `v-u` path for each `w` row, and optionally smooths
the resulting surface. Flat-cost path ties prefer the center lag. Returned lags
are finite `float32` values.

`bstrain1` constrains lag variation along `v`; `bstrain2` constrains variation
along `w`. Three-dimensional attribute smoothing applies the corresponding DP
operation in both tangential directions.

Recursive filters from the reference implementation are represented by
SciPy-backed smoothing helpers. Their kernels and boundary responses are part
of the Python numerical contract rather than Mines JTK equality.

## Surface orientation

Each accepted surface vote carries strike and dip derived from centered
`du/dv` and `du/dw` differences on the local `u(w,v)` surface.

`set_surface_orientation_smoothing(sigma)` accepts a nonnegative finite value.
The constructor default is `max(rv, rw)`. A value of `0.0` computes orientation
from the unsmoothed surface.

`set_surface_orientation_backend(...)` accepts:

| Backend | Contract |
| --- | --- |
| `"full_surface"` | Smooths the complete surface before evaluating center differences. |
| `"center_separable"` | Computes only the smoothed `3 x 3` center patch with separable Gaussian weights and nearest-edge folding. |

Both backends require at least three samples along `w` and `v` when
surface-derived orientation is evaluated. The masked boundary policy defines
seed-orientation fallbacks for cropped or smaller surfaces.

## Support policy

```python
voter.set_surface_support_policy(
    min_fraction=0.5,
    exponent=1.0,
)
```

`min_fraction` must lie in `[0, 1]`; `exponent` must be nonnegative and finite.
For one seed:

- the vote is skipped when `support_fraction < min_fraction`;
- otherwise, a positive exponent multiplies the average vote by
  `support_fraction ** exponent`.

The default `(0.0, 0.0)` is a no-op. The definition of `support_fraction`
depends on the selected boundary policy.

## Reference surface-voting policy

`surface_voting_boundary_policy="reference"` is the constructor default. It
uses the following contract for each seed.

### UVW sampling

The seed-local cost box is initialized to `1.0`. Only normal lags admitted by
`lmins` and `lmaxs` are sampled. Global coordinates are computed in `float32`,
rounded with:

```text
floor(x + 0.5)
```

and clamped independently to the volume bounds. The admitted cost is:

```text
1 - voting_evidence
```

The public `samples_in_uvw_box(...)` method always uses this reference-clamping
contract, independently of the boundary policy selected for `apply_voting`.
Standalone calls do not require the supplied `fx` values to lie in `[0, 1]`.

### Surface scoring and boundaries

The extracted surface is scored and accumulated only at rounded points that
satisfy:

```text
0 <= i1 < n1
0 <  i2 < n2 - 1
0 <  i3 < n3 - 1
```

An `i1` face sample is therefore permitted, while `i2` and `i3` face samples
are excluded. Scoring and accumulation use the same target predicate.

The reference support fraction is:

```text
valid surface point count / surface.size
```

A surface with no valid point is skipped. Support thresholding and weighting
are then applied before orientation and accumulation.

### Accumulation

Every accepted surface point adds its average surface vote `fa` to the
accumulated evidence `fe`. The orientation maps and internal maximum-vote map
are updated only when:

```text
fa > previous strongest local vote
```

Equal or weaker votes retain the previously stored orientation.

Each accepted center write receives two reinforcement writes:

- when `abs(normal[2]) > abs(normal[1])`, write to `i3-1` and `i3+1`;
- otherwise, write to `i2-1` and `i2+1`.

Reinforcement writes use full-volume bounds checks. The center-point predicate
is not widened by those defensive checks.

## Masked in-bounds surface-voting policy

```python
voter.set_surface_voting_boundary_policy("masked_in_bounds")
```

`masked_in_bounds` is a separate boundary contract with mask-aware sampling,
DP, scoring, and accumulation.

A local lag is valid only when both conditions hold:

1. it is admitted by `lmins` and `lmaxs`;
2. its Java-rounded global coordinate is inside the volume.

Out-of-bounds lags are not clamped. Their cost remains `1.0`, their validity
mask remains false, and they cannot participate in masked attribute smoothing,
accumulation, backtracking, or scoring.

A tangential `(w, v)` column is supported when it contains at least one valid
normal lag. The policy selects the largest all-supported axis-aligned rectangle
that contains the local origin. Ties prefer lower origin asymmetry and then
lexicographically smaller `(w_start, v_start, w_stop, v_stop)`. Cropped
rectangles retain explicit `w_offset` and `v_offset` values for mapping back to
global coordinates.

After optional surface smoothing, the surface is checked against the validity
mask and strain limits in both tangential directions. Deterministic global
recovery is used when the smoothed result is not jointly feasible. A fractional
lag already inside a valid Java-rounding cell is retained when it satisfies the
mask and both strain constraints. The seed is skipped when no jointly feasible
surface exists.

Scoring and accumulation validate every selected lag before writing any vote.
A detected invalid selected sample skips the seed without partially committing
its surface. Center writes may occur on all six volume faces; reinforcement
writes remain bounds checked.

The masked support fraction is:

```text
valid selected point count / ((2*rw+1) * (2*rv+1))
```

The denominator is the full tangential patch, not the selected rectangle.

Orientation is selected as follows:

| Surface domain | Orientation source |
| --- | --- |
| full box with at least `3 x 3` samples | surface-derived orientation |
| cropped box | seed strike/dip, recorded as `seed_boundary_fallback` |
| smaller full box | seed strike/dip, recorded as `seed_small_surface_fallback` |

Stable skip reasons are:

```text
no_supported_origin
no_feasible_surface
no_valid_surface_samples
support_below_min_fraction
invalid_selected_sample
```

`surface_projection_count` is the number of `(w, v)` columns whose lag differs
between the raw smoothed surface and the final mask-and-strain-feasible surface.
Each changed column is counted once. It is not a count of recovery iterations;
the no-surface-smoothing path reports zero.

## Surface-voting diagnostics

Every `apply_voting(...)` or `apply_voting_from_seeds(...)` run replaces the
previous per-seed diagnostics with an immutable tuple in seed order. Each entry
contains:

```text
seed_index
policy
full_tangential_column_count
selected_tangential_column_count
admissible_lag_count
in_bounds_lag_count
support_fraction
surface_center_lag
surface_projection_count
selected_invalid_sample_count
center_vote_write_count
face_center_vote_count
orientation_source
skipped
skip_reason
```

The diagnostics are available through `surface_voting_diagnostics` and
`last_surface_voting_diagnostics`.

`surface_voting_diagnostic_summary()` returns JSON-safe aggregate fields for
the most recent run. Its `policy` field identifies the policy used by that run;
changing the configured policy without executing another run does not rewrite
stored diagnostics. Support fractions are accumulated explicitly from left to
right so serialized summaries remain stable across supported Python versions.

## Final vote-map normalization

After all seed votes are accumulated, `apply_voting(...)` transforms `fe` into
`fv` as follows:

```text
optional Gaussian smoothing
  -> subtract global minimum
  -> divide by the post-subtraction maximum when positive
  -> apply 1 - (1 - x) ** 8
  -> clip to [0, 1]
```

The constructor default `final_normalization_smoothing=0.0` disables the first
step. A constant input becomes an all-zero vote volume.

```python
voter.set_final_normalization_smoothing(1.0)
```

This setter accepts a nonnegative finite sigma. It affects `fv` only; it does
not change `vp` or `vt`.

## Voter thinning

`OptimalSurfaceVoter.thin(fv, vp, vt, ...)` validates finite matching global
volumes and returns one `float32` thinned vote volume with the same shape. It
accepts five modes.

### Reference thinning

`mode="reference"` is the default. It:

1. smooths `fv` along `i3` and `i2`, but not `i1`;
2. folds `vp` into `[0, 180)` and selects a horizontal, diagonal, or vertical
   comparison direction in the `i2-i3` plane;
3. keeps strict local maxima against the two directional neighbors;
4. writes the smoothed likelihood at retained samples;
5. for folded strikes strictly between 60 and 120 degrees, also writes the
   retained value to the adjacent `i3-1` sample, clamped at the lower face.

```python
fvt = voter.thin(
    fv,
    vp,
    vt,
    mode="reference",
    reference_sigma=1.0,
)
```

Voter reference thinning does not apply scanner edge cleanup.

### Fault-normal thinning

`mode="normal"` smooths `fv` with sigma `1.0`, samples one fault-normal step in
both directions with linear nearest-boundary interpolation, retains strict
maxima, and writes the original `fv` values.

```python
fvt = voter.thin(fv, vp, vt, mode="normal")
```

### Plateau-aware fault-normal thinning

`mode="normal_plateau"` uses non-strict fault-normal comparisons within
`plateau_tolerance`. Candidates are grouped into contiguous runs along each
sample's dominant normal axis. The largest `plateau_tie_breaker` value is
retained; an all-equal run retains its center sample. When no tie-breaker volume
is supplied, `fv` is used.

### Hybrid thinning

`mode="hybrid"` computes local strike/dip roughness over positive vote support.
It selects reference-thinning output where roughness does not exceed
`hybrid_orientation_gradient_threshold` and fault-normal output in rougher
regions.

### Hybrid v2 thinning

`mode="hybrid_v2"` uses vote support above `1e-6`. It starts from reference
thinning, adopts positive fault-normal candidates in rough regions, and uses
plateau candidates near all volume faces when fault-normal output is absent or
locally sparse.

The related defaults are:

| Setting | Default |
| --- | --- |
| `reference_sigma` | `1.0` |
| `hybrid_orientation_gradient_threshold` | `8.0` degrees |
| `hybrid_v2_edge_margin` | `2` samples |
| `plateau_tolerance` | `1e-6` |
| `plateau_tie_breaker` | `fv` |

Thinning mode is independent of scanner backend, scanner thinning, workflow,
surface-voting boundary policy, and skinning policy.

## Scanner and voter thinning distinction

Scanner and voter reference thinning share the strike-binned NMS helper but
have different contracts.

| Property | Scanner reference thinning | Voter reference thinning |
| --- | --- | --- |
| Input | scanner `ft`, `pt`, `tt` | voter `fv`, `vp`, `vt` |
| Retained value | smoothed scanner likelihood | smoothed vote likelihood |
| folded-strike reinforcement | disabled | enabled |
| edge cleanup | applied by default on selected `i2` and `i3` faces | not applied |
| returned data | likelihood, strike, and dip volumes | likelihood volume only |
| rejected orientation value | zero in returned strike/dip arrays | not exposed |

The voter must not inherit scanner edge cleanup implicitly. Evaluation and
publication configurations record the two thinning stages independently.

## Integration with scanner output

Scanner attributes can be thinned and passed directly to the voter:

```python
from pyosv.orient3d import FaultOrientScanner3
from pyosv.voting3d import OptimalSurfaceVoter

scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
ft, pt, tt = scanner.scan(
    phi_min=0.0,
    phi_max=180.0,
    theta_min=45.0,
    theta_max=90.0,
    g=image,
)
fet, fpt, ftt = scanner.thin(ft, pt, tt)

voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
fv, vp, vt = voter.apply_voting(
    d=3,
    fm=0.5,
    ft=fet,
    pt=fpt,
    tt=ftt,
)
fvt = voter.thin(fv, vp, vt)
```

All global arrays in this sequence use shape `(n3, n2, n1)`. Scanner backend,
scanner thinning, downstream workflow, surface-voting boundary policy, voter
thinning, and skinning are separate configuration axes.

## Verification and equivalence boundary

The Python suite verifies:

- constructor defaults, setters, validation, and per-instance state;
- lag ranges, Java-style rounding, coordinate order, and UVW costs;
- deterministic seed ordering and exact greedy suppression;
- unmasked DP paths, surfaces, smoothing, strain behavior, and tie handling;
- mask-aware state validity, bidirectional strain feasibility, deterministic
  recovery, and projection accounting;
- reference target boundaries, vote reinforcement, and strongest-orientation
  updates;
- reference and masked policy routing, support controls, crop offsets,
  all-face masked writes, and immutable diagnostics;
- final normalization, surface orientation, and all voter-thinning modes;
- Python and Numba agreement for accelerated kernels;
- Synthetic and F3 evaluation contracts that consume voter outputs.

Tests do not require bit-exact Java or Mines JTK arrays. F3 public outputs are
comparison targets rather than geological truth or method-level Java fixtures.
`reference_osv/` is read-only reference material and is not a runtime
dependency.

## Related specifications

- [3D Voter Reference Mapping](reference_mapping_voting3d.md)
- [3D Orientation Scanning](orient3d.md)
- [Reference-Like 3D Thinning](reference_like_thinning.md)
- [Skinning](skinning.md)
- [Mode Comparison Contract](mode_comparison.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [F3 3D Reference Data Validation](f3d_validation.md)
