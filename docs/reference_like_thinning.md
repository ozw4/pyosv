# Reference-Like 3D Thinning

`pyosv.thinning3d` defines the shared strike-binned nonmaximum-suppression
contract used by scanner and voter reference thinning.

Global arrays use shape `(n3, n2, n1)` and indexing
`array[i3, i2, i1]`. Strike and dip values are expressed in degrees. Public
thinning functions convert numeric inputs to `float32`, require finite values,
and require matching three-dimensional shapes.

The implementation follows the strike-binned structure of the Java reference
where practical, but it is not a bit-exact Mines JTK port. Smoothing uses SciPy
Gaussian filters, and returned values follow the Python array, boundary, and
sentinel contracts described below.

## Independent configuration axes

These settings are independent:

- scanner backend;
- scanner-thinning mode;
- downstream workflow;
- surface-voting boundary policy;
- voter-thinning mode;
- skinning policy.

A scanner backend named `reference-like` does not select scanner thinning.
Likewise, `mode="reference"` on scanner or voter thinning does not select a
workflow. Reports must record each setting separately.

## Shared strike-binned NMS

The shared helpers are:

```python
from pyosv.thinning3d import (
    reference_like_3d_nms_mask,
    reference_like_3d_thin_values,
    remove_reference_edge_effects_3d,
)
```

### Validation and smoothing

`reference_like_3d_nms_mask(values, strike, ...)` and
`reference_like_3d_thin_values(values, strike, ...)` require finite matching
`(n3, n2, n1)` arrays. `strike` is interpreted in degrees.

Before comparison, `values` is smoothed with:

```text
scipy.ndimage.gaussian_filter(
    values,
    sigma=(sigma, sigma, 0.0),
    mode="nearest",
)
```

Smoothing therefore acts along `i3` and `i2`, but not along `i1`.
`sigma` must be finite and nonnegative. A zero sigma, or an empty input, returns
a `float32` copy without filtering. Inputs are not modified.

### Strike bins and comparison directions

Strike is folded into `[0, 180)` with `strike % 180`. Each sample is compared
with the two neighbors in the direction selected by its folded strike:

| Folded strike | Bin | Neighbor offset `(d3, d2)` |
| --- | --- | --- |
| `[0, 22.5)` or `[157.5, 180)` | horizontal | `(0, 1)` |
| `[22.5, 67.5)` | positive diagonal | `(1, -1)` |
| `[67.5, 112.5)` | vertical | `(1, 0)` |
| `[112.5, 157.5)` | negative diagonal | `(1, 1)` |

For offset `(d3, d2)`, the center is compared with
`(i3+d3, i2+d2)` and `(i3-d3, i2-d2)` at the same `i1`.

With `strict=True`, a sample is retained only when:

```text
center > plus_neighbor
and
center > minus_neighbor
```

With `strict=False`, both comparisons use `>=`. Samples that do not have both
neighbors for their selected direction are outside the comparison region and
are not retained.

### Public helper outputs

`reference_like_3d_nms_mask(...)` returns a boolean mask with the same shape as
the inputs. Its `strict` argument defaults to `True`.

`reference_like_3d_thin_values(...)` always uses strict comparison and returns:

```text
(thinned_values, keep_mask)
```

`thinned_values` is a zero-initialized `float32` array. Retained positions
receive the smoothed comparison value, not the original input value.
`keep_mask` identifies the strict local maxima before optional retained-sample
reinforcement.

### Retained-sample reinforcement

`reference_like_3d_thin_values(..., reinforce_vertical=True)` applies the voter
reinforcement rule after strict NMS.

Strike is first mapped to `[0, 360)` and folded around 180 degrees. A retained
sample is reinforced only when its folded strike satisfies the strict interval:

```text
60 < folded_strike < 120
```

For each qualifying sample at `(i3, i2, i1)`, the same smoothed value is copied
to:

```text
(max(i3 - 1, 0), i2, i1)
```

The reinforcement is one-sided toward lower `i3`. The returned `keep_mask`
continues to describe the original strict maxima and does not include the
reinforced neighbor.

## Scanner thinning

`FaultOrientScanner3.thin(ft, pt, tt, ...)` returns:

```text
(fet, fpt, ftt)
```

All three outputs are `float32` arrays with the same global shape. `fpt` and
`ftt` copy the input strike and dip only at retained samples. Every other
orientation sample is zero.

### Reference scanner thinning

The default call is:

```python
fet, fpt, ftt = scanner.thin(
    ft,
    pt,
    tt,
    mode="reference",
    reference_sigma=1.0,
    remove_edge_effects=True,
)
```

The reference scanner path:

1. calls `reference_like_3d_thin_values(ft, pt, ...)`;
2. disables retained-sample reinforcement;
3. writes smoothed likelihood at strict strike-binned maxima;
4. applies scanner edge cleanup when `remove_edge_effects=True`;
5. copies input strike and dip only at the final retained samples.

`reference_sigma` controls the shared `i3-i2` Gaussian smoothing and must be
finite and nonnegative.

### Scanner edge cleanup

`remove_reference_edge_effects_3d(values, strike, dip)` returns:

```text
(cleaned_values, cleaned_strike, cleaned_dip, keep_mask)
```

A nonzero input value is treated as retained before cleanup. The fault-normal
components used by this operation are:

```text
w2 =  sin(dip) * cos(strike)
w3 = -sin(dip) * sin(strike)
```

The fixed edge width is five samples. A retained sample is removed when either
condition holds:

```text
sample is within five samples of an i3 face
and w3**2 > cos(30 degrees)**2
```

```text
sample is within five samples of an i2 face
and w2**2 > cos(30 degrees)**2
```

No `i1` face cleanup is applied. Removed values, strikes, and dips are all set
to zero. The component comparison is strict. When `n2` or `n3` is no greater
than twice the edge width, the two face bands cover that complete axis.

Set `remove_edge_effects=False` when the selected scanner contract excludes
this cleanup:

```python
fet, fpt, ftt = scanner.thin(
    ft,
    pt,
    tt,
    mode="reference",
    reference_sigma=1.0,
    remove_edge_effects=False,
)
```

### Fault-normal scanner thinning

`mode="normal"` performs local maximum suppression along the sampled fault
normal derived from `pt` and `tt`:

```python
fet, fpt, ftt = scanner.thin(ft, pt, tt, mode="normal")
```

For each sample, the original `ft` volume is linearly sampled one fault-normal
step in both directions with nearest-boundary handling. A sample is retained
only when it is positive and strictly greater than both sampled neighbors.
Retained likelihoods keep the original `ft` values.

Scanner edge cleanup is not applied in `mode="normal"`.

## Voter thinning

`OptimalSurfaceVoter.thin(fv, vp, vt, ...)` requires finite matching global
volumes and returns one `float32` thinned vote volume.

### Reference voter thinning

The default call is:

```python
fvt = voter.thin(
    fv,
    vp,
    vt,
    mode="reference",
    reference_sigma=1.0,
)
```

This path calls the shared strike-binned helper with
`reinforce_vertical=True`. It therefore:

- smooths `fv` along `i3` and `i2`, but not `i1`;
- uses `vp` to select the strike comparison direction;
- keeps strict directional maxima;
- writes smoothed vote likelihood at retained samples;
- applies the one-sided lower-`i3` reinforcement for folded strikes strictly
  between 60 and 120 degrees;
- does not apply scanner edge cleanup.

`vt` is validated as part of the matching voter attribute contract, but the
reference strike-binned comparison itself uses `fv` and `vp`.

### Additional voter-thinning modes

The voter accepts five modes in total:

| Mode | Contract |
| --- | --- |
| `reference` | Shared strike-binned NMS with voter reinforcement. |
| `normal` | Smooths `fv` with `smooth3d(..., 1.0)`, samples one fault-normal step in both directions, keeps strict maxima, and writes original `fv` values. |
| `normal_plateau` | Allows near-equal normal-direction candidates within `plateau_tolerance`, groups contiguous candidates along each sample's dominant normal axis, and retains one sample per run using `plateau_tie_breaker`. |
| `hybrid` | Uses reference output where orientation roughness is at or below the threshold and normal output where roughness exceeds it. |
| `hybrid_v2` | Starts from reference output, substitutes positive normal candidates in rough regions, and adds a plateau fallback near volume faces when the normal result is absent or locally sparse. |

The voter defaults are:

```text
reference_sigma = 1.0
hybrid_orientation_gradient_threshold = 8.0
hybrid_v2_edge_margin = 2
plateau_tolerance = 1.0e-6
plateau_tie_breaker = fv
```

`normal_plateau` requires `fv > 1.0e-6` for a plateau candidate. If the
tie-breaker is constant across one run, the retained position is its lower
center sample for an even-length run and its center sample for an odd-length
run. Otherwise, the first maximum tie-breaker position is retained.

`hybrid` computes orientation roughness only across positive `fv` support.
`hybrid_v2` uses support above `1.0e-6`; its face region includes all six volume
faces within `hybrid_v2_edge_margin`.

These modes select voter thinning only. They do not change scanner output,
scanner thinning, voting boundary policy, or workflow configuration.

## Scanner and voter reference contracts

Scanner and voter reference thinning use the same strict strike-binned helper
but differ at their call boundaries:

| Property | Scanner reference thinning | Voter reference thinning |
| --- | --- | --- |
| Inputs | `ft`, `pt`, `tt` | `fv`, `vp`, `vt` |
| Strike used for NMS | `pt` | `vp` |
| Retained value | smoothed scanner likelihood | smoothed vote likelihood |
| Lower-`i3` reinforcement | disabled | enabled for folded strike in `(60, 120)` |
| Edge cleanup | selected by `remove_edge_effects`, default `True` | not applied |
| Return value | likelihood, strike, and dip arrays | likelihood array |
| Nonretained orientation | zero | not returned |

Scanner and voter thinning must remain separate report fields even when both are
set to `reference`.

## Evaluation and reporting

For every comparison, record at least:

```text
scanner_backend
scanner_thin_mode
workflow_mode
surface_voting_boundary_policy
voter_thin_mode
```

F3 publication evaluation uses the complete volume as one evaluation unit.
Crops, slices, regions, and tiles are local diagnostics and are not independent
replicates. F3 public outputs are comparison targets rather than geological
truth.

Useful thinning diagnostics include nonzero fraction, buffered ridge overlap,
sparse-ridge distance, and fixed-threshold ridge overlays. Interpret those
metrics together with the selected scanner, workflow, boundary, and voter
settings; a thinning metric alone does not identify which upstream stage caused
a difference.

## Verification boundary

The test suite covers:

- finite-input, shape, dtype, and nonmutation contracts;
- all four strike bins and their directional neighbors;
- strict and non-strict NMS behavior;
- directional boundary exclusion;
- `i3-i2` smoothing and retained smoothed values;
- strict reinforcement-angle boundaries and lower-`i3` writes;
- scanner `i2`/`i3` edge cleanup and the absence of `i1` cleanup;
- scanner reference and fault-normal outputs;
- voter reference, normal, plateau, and hybrid modes;
- separation between scanner cleanup and voter reinforcement.

The tests require deterministic Python behavior and numerical tolerances where
SciPy filtering is involved. They do not require bit-exact Java or Mines JTK
arrays.

## Related specifications

- [3D Orientation Scanning](orient3d.md)
- [3D Voting Conventions](3d_voting.md)
- [3D Scanner Reference Mapping](reference_mapping_orient3d.md)
- [3D Voter Reference Mapping](reference_mapping_voting3d.md)
- [Mode Comparison Contract](mode_comparison.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [F3 3D Reference Data Validation](f3d_validation.md)
