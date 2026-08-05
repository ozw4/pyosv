# Fault-warping contract

`pyosv.fault_warping` defines the typed, Atlas-independent numerical contract
for estimating an apparent sample-axis shift from seismic signals on opposite
sides of a known fault surface.

## Contract version and implementation status

```python
FAULT_WARPING_CONTRACT_VERSION = "pyosv.fault_warping.v1"
```

This is a version of the Python API and its numerical semantics. It is not a
file format, serialization format, or Atlas artifact schema.

The module is **contract only** at this stage. It defines data contracts and an
estimator protocol, but implements no side sampling, interpolation, similarity
calculation, cost volume, dynamic warping, graph regularization, cycle
calculation, sub-sample refinement, or shift estimation. There is intentionally
no concrete estimator, public `estimate_fault_apparent_shifts()` function,
dummy result, `NotImplementedError` entry point, or zero-shift fallback.

## Ownership boundary

`pyosv.fault_warping` owns pure numerical processing contracts for:

- a 3D seismic-amplitude volume and a volume-wide valid mask;
- local coordinates, strike, dip, and topology for a known fault-surface graph;
- positive-side and negative-side definitions derived from a fault normal;
- reflector slopes and configuration needed to search sample-axis lags;
- row-aligned apparent shifts and numerical diagnostics;
- NumPy-array validation and numerical processing; and
- a common semantic contract for later Python and optional Numba backends.

It does not discover faults, run DL inference, run OSV voting, or perform
skinning. `FaultSurfaceGraph` is deliberately an array contract rather than a
direct `FaultSkin` dependency; a `FaultSkin` adapter is future work.

### Explicit non-ownership

This package must not import `seis_atlas` or `seis_fault_workflow`, and does
not own Atlas inference runs, `PreparedField`, workflow runs, skin runs, paths,
files, checksums, manifests, artifact writers, or artifact publication.

It also does not own DL fault probabilities, OSV-score fusion, surface
confidence calibration, accepted/review/rejected business classifications,
pseudo-label generation, PNG/HTML/CSV/NPZ publication, a CLI, viewer
integration, TWT-to-depth or metre conversion, or any true-slip, strike-slip,
full-vector-displacement, hanging-wall, or footwall estimate. Those concerns
belong to a later `seis_fault_workflow` integration layer.

This is a PyOSV native extension: it has no requirement to be identical to a
`reference_osv` feature. Documentation must distinguish it from
reference-compatible functionality rather than imply a nonexistent reference
equivalence.

## Public API and construction behavior

```python
from pyosv.fault_warping import (
    FAULT_WARPING_CONTRACT_VERSION,
    FaultSurfaceGraph,
    ReflectorSlopeVolume,
    FaultWarpingInput,
    FaultWarpingConfig,
    FaultWarpingResult,
    FaultWarpingEstimator,
)
```

The package root does not re-export these names. Cross-module transport uses
these typed contracts, not raw dictionaries, untyped tuples, or `Any`.

Array-holding contracts are frozen, slotted dataclasses with `eq=False`.
Construction validates but never casts, copies, normalizes, sorts, mutates, or
changes the writeability of caller-owned arrays. “Frozen” prevents field
rebinding only; it does not make caller-owned NumPy arrays read-only. Contract
violations fail fast with `TypeError` or `ValueError`.

The contract package imports only NumPy and the Python standard library.
SciPy, Numba, threadpoolctl, Atlas packages, and artifact/I/O dependencies are
not needed merely to import it.

## Arrays, coordinates, and orientation

A 3D volume has shape `(n3, n2, n1)` and is indexed as
`volume[i3, i2, i1]`. Surface coordinates use component order `(x1, x2, x3)`:

- `x1` is the sample/time/depth-index axis;
- `x2` and `x3` are the second and third spatial-index axes.

All coordinates are in the input volume's local index frame. A fractional
surface coordinate is valid when it lies in the closed local bounds
`0 <= x1 <= n1 - 1`, `0 <= x2 <= n2 - 1`, and `0 <= x3 <= n3 - 1`.
No survey azimuth, metre, millisecond, depth, or other physical-coordinate
system is implied.

Strike and dip follow `pyosv.geometry` grid/index-space conventions. Strike is
not a geographic azimuth. For this section only, let
`w = (w1, w2, w3)` denote the fault-normal vector returned by
`fault_normal_vector_from_strike_and_dip(strike_deg, dip_deg)`. This local
symbol must not be confused with the existing 3D-voting use of `w` for a
strike-axis vector.

## Side convention

The MVP side-sampling direction is the horizontal index-space component of the
fault normal:

```text
nh = normalize((0, w2, w3))
```

For surface-cell position `x` and positive side offset `h`:

```text
positive side = x + h * nh
negative side = x - h * nh
```

`positive` and `negative` express only this geometric sign. They do not imply
hanging wall, footwall, or a resolved fault sense. A cell with a zero or
numerically unstable `nh` is not estimable by a later solver.

## Apparent-shift convention

`tau` is an apparent sample-axis shift in samples defined by:

```text
positive_side(t) ≈ negative_side(t + tau)
```

A positive `tau` means that the matching negative-side event occurs at a
larger sample index than the positive-side event.

Use `apparent_shift_samples` or `shift_samples` in public names and docstrings;
do not call this quantity merely `slip`. It is not true fault slip, a physical
vertical throw, down-dip displacement, hanging-wall displacement, depth
conversion, strike-slip, or a full vector displacement.

## Reflector-slope convention

`ReflectorSlopeVolume` stores:

```text
p2 = dx1 / dx2
p3 = dx1 / dx3
```

Both have units of index samples per index-grid unit. Physical dip, degrees,
metres per trace, and implicit unit conversion are not accepted.

## Validity and support semantics

`valid_mask` is required and is shared by amplitude and both slope volumes.
At `valid_mask=True`, amplitude, `p2`, and `p3` must be finite. Values where
`valid_mask=False` are outside numerical processing and need not be finite.

Every result is aligned to input surface row order. On an invalid output row,
every floating diagnostic is `NaN` and every diagnostic boolean is `False`
(in particular, `boundary_hit=False`). A valid output row has finite floating
diagnostics. Low correlation and inability to estimate are distinct: a solver
may return a valid solution with low correlation.

## Typed contracts

### `FaultSurfaceGraph`

`FaultSurfaceGraph` represents one surface or surface patch. Its required
one-dimensional, row-aligned fields are:

```text
x1, x2, x3: float32
strike_deg, dip_deg: float32
ca_index, cb_index, cl_index, cr_index: int64
cell_support_weight: float32 | None
```

All required arrays have the same nonzero length. Floating arrays have exact
`float32` dtype; topology arrays have exact `int64` dtype. Coordinates,
strike, dip, and optional support weights are finite. `strike_deg` is in
`[0, 360)` and `dip_deg` is in `(0, 90]`.

Each topology value is either `-1` (no link) or a valid row index. Self-links
are prohibited. Links are reciprocal:

```text
ca_index[i] == j  iff  cb_index[j] == i
cl_index[i] == j  iff  cr_index[j] == i
```

The contract does not impose connectivity, branching, intersection, or
geological-validity rules. When absent, `cell_support_weight` means a generic
support weight of `1.0` for every row. When supplied, it is finite `float32`
in `[0, 1]`. It is not a DL probability, OSV score, or calibrated confidence.

### `ReflectorSlopeVolume`

`p2` and `p3` are exact-`float32`, three-dimensional arrays with identical
shape. This type does not duplicate a valid mask; finite-at-valid-voxel
validation occurs in `FaultWarpingInput`.

### `FaultWarpingInput`

```text
amplitude: float32 3D array
valid_mask: bool 3D array
surface: FaultSurfaceGraph
reflector_slopes: ReflectorSlopeVolume
```

All four volume arrays have the same shape. The amplitude and slopes are
finite at valid voxels, and all surface coordinates lie within the local volume
bounds. Reflector slopes are required; no omitted-slope or implicit-zero-slope
fallback exists.

### `FaultWarpingConfig`

```text
side_offset_grid: float
window_radius_samples: int
lag_min_samples: int
lag_max_samples: int
max_shift_strain: float
minimum_valid_fraction: float
similarity_metric: Literal["zncc"] = "zncc"
subsample_refinement: bool = False
```

Boolean values are never accepted as integers. All floating values are finite;
`side_offset_grid > 0`, `window_radius_samples >= 1`,
`lag_min_samples <= 0 <= lag_max_samples`,
`lag_min_samples < lag_max_samples`, `0 < max_shift_strain <= 1`, and
`0 < minimum_valid_fraction <= 1`. The only accepted similarity metric is
`"zncc"` and `subsample_refinement` is an exact boolean.

No operational defaults are supplied for side offset, window radius, lag
bounds, strain, or valid fraction; callers choose those scientific parameters
explicitly.

### `FaultWarpingResult`

All fields are one-dimensional and aligned to the input surface row order:

```text
valid: bool
shift_samples: float32
correlation_before: float32
correlation_after: float32
cost_margin: float32
cycle_residual_samples: float32
valid_sample_fraction: float32
boundary_hit: bool
```

All arrays have the same length. Valid rows have finite float values;
`correlation_before` and `correlation_after` are in `[-1, 1]`,
`cost_margin >= 0`, `cycle_residual_samples >= 0`, and
`valid_sample_fraction` is in `[0, 1]`. Invalid rows have `NaN` for every
float field and `boundary_hit=False`.

`correlation_gain` is a computed row-aligned `float32` property equal to
`correlation_after - correlation_before`; it preserves `NaN` on invalid rows.
A result contains no surface ID, Atlas run ID, path, DL score, OSV score, or
confidence class.

## Estimator protocol

```python
@runtime_checkable
class FaultWarpingEstimator(Protocol):
    def estimate(
        self,
        inputs: FaultWarpingInput,
        config: FaultWarpingConfig,
    ) -> FaultWarpingResult:
        ...
```

This protocol is the only estimator-facing API in v1. Later Python or Numba
backends may implement it without changing the public coordinate, validity, or
result semantics.

## NumPy support

PyOSV's supported NumPy line is NumPy 1.x, declared by the authoritative
dependency bound `numpy<2`. NumPy 2 support is not claimed by this contract.
The project-wide byte-level fixture policy remains documented in
[the refactoring non-regression contract](refactoring_contract.md).
