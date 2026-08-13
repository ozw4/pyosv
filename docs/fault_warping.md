# Fault-Warping Contract

`pyosv.fault_warping` defines Atlas-independent typed contracts for estimating
an apparent sample-axis shift from seismic signals on opposite sides of a known
fault surface.

The package defines inputs, configuration, results, validation, and an estimator
protocol. It does not provide a concrete estimator or execute side sampling,
interpolation, similarity calculation, cost-volume construction, dynamic
warping, graph regularization, cycle calculation, sub-sample refinement, or
shift estimation.

## Contract identity and public surface

```python
FAULT_WARPING_CONTRACT_VERSION = "pyosv.fault_warping.v1"
```

The version identifies the Python API and its numerical semantics. It is not a
file format, serialization format, manifest schema, or artifact version.

The public import surface is:

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

The package root `pyosv` does not re-export these names.
`pyosv.fault_warping` does not expose an
`estimate_fault_apparent_shifts()` function, a built-in estimator, a no-op
estimator, a zero-shift fallback, or an entry point that raises
`NotImplementedError`. Callers supply an object that implements
`FaultWarpingEstimator`.

## Ownership boundary

The package owns these numerical contracts:

- a 3D seismic-amplitude volume and a volume-wide validity mask;
- row-aligned coordinates, strike, dip, topology, and optional support weights
  for a known fault-surface graph;
- positive-side and negative-side geometry derived from a fault normal;
- reflector slopes and explicit lag-search configuration;
- row-aligned apparent shifts and numerical diagnostics;
- NumPy-array validation; and
- the estimator protocol shared by conforming implementations.

The package does not discover faults, run inference, perform OSV scanning or
voting, construct skins, or adapt `FaultSkin` objects. `FaultSurfaceGraph` is the
public surface transport type; a `FaultSkin` adapter is not part of this API.

The package also does not own:

- `seis_atlas` or `seis_fault_workflow` objects and execution;
- paths, files, checksums, manifests, artifact writers, or publication;
- DL probabilities, OSV-score fusion, or confidence calibration;
- accepted/review/rejected classifications or pseudo-label generation;
- PNG, HTML, CSV, or NPZ output;
- a CLI or viewer integration;
- TWT-to-depth, metre, or other physical-unit conversion; or
- true fault slip, vertical throw, down-dip displacement, strike-slip,
  hanging-wall displacement, footwall displacement, or a full displacement
  vector.

Fault warping is a PyOSV-native contract and has no corresponding
`reference_osv` equivalence claim.

## Construction and dependency contract

Array-holding contracts are frozen, slotted dataclasses with value-based
dataclass equality disabled. Construction validates caller-owned arrays but
does not cast, copy, normalize, sort, mutate, or change their writeability.

`frozen=True` prevents field rebinding. It does not make referenced NumPy arrays
read-only. Callers remain responsible for preserving the validated array
contract after construction.

Array fields require `numpy.ndarray` objects with exact dtypes and dimensions.
Contract violations raise `TypeError` for incompatible Python or dtype
categories and `ValueError` for invalid dimensions, lengths, values, ranges, or
relationships.

Importing `pyosv.fault_warping` requires only NumPy and the Python standard
library. The package does not import SciPy, Numba, threadpoolctl, Matplotlib,
Pandas, PyTorch, Segyio, Atlas packages, or artifact/I/O frameworks.

## Arrays, coordinates, and orientation

A global volume has shape `(n3, n2, n1)` and is indexed as:

```text
volume[i3, i2, i1]
```

Surface coordinates use component order `(x1, x2, x3)`:

- `x1` is the sample, time-index, or depth-index axis;
- `x2` and `x3` are the second and third spatial-index axes.

All coordinates are expressed in the input volume's local index frame. A
fractional coordinate is within the volume when:

```text
0 <= x1 <= n1 - 1
0 <= x2 <= n2 - 1
0 <= x3 <= n3 - 1
```

No survey azimuth, metre, millisecond, depth, or other physical-coordinate
system is implied.

Strike and dip follow `pyosv.geometry` grid/index-space conventions. Strike is
not a geographic azimuth. In the side convention below, let
`w = (w1, w2, w3)` denote the fault normal returned by
`fault_normal_vector_from_strike_and_dip(strike_deg, dip_deg)`. This local
symbol is distinct from the 3D-voting convention in which `w` denotes the
fault-strike axis.

## Side convention

The side-sampling direction is the normalized horizontal index-space component
of the fault normal:

```text
nh = normalize((0, w2, w3))
```

For surface position `x` and positive offset `h`:

```text
positive side = x + h * nh
negative side = x - h * nh
```

`positive` and `negative` identify only this geometric sign. They do not imply
hanging wall, footwall, or fault sense. An estimator must treat a row as
non-estimable when `nh` is zero or cannot be normalized stably.

## Apparent-shift convention

`tau` is an apparent sample-axis shift in samples:

```text
positive_side(t) approximately equals negative_side(t + tau)
```

A positive `tau` means that the matching negative-side event occurs at a larger
sample index than the positive-side event.

Public names use `apparent_shift_samples` or `shift_samples`, not an unqualified
`slip`. This quantity is not true fault slip, physical vertical throw,
down-dip displacement, hanging-wall displacement, depth conversion,
strike-slip, or a full displacement vector.

## Reflector-slope convention

`ReflectorSlopeVolume` stores:

```text
p2 = dx1 / dx2
p3 = dx1 / dx3
```

Both values are measured in index samples per index-grid unit. They are not
physical dip angles, metres per trace, or implicitly converted physical units.

## Validity and row alignment

`valid_mask` is required and is shared by amplitude, `p2`, and `p3`. At voxels
where `valid_mask` is true, all three floating volumes must be finite. Values at
false mask voxels are outside numerical processing and may be non-finite.

Every estimator result is aligned to the input surface row order. A valid result
row has finite floating diagnostics. An invalid row has:

```text
valid = False
all floating result fields = NaN
boundary_hit = False
```

Low correlation and inability to estimate are distinct. A valid estimate may
have low correlation when all other validity requirements are satisfied.

## Typed contracts

### `FaultSurfaceGraph`

`FaultSurfaceGraph` represents one nonempty surface or surface patch. Its fields
are one-dimensional and row-aligned:

```text
x1, x2, x3: float32
strike_deg, dip_deg: float32
ca_index, cb_index, cl_index, cr_index: int64
cell_support_weight: float32 | None
```

All required arrays have the same positive length. Coordinate and angle arrays
must be finite. Angle ranges are:

```text
0 <= strike_deg < 360
0 < dip_deg <= 90
```

Each topology value is either `-1` for no link or a valid surface row index.
Self-links are prohibited. Link pairs are reciprocal:

```text
ca_index[i] == j  iff  cb_index[j] == i
cl_index[i] == j  iff  cr_index[j] == i
```

The graph contract does not require one connected component and does not impose
branching, intersection, manifold, or geological-validity rules.

When supplied, `cell_support_weight` has the same length as the surface arrays,
uses exact `float32`, is finite, and lies in `[0, 1]`. When omitted, its semantic
value is a generic weight of `1.0` for each row. It is not a DL probability, an
OSV score, or calibrated confidence.

`FaultSurfaceGraph` validates finite coordinates but does not know a volume
shape. Coordinate bounds are checked by `FaultWarpingInput`.

### `ReflectorSlopeVolume`

```text
p2: float32, shape (n3, n2, n1)
p3: float32, shape (n3, n2, n1)
```

The two arrays must have identical shapes. This type does not contain a validity
mask and does not require every voxel to be finite by itself. Finite-at-valid-
voxel checks belong to `FaultWarpingInput`.

### `FaultWarpingInput`

```text
amplitude: float32, shape (n3, n2, n1)
valid_mask: bool, shape (n3, n2, n1)
surface: FaultSurfaceGraph
reflector_slopes: ReflectorSlopeVolume
```

Amplitude, mask, `p2`, and `p3` must have the same shape. Amplitude and slopes
must be finite wherever the mask is true. Every surface coordinate must lie
inside the closed local volume bounds.

Reflector slopes are mandatory. The contract has no omitted-slope or
implicit-zero-slope fallback.

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

The six scientific search parameters before `similarity_metric` have no
defaults. Their constraints are:

```text
side_offset_grid > 0
window_radius_samples >= 1
lag_min_samples <= 0 <= lag_max_samples
lag_min_samples < lag_max_samples
0 < max_shift_strain <= 1
0 < minimum_valid_fraction <= 1
```

Boolean values are not accepted as integer or real parameters. All real-valued
parameters must be finite. `similarity_metric` accepts only `"zncc"`.
`subsample_refinement` must be an exact built-in `bool`.

### `FaultWarpingResult`

All fields are one-dimensional and have the same length:

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

For valid rows:

```text
all floating fields are finite
-1 <= correlation_before <= 1
-1 <= correlation_after <= 1
cost_margin >= 0
cycle_residual_samples >= 0
0 <= valid_sample_fraction <= 1
```

For invalid rows, every floating field is `NaN` and `boundary_hit` is false.

`correlation_gain` is a computed row-aligned `float32` property:

```text
correlation_gain = correlation_after - correlation_before
```

It preserves `NaN` on invalid rows. A result does not carry a surface ID, Atlas
run ID, path, DL score, OSV score, or confidence class.

## Estimator protocol

`FaultWarpingEstimator` is a runtime-checkable structural protocol:

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

A conforming estimator preserves the coordinate, side, sign, validity, row
alignment, and result semantics defined here. Python, Numba-backed, or other
implementations use the same protocol; implementation technology does not
change the public scientific contract.

## Verification and supported runtime

The package tests verify:

- the exact public export surface and contract version;
- frozen, slotted, `eq=False` dataclass structure;
- exact array dtypes, dimensions, lengths, finiteness, ranges, and reciprocal
  topology;
- caller-array identity, contents, and writeability preservation;
- valid-mask and coordinate-bound semantics;
- configuration validation and exact boolean handling;
- valid and invalid result-row contracts;
- `correlation_gain` semantics;
- the runtime-checkable estimator protocol and method signature; and
- the absence of forbidden import dependencies.

The supported dependency line is declared as `numpy<2`. NumPy 2 behavior is
outside this contract. The byte-level fixture policy is defined in
[Refactoring Non-Regression Contract](refactoring_contract.md).

## Related specifications

- [Architecture](architecture.md)
- [Skinning](skinning.md)
- [Reference-First Equivalence Policy](equivalence_policy.md)
- [Refactoring Non-Regression Contract](refactoring_contract.md)
