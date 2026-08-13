# Reference-First Equivalence Policy

`pyosv` targets reference-oriented semantic alignment for functionality mapped
to `reference_osv`. Bitwise equality with Java, Jython, or Mines JTK is not a
package requirement.

The required result is a deterministic Python implementation that preserves the
documented scientific meaning, geometry, stage responsibilities, and public API
contract. Numerical differences caused by interpolation kernels, smoothing
kernels, boundary treatment, and floating-point evaluation are accepted only
inside an explicitly documented equivalence boundary.

PyOSV-native functionality that has no corresponding `reference_osv` feature is
governed by its own public contract. It must not be described as
reference-equivalent.

## Required invariants

Reference-oriented implementations preserve the following unless a more
specific public contract states otherwise:

- global array shape and indexing conventions;
- geometric component order and angle conventions;
- processing-stage order and ownership;
- public parameter meanings and defaults;
- lag, strain, threshold, and normalization semantics;
- rounding, clamping, boundary inclusion, and tie-breaking rules;
- output shape, dtype, sentinel, and nonmutation contracts;
- deterministic ordering for seeds, candidates, cells, and artifacts.

Repository-wide array conventions are:

```text
2D arrays:        (n2, n1), indexed as array[i2, i1]
3D arrays:        (n3, n2, n1), indexed as array[i3, i2, i1]
2D components:    (x1, x2)
3D components:    (x1, x2, x3)
local 3D voting:  (w, v, u), with u normal, v dip, and w strike
```

A numerical optimization or alternate backend must preserve these invariants or
be exposed as a separate, explicitly named contract.

## Decision order

When implementation choices conflict, apply this order:

1. Preserve the current public Python contract.
2. Preserve the mapped reference control flow and geometric meaning.
3. Represent unavailable Mines JTK numerical operations with a documented
   Python numerical contract.
4. Keep materially different algorithms behind explicit backend, mode, policy,
   or workflow selections.
5. Evaluate quality-oriented behavior with evidence appropriate to its stated
   semantics.

A faster or more robust implementation does not become equivalent merely
because its final image appears similar. The implementation must preserve the
stage-specific inputs, outputs, and decisions named by its contract.

## Python numerical substitutions

PyOSV does not require a JVM, Jython, Gradle, or Mines JTK at runtime. Reference
numerical operations are represented with NumPy, SciPy, and optional Numba
kernels.

Typical substitutions include:

- JTK sinc interpolation represented by SciPy coordinate interpolation or a
  documented structured interpolation kernel;
- JTK recursive exponential or Gaussian filters represented by SciPy Gaussian
  or explicit separable smoothing;
- Java array kernels represented by deterministic Python or Numba kernels.

These substitutions may differ in interpolation response, recursive-filter
response, edge handling, floating-point accumulation order, and the last bits of
returned values. Each numerical call site must still define the applicable axes,
sigma or scale, extrapolation mode, rounding rule, and output dtype.

Optional acceleration is an implementation choice, not a distinct scientific
method. Accelerated and fallback paths share the same public shape, dtype,
validation, and semantic contracts.

## Independent configuration axes

The following choices are independent and must be recorded separately in
reports and artifacts:

- scanner backend;
- scanner-thinning mode;
- downstream workflow;
- surface-voting boundary policy;
- voter-thinning mode;
- skinning method and reskin policy.

The current default and explicit selections include:

| Area | Default or reference-oriented selection | Other explicit selections |
| --- | --- | --- |
| 2D scanner | `FaultOrientScanner2.scan()` | `scan_fast()` |
| 3D scanner | `FaultOrientScanner3.scan()` / `scan_reference_like()` | `scan_quality()`, `scan_fast()`, and the documented directional backend |
| 3D scanner thinning | `mode="reference"` | `mode="normal"` |
| Surface-voting boundary policy | `reference` | `masked_in_bounds` |
| 3D voter thinning | `mode="reference"` | `normal`, `normal_plateau`, `hybrid`, `hybrid_v2` |
| Skinning | `method="reference"` | `quality`, `connected_component` |

A shared name such as `reference` does not make two stages numerically
identical. Scanner reference thinning and voter reference thinning share a
strike-binned helper but have different reinforcement, edge-cleanup, and return
contracts. Detailed behavior belongs in the stage-specific documentation.

## Evidence semantics

Evidence types answer different questions and must remain separate.

### Focused regression tests

Unit and integration tests verify current Python contracts such as shape, dtype,
finiteness, value bounds, deterministic ordering, boundary behavior, and
specific intermediate decisions. Exact equality is appropriate when the Python
contract requires it, including agreement between Python and accelerated
kernels. Such equality does not establish Java or Mines JTK identity.

### Controlled synthetic evaluation

Synthetic cases provide known geometry and are the evidence source for
localization, orientation, thinning, skin topology, and other truth-based
metrics. Deterministic cases are one evaluation case, not statistical
replicates merely because they contain multiple slices or regions.

### F3 public-reference evaluation

The public F3 `fl.dat`, `fv.dat`, and `fvt.dat` volumes are comparison targets.
They are not independent geological truth or direct method-level Java fixtures.
Publication-facing F3 comparison treats the complete volume as one evaluation
unit. Crops, slices, regions, and tiles are diagnostics within that unit and are
not independent replicates.

### Optional reference reports

Optional reports compare Python outputs with external reference artifacts. They
measure agreement under the recorded configuration and numerical environment.
They do not add a runtime dependency on the reference implementation and do not
imply bitwise equivalence.

### Publication bundles

Publication validation establishes recorded provenance and file integrity for a
completed bundle. It does not replay the source experiments and does not by
itself establish numerical equivalence or scientific generalization.

## Practical-equivalence metrics

The shared comparison helpers are implemented in `pyosv.metrics`. They support
regression tests and diagnostic reports; they do not define one universal
acceptance score.

- `finite_value_report(x)` reports shape, size, finite and non-finite counts,
  finite fraction, and finite-only minimum, maximum, and mean.
- `normalized_correlation(a, b)` computes zero-mean normalized correlation for
  matching finite nonempty arrays. It returns `0.0` when either centered input
  has zero norm.
- `top_percentile_mask(...)` defines a percentile mask. With
  `positive_only=True`, values at or below `positive_epsilon` are excluded
  before the percentile is calculated.
- `top_percentile_overlap(...)` reports exact overlap statistics for two
  percentile masks.
- `buffered_ridge_overlap(...)` reports exact and radius-buffered precision,
  recall, F1, and Jaccard statistics for sparse ridge masks.
- `sparse_ridge_distance_metrics(...)` reports symmetric Euclidean
  distance-transform summaries. Distance values are `None` when either sparse
  mask is empty.
- `orientation_angle_error(...)` reports wrapped angular error for a specified
  period.
- `strike_dip_angle_error(...)` reports wrapped strike error and absolute dip
  error.
- `orientation_field_report(...)` summarizes strike and dip values on a
  high-likelihood mask.

Reports must record every parameter that changes a metric definition, including
percentile, positive-only selection, epsilon, buffer radius, angular period, and
mask source. Synthetic truth metrics and F3 public-reference metrics must not be
combined into one accuracy value.

## Acceptance criteria

A metric threshold is binding only when it is defined by a current automated
test, promotion specification, validation contract, or publication contract.
A diagnostic report without such a threshold records evidence but does not
create an acceptance criterion.

Default tests require deterministic Python behavior and the documented public
contracts. They do not require external Java/JTK data. Per-sample equality with
an external reference is required only when a feature-specific contract states
that requirement explicitly.

Byte-level fixtures protect a defined Python artifact contract. They are
non-regression fixtures, not evidence of Java/JTK equality.

## Reference data policy

`reference_osv/` is an external, read-only reference checkout or bind mount. It
is not committed, packaged, modified, or used as an output directory.

Reference paths resolve under `./reference_osv` by default. Set the environment
variable when the checkout is elsewhere:

```bash
export PYOSV_REFERENCE_OSV=/absolute/path/to/osv-master
```

Default tests skip reference-dependent cases when the reference root or required
files are absent.

Run the optional 2D voting reference report with:

```bash
PYOSV_REFERENCE_OSV=/absolute/path/to/osv-master \
PYOSV_RUN_SLOW_REFERENCE_VOTING=1 \
python -m pytest -q tests/test_voting2d_reference_smoke.py
```

Run the optional 2D scanner reference report with:

```bash
PYOSV_REFERENCE_OSV=/absolute/path/to/osv-master \
PYOSV_RUN_SLOW_REFERENCE_SCANNER=1 \
python -m pytest -q tests/test_orient2d_reference_report.py
```

## Documentation requirements

A stage-specific document that claims reference-oriented behavior must state:

- the mapped reference responsibility;
- the current Python API and defaults;
- the shape, coordinate, dtype, and boundary contract;
- the Python numerical substitute for unavailable reference kernels;
- any explicit alternate backend, mode, or policy;
- the evidence type used to assess agreement.

Documentation describes the current contract. Implementation history,
transition instructions, issue references, and experiment chronology do not
belong in the specification.

## Related specifications

- [2D Orientation Scanning](orient2d.md)
- [2D Voter Reference Mapping](reference_mapping_voting2d.md)
- [3D Reference Alignment](reference_alignment_3d.md)
- [3D Orientation Scanning](orient3d.md)
- [3D Scanner Reference Mapping](reference_mapping_orient3d.md)
- [3D Voting Conventions](3d_voting.md)
- [3D Voter Reference Mapping](reference_mapping_voting3d.md)
- [Reference-Like 3D Thinning](reference_like_thinning.md)
- [Skinning](skinning.md)
- [Mode Comparison Contract](mode_comparison.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [F3 3D Reference Data Validation](f3d_validation.md)
