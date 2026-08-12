# 3D Reference Alignment

This document defines the current audit boundary for aligning the 3D scanner
and optimal-surface voter with the read-only Java reference implementation.
Detailed method mappings are maintained in the scanner and voter mapping
documents linked below.

`pyosv` follows a reference-first policy for fault-interpretation workflows.
Bit-exact reproduction of Java, Jython, or Mines JTK output is not required.
The implementation must preserve the documented geometry, control flow, shape,
dtype, and stage semantics while using Python, NumPy, SciPy, and optional Numba.

## Reference and evidence boundary

`reference_osv/` is an external, read-only reference implementation. Do not
modify it, commit it, include it in package distributions, or write generated
outputs under it. Generated reports, figures, and `.dat` files belong under
`outputs/` or another ignored working directory.

The public F3 workflow starts from `ep.dat` and compares Python outputs with the
public `fl.dat`, `fv.dat`, and `fvt.dat` volumes. Publication-facing F3
comparison uses the complete `(420, 400, 100)` volume as one evaluation unit.
The public volumes are comparison targets, not independent geological truth,
and crops are not publication samples or statistical replicates. The complete
protocol is defined in [F3 3D Reference Data Validation](f3d_validation.md).

Controlled synthetic evaluation uses generated truth geometry and is the source
for known-truth localization, orientation, thinning, and skinning metrics. F3
reference agreement and synthetic truth quality are separate evidence types and
must not be combined into one accuracy claim.

## Alignment boundaries

Audit these stages independently:

```text
input -> scanner -> scanner thinning -> voting -> voter thinning -> skinning
```

Scanner backend, scanner thinning, workflow mode, and voter thinning are
separate configuration axes. Workflow mode does not select the scanner backend
or scanner thinning, and scanner reference thinning is distinct from voter
reference thinning.

The final `fvt` result can change because of:

- scanner likelihood and orientation selection;
- scanner-thinning policy and edge cleanup;
- seed selection and local coordinate sampling;
- dynamic-programming accumulation, backtracking, and smoothing;
- surface-vote accumulation and orientation updates;
- workflow-owned voting and skinning settings;
- voter-thinning policy;
- final vote-map normalization.

Change and evaluate one stage-level behavior at a time. Multi-stage changes
make numerical attribution ambiguous and can conceal regressions.

`FaultOrientScanner3.thin(...)` and `OptimalSurfaceVoter.thin(...)` default to
reference-like strike-binned thinning. Explicit `mode="normal"` selects
fault-normal thinning. Scanner reference thinning applies scanner edge cleanup;
voter reference thinning uses voter-specific reinforcement and does not apply
that cleanup.

`OptimalSurfaceVoter.apply_voting()` performs final vote normalization without
final vote-map smoothing by default: subtract the global minimum, divide by the
positive global maximum, and apply `1 - (1 - x) ** 8`.
`set_final_normalization_smoothing(sigma)` explicitly enables smoothing before
that normalization.

## Method-level audit procedure

1. Select one Java method or helper and one Python implementation unit.
2. Record input and output shapes, dtype, coordinate order, angle convention,
   boundary behavior, and mutable state.
3. Build a small synthetic fixture that exercises the method without F3 data.
4. Compare intermediate arrays or scalar decisions, not only the final F3
   output.
5. Classify each difference using the categories below.
6. Add focused regression coverage for the selected behavior.
7. Keep the implementation change scoped to that method or helper.
8. Run the default tests and formatting checks; use external F3 commands only
   when the audited behavior requires reference-data evidence.

Default checks:

```bash
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
```

## Difference categories

| Category | Audit questions |
| --- | --- |
| Coordinate convention | Are arrays indexed as `(n3, n2, n1)` while vectors use `(x1, x2, x3)` component order? Are local samples consistently represented as `(w, v, u)`? |
| Angle convention | Are strike `phi` and dip `theta` ranges, wrapping, binning, and vector formulas consistent with their consumers? |
| Interpolation | Which JTK interpolation use site is represented by SciPy? Are coordinate order, rounding, extrapolation, and boundary modes explicit? |
| Smoothing and filtering | Which recursive reference filter is approximated? Are sigma, axis order, kernel behavior, and edge handling explicit? |
| Rounding | Does the path use Java-style nearest rounding, flooring, clamping, or casting? Are ties deterministic? |
| Local UVW sampling | Are seed-centered box dimensions, lag offsets, admissible ranges, and inclusion rules preserved? |
| Dynamic programming | Are accumulation direction, strain constraints, lag ranges, smoothing, and backtracking isolated from voting accumulation? |
| Thinning | Is suppression performed along a fault normal or through strike-binned `i2-i3` neighborhoods? Are retained values taken from the intended source array? |
| Normalization | Are minimum subtraction, maximum scaling, zero-range behavior, clipping, dtype conversion, and optional smoothing applied at the specified stage? |

## Detailed mapping documents

- [3D Scanner Reference Mapping](reference_mapping_orient3d.md) maps
  `FaultOrientScanner3.java`, scanner-local interpolation and filtering, sampling,
  geometry conversion, normalization, and thinning to the current Python
  implementation.
- [3D Voter Reference Mapping](reference_mapping_voting3d.md) maps
  `OptimalSurfaceVoter.java`, `FaultCell`, local UVW sampling, dynamic
  programming, vote accumulation, normalization, and voter thinning to the
  current Python implementation.

Related specifications:

- [Reference-First Equivalence Policy](equivalence_policy.md)
- [3D Orientation Scanning](orient3d.md)
- [3D Voting Conventions](3d_voting.md)
- [Reference-Like 3D Thinning](reference_like_thinning.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [F3 3D Reference Data Validation](f3d_validation.md)
