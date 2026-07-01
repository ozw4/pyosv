# 3D Reference Alignment

This document is the entry point for aligning the 3D scanner and
optimal-surface voter with the read-only Java reference implementation. It
organizes the audit work before changing scanner or voter logic.

`pyosv` targets practical agreement for fault interpretation workflows, not
bit-exact reproduction of Java, Jython, Mines JTK, or Gradle behavior. Alignment
work should explain observed differences, isolate their source, and add focused
regression coverage before implementation changes are made.

## Current F3 Status

The public F3 3D validation workflow starts from `ep.dat` and compares Python
outputs with existing reference volumes:

- `fl.dat`: reference fault likelihood.
- `fv.dat`: reference OSV fault volume.
- `fvt.dat`: reference thinned OSV fault volume.

F3 data is external and optional. Normal tests must not require the F3 data root
or the `reference_osv/` bind mount. Use `docs/f3d_validation.md` for optional
smoke, crop, multi-crop, ablation, and full-volume commands.

Current comparisons are report-oriented. The previous current/current baseline
context documented for thinned `fvt` is:

```text
normalized_correlation.interior.fvt.mean ~= 0.224
buffered_ridge_overlap.interior.fvt.buffered_f1.mean ~= 0.075
exact fvt ridge overlap F1/Jaccard = 0.0
```

These numbers are context for investigation, not acceptance thresholds. Do not
claim equivalence or tune parameters until an actual report has been generated
and reviewed for the specific change under audit.

## Alignment Boundaries

`reference_osv/` is a read-only external reference implementation. Do not modify
it, write generated outputs under it, or commit it. Generated reports, figures,
and `.dat` files belong under `outputs/` or another ignored working directory.

Separate scanner, thinning, and voting differences before changing behavior.
The same final `fvt` metric can move because of:

- scanner likelihood and angle choices;
- scanner thinning choices;
- voter seed selection, local sampling, dynamic programming, or accumulation;
- voter thinning choices;
- normalization and post-processing differences.

Changing more than one stage at a time makes F3 metrics hard to interpret and
can hide regressions in synthetic tests. Prefer method-level parity tests and
small synthetic fixtures before changing scanner or voter logic.

`scanner_thin_mode=reference` is not automatically adopted as the default. It
is a diagnostic mode that is closer to the Java strike-binned thinning pattern,
but it remains a Pythonic approximation with SciPy smoothing and repository
shape conventions. The default `normal` mode preserves existing `pyosv`
behavior, tests, and user expectations until a later issue explicitly changes
the public contract.

This phase does not tune parameters to chase F3 metrics. Parameter changes are
allowed only when they follow from an audited method difference and come with
focused tests and updated documentation for any public API or default change.

## Method-Level Audit Workflow

1. Pick one reference method or helper and one Python function to audit.
2. Record the expected input and output shapes, dtype, coordinate order, and
   angle convention.
3. Build a small synthetic fixture that exercises the method without F3 data.
4. Compare intermediate arrays or scalar decisions, not only final F3 outputs.
5. Classify every difference using the categories below.
6. Add a method-level parity or regression test for the chosen behavior.
7. Only then change scanner or voter logic, keeping the edit scoped to the
   audited method.
8. Run default tests and formatting checks. Run optional F3 commands only for
   report generation or manual validation.

Useful default checks:

```bash
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
```

Useful optional F3 commands are documented in `docs/f3d_validation.md`, including
the external data smoke test, small crop validation, reference-like thinning
validation, thinning ablation report, large crop validation, and full F3 run.

## Difference Categories

| Category | Audit questions |
| --- | --- |
| Coordinate convention | Are arrays indexed as `(n3, n2, n1)` while vectors keep `(x1, x2, x3)` component order? Are local samples using `(w, v, u)` consistently? |
| Angle convention | Are strike `phi` and dip `theta` ranges, wrapping, binning, and vector formulas consistent with the consumer of the arrays? |
| Interpolation | Is a JTK `SincInterpolator` use site approximated with SciPy interpolation? Are coordinate order and boundary modes explicit? |
| Smoothing/filtering | Is a JTK recursive filter approximated with Gaussian or separable smoothing? Are sigma, axis order, and boundary handling documented? |
| Rounding | Does the Java path round, floor, clamp, or cast indices differently from Python/NumPy? Are tie-breaks deterministic? |
| Local UVW sampling | Are seed-centered local boxes, axis lengths, and lag offsets sampled in the same order and with the same inclusion rules? |
| Dynamic programming | Are accumulation direction, strain limits, lag ranges, smoothing, and backtracking rules isolated from voting accumulation? |
| Thinning | Is the comparison along the fault normal or in strike-binned `i2-i3` neighborhoods? Are retained values copied from smoothed or original arrays? |
| Normalization | Are max scaling, zero-volume behavior, clipping, and dtype conversion applied at the same stage? |

## Follow-Up Documents

| Follow-up | Purpose | Status |
| --- | --- | --- |
| Scanner mapping | Map `FaultOrientScanner3.java` methods to `src/pyosv/orient3d.py` functions and document intentional approximations in [reference_mapping_orient3d.md](reference_mapping_orient3d.md). | Added |
| Voter mapping | Map `OptimalSurfaceVoter.java` methods to `src/pyosv/voting3d.py`, `src/pyosv/dp.py`, and related helpers in [reference_mapping_voting3d.md](reference_mapping_voting3d.md). | Added |
| Parity tests | Define method-level parity fixtures that do not require F3 data and cover scanner, UVW sampling, DP, voting, and thinning behavior. | Planned |
| Reference-like scanner skeleton | Sketch any opt-in reference-like scanner path only after mapping and parity tests identify a concrete need. | Planned |

Related existing documents:

- `docs/equivalence_policy.md`
- `docs/f3d_validation.md`
- `docs/orient3d.md`
- `docs/3d_voting.md`
- `docs/reference_mapping_orient3d.md`
- `docs/reference_mapping_voting3d.md`
- `docs/reference_like_thinning.md`
- `docs/reference_mapping.md`
