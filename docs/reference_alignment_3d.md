# 3D Reference Alignment

This document is the entry point for aligning the 3D scanner and
optimal-surface voter with the read-only Java reference implementation. It
organizes the audit work before changing scanner or voter logic.

`pyosv` follows a reference-first policy for fault interpretation workflows
while avoiding bit-exact reproduction requirements for Java, Jython, Mines JTK,
or Gradle behavior. Alignment work should explain observed differences, isolate
their source, and add focused regression coverage before implementation changes
are made.

## Current F3 Status

The public F3 3D validation workflow starts from `ep.dat` and compares Python
outputs with existing reference volumes:

- `fl.dat`: reference fault likelihood.
- `fv.dat`: reference OSV fault volume.
- `fvt.dat`: reference thinned OSV fault volume.

F3 data is external and optional. Normal tests must not require the F3 data root
or the `reference_osv/` bind mount. Publication-facing comparison uses the
complete `(420, 400, 100)` volume as one evaluation unit; crops are not
publication samples or statistical replicates. Start with the full-volume
protocol in [F3 3D Reference Data Validation](f3d_validation.md). Its current
`run_3d_f3d_full.py` runner is a single reference-like baseline scan/vote path,
not the planned full-volume 2×2 scanner-backend/workflow runner. Smoke, crop,
multi-crop, and ablation commands remain optional legacy/internal diagnostics.

Controlled synthetic reports are a separate truth-quality diagnostic documented
in `docs/synthetic_quality.md`. Oracle synthetic mode isolates the downstream
voting, thinning, and skinning stages from controlled truth attributes, while
scanner-inclusive mode evaluates the scanner plus those downstream stages
against the same known truth geometry. Use these reports to localize behavior
before interpreting F3 agreement changes; they are not Java-reference parity
claims.

Current comparisons are report-oriented. The previous normal/normal baseline
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

Audit scanner backend, scanner thinning, workflow mode, and voter thinning as
separate stages before changing behavior. Workflow mode does not select the
scanner backend or scanner thinning, and scanner reference thinning is distinct
from voter reference thinning. The same final `fvt` metric can move because of:

- scanner-backend likelihood and angle choices;
- scanner-thinning choices;
- downstream workflow defaults;
- voter seed selection, local sampling, dynamic programming, or accumulation;
- voter-thinning choices;
- final vote-map normalization and post-processing differences.

Changing more than one stage at a time makes F3 metrics hard to interpret and
can hide regressions in synthetic tests. Prefer method-level parity tests and
small synthetic fixtures before changing scanner or voter logic.

`scanner_thin_mode=reference` and `voter_thin_mode=reference` are the current
reference-first defaults. They are closer to the Java strike-binned thinning
pattern, but remain Pythonic approximations with SciPy smoothing and repository
shape conventions. The older fault-normal paths remain available only through
explicit `mode="normal"` API calls or `--scanner-thin-mode normal` /
`--voter-thin-mode normal` report flags for legacy comparisons.

`OptimalSurfaceVoter.apply_voting()` uses reference-style final normalization by
default: no final vote-map smoothing, then min subtraction, max scaling when
`max > 0`, and the `1 - (1 - x) ** 8` transform. The optional
`set_final_normalization_smoothing(sigma)` / `--final-normalization-smoothing`
path smooths the accumulated vote map before that normalization and is only for
older pyosv-style practical comparisons.

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

The full-volume F3 protocol and current baseline runner are documented first in
`docs/f3d_validation.md`. That document also preserves optional external-data
smoke, small-crop, multi-crop, reference-like-thinning, ablation, and large-crop
diagnostics.

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
| Normalization | Are min subtraction, max scaling when `max > 0`, zero-volume behavior, clipping, dtype conversion, and optional final vote-map smoothing applied at the same stage? |

## Follow-Up Documents

| Follow-up | Purpose | Status |
| --- | --- | --- |
| Scanner mapping | Map `FaultOrientScanner3.java` methods to `src/pyosv/orient3d.py` functions and document intentional approximations in [reference_mapping_orient3d.md](reference_mapping_orient3d.md). | Added |
| Voter mapping | Map `OptimalSurfaceVoter.java` methods to `src/pyosv/voting3d.py`, `src/pyosv/dp.py`, and related helpers in [reference_mapping_voting3d.md](reference_mapping_voting3d.md). | Added |
| Parity tests | Define method-level parity fixtures that do not require F3 data and cover scanner, UVW sampling, DP, voting, and thinning behavior. | Planned |
| Reference-like scanner follow-up | Refine the current default reference-like scanner with scanner-only reports, rotated-boundary checks, sigma mapping, and interpolation audits. | Planned |

Related existing documents:

- `docs/equivalence_policy.md`
- `docs/f3d_validation.md`
- `docs/orient3d.md`
- `docs/3d_voting.md`
- `docs/reference_mapping_orient3d.md`
- `docs/reference_mapping_voting3d.md`
- `docs/reference_like_thinning.md`
- `docs/reference_mapping.md`
