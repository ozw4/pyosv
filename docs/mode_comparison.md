# Scanner, Workflow, Thinning, and F3 Reference Comparison

This document defines the canonical terminology for public synthetic-data and
F3 full-volume comparisons. It separates configuration choices from comparison
targets; none of the terms below are interchangeable.

## 1. Scope and terminology

1. **Scanner backend** selects how fault likelihood, strike, and dip are
   scanned. Its report/config label is `scanner_backend`; the comparison values
   are `reference-like` and `quality`. The public API entry points are
   `FaultOrientScanner3.scan()`, `scan_reference_like()`, and `scan_quality()`.
2. **Workflow mode** selects downstream defaults for voting, voter thinning,
   skinning, and diagnostics. Its report/config label is `workflow_mode`, with
   values `reference`, `quality`, and `diagnostic`.
3. **Thinning mode** is a stage-specific choice. `scanner_thin_mode` controls
   scanner-output thinning, while `voter_thin_mode` controls vote-volume
   thinning. The value `reference` therefore names two distinct operations.
4. **Reference target** means the public F3 `fl.dat`, `fv.dat`, and `fvt.dat`
   outputs used for agreement comparisons. It is a comparison target, not a
   mode and not independent geological ground truth.

Do not use the phrases “reference mode” or “quality mode” without a qualifier.
Use, for example, “reference workflow,” “quality scanner backend,” “scanner
reference thinning,” or “F3 public reference target.” Python API names use
underscores, such as `scan_reference_like()`. Existing report values retain
their spelling, including the hyphen in `reference-like`.

The `fast` scanner backend and the report-local `ensemble` scanner backend also
exist, but they are diagnostics outside the primary public comparison defined
here.

## 2. Canonical naming table

| Concept | Canonical label | Current code/report value | Implementation entry point | What it changes | What it does not change |
| --- | --- | --- | --- | --- | --- |
| Reference-like scanner backend | reference-like scanner backend | `scanner_backend=reference-like` | `FaultOrientScanner3.scan()` / `scan_reference_like()` | The angle samples and scanner execution that produce `ft`, `pt`, and `tt` | Workflow defaults, either thinning stage, or the F3 target |
| Quality scanner backend | quality scanner backend | `scanner_backend=quality` | `FaultOrientScanner3.scan_quality()` | Refines the reference-like angle grid and can return scanner confidence | The scoring path, downstream workflow, either thinning stage, or guaranteed accuracy |
| Reference workflow | reference workflow | `workflow_mode=reference` | Synthetic workflow profile resolution | Downstream voter-thinning and skinning defaults | Scanner backend and scanner thinning |
| Quality workflow | quality workflow | `workflow_mode=quality` | Synthetic workflow profile resolution | Downstream `hybrid_v2` voter thinning and quality-skinner defaults | Scanner backend and scanner thinning |
| Diagnostic workflow | diagnostic workflow | `workflow_mode=diagnostic` | Synthetic workflow profile resolution | Reference-workflow defaults plus thinning diagnostics | Scanner backend, scanner thinning, or a third production algorithm |
| Scanner reference thinning | scanner reference thinning | `scanner_thin_mode=reference` | `FaultOrientScanner3.thin(..., mode="reference")` | Non-maximum suppression of scanner attributes before voting | Voter thinning or workflow selection |
| Voter reference thinning | voter reference thinning | `voter_thin_mode=reference` | `OptimalSurfaceVoter.thin(..., mode="reference")` | Non-maximum suppression of the vote volume | Scanner thinning or workflow selection |
| F3 public reference outputs | F3 public reference target | `fl.dat`, `fv.dat`, `fvt.dat` | F3 data loading and comparison metrics | The external outputs against which agreement is measured | Any algorithm setting or independent accuracy truth |

In Python, `SyntheticScannerConfig.backend` stores the scanner backend within
the nested scanner configuration, while public comparison reports and CLI
configuration identify the concept as `scanner_backend`.

## 3. Scanner implementation differences

`FaultOrientScanner3.scan()` is the current default reference-like scanner
path and delegates directly to `scan_reference_like()`. The reference-like
scanner backend uses a Java-inspired strike grid beginning at 0 degrees, with
20-degree spacing and 18 samples before clipping to the requested range. A
narrow requested range containing no fixed-grid sample falls back to its lower
endpoint. Its dip grid preserves the requested endpoints and uses approximately
5-degree spacing.

`scan_quality()` uses the same reference-like scoring path. It subdivides every
interval in the base reference-like strike and dip grids by
`refinement_factor`; it does not replace the score with another algorithm. The
factor must be an integer from 1 through 4 and defaults to 2. Factor 1 is
equivalent to a reference-like scan when the backend and all other options are
the same. With `return_confidence=True`, the quality scanner backend also
returns a normalized confidence map derived from the gap between the best and
second-best sampled orientation scores.

The name `quality` alone is not evidence of higher accuracy. Accuracy must be
measured on controlled synthetic truth and practical behavior must be reviewed
on real data. `scan_fast()` remains an explicit legacy derivative-bank scanner
backend, and the synthetic report has a diagnostic `ensemble` scanner backend;
neither is a primary axis of the public comparison matrix in this document.
See [3D Orientation Scanning](orient3d.md) for the scanner algorithm and API
details.

## 4. Workflow implementation differences

The following table gives the effective defaults resolved by the synthetic
workflow profiles and `SyntheticSkinningConfig`:

| Setting | Reference workflow | Quality workflow | Diagnostic workflow |
| --- | --- | --- | --- |
| Voter thinning | `reference` | `hybrid_v2` | `reference` |
| Skinner method | `reference` | `quality` | `reference` |
| Skinner minimum likelihood | `0.5` | `None` / adaptive | `0.5` |
| Growth source | `thinned` | `pre_thin` | `thinned` |
| Accepted occupancy radius | `None` (effective `5` in report) | `1` | `None` (effective `5` in report) |
| Boundary fallback | `false` | `true`, policy `empty_primary` | `false` |
| Thinning diagnostics | off | off | on |

Explicit CLI or configuration values override workflow defaults. Workflow mode
does not select a scanner backend and does not select `scanner_thin_mode`.
Conversely, choosing the `quality` scanner backend does not implicitly choose
the quality workflow. These choices must be recorded separately.

For boundary fallback, omitting both CLI controls retains these workflow
defaults. `--skinner-boundary-fallback` explicitly enables the configured
fallback, while `--no-skinner-boundary-fallback` explicitly disables it and
therefore overrides the quality-workflow default. The configured fallback
policy is preserved in either case.

The diagnostic workflow is the reference workflow with thinning diagnostics
enabled. It is not a third production-quality algorithm. The quality workflow
uses adaptive skin likelihood when no explicit minimum is supplied, grows from
the pre-thin vote volume, uses an accepted-occupancy radius of 1, and enables
the `empty_primary` boundary fallback. Further workflow and benchmark details
remain in [Quality Mode](quality_mode.md) and
[Controlled Synthetic Quality](synthetic_quality.md).

## 5. Comparison matrix and labels

Public figures and tables use this canonical 2×2 matrix:

| Short label | Scanner backend | Workflow mode |
| --- | --- | --- |
| `RL-REF` | `reference-like` | `reference` |
| `RL-QUAL` | `reference-like` | `quality` |
| `Q-REF` | `quality` | `reference` |
| `Q-QUAL` | `quality` | `quality` |

The short labels are figure/table conveniences only. They do not rename fields
or values in the report schema. Synthetic reports can currently specify the
scanner backend and workflow mode independently, so those dimensions can be
evaluated as separate configuration axes. The corresponding F3 full-volume
2×2 runner is planned work, as described in section 7.

When presenting results, expand the two axes in a caption or table header; do
not infer either axis from a bare `reference` or `quality` value.

## 6. Synthetic and F3 evaluation roles

Synthetic data provides known truth. It can therefore evaluate scanner
orientation and likelihood accuracy, downstream vote/skin recovery, and
topology. These controlled tests are the source for accuracy claims.

For publication comparisons on F3, the evaluation unit is only the full
`(420, 400, 100)` volume. Existing F3 crop, large-crop, and multi-crop paths are
legacy or internal diagnostics; crops are not publication samples, independent
replicates, or repeated experiments. The public F3 `fl.dat`, `fv.dat`, and
`fvt.dat` volumes support reference-agreement measurements, but they are not
independent geological ground truth and must not be labeled as accuracy truth.

If a future implementation uses internal chunking to make full-volume
execution practical, it must reconstruct results in full-volume coordinates
before computing metrics or producing figures. Chunk boundaries or overlapping
chunks must not become evaluation samples. See
[F3 3D Reference Data Validation](f3d_validation.md) for data layout and the
existing operational validation paths.

## 7. Current implementation versus planned work

### Current implementation

- The synthetic report can specify `scanner_backend` and `workflow_mode`
  separately. It can therefore run a selected scanner/workflow pairing without
  treating the two choices as one mode.
- [`examples/run_3d_f3d_full.py`](../examples/run_3d_f3d_full.py) is a single
  full-volume scan/vote runner. It calls `FaultOrientScanner3.scan()`, then
  performs separately configurable scanner and voter thinning.
- The current F3 full-volume runner does not implement `workflow_mode`, the
  quality skinner, skinning comparisons, or the canonical 2×2 matrix.

### Planned work

F3 full-volume execution of all four matrix cells, shared raw-scan execution,
and new comparison metrics and figures belong to later PRs. They are not
implemented by this documentation change. In particular, this document does
not define a new CLI command, report schema, output artifact, or generated
file. Any future runner must state how raw scans are shared without conflating
scanner backend outputs or downstream workflow results.

## 8. Source map

| Area | Source paths | Responsibility |
| --- | --- | --- |
| Scanner API and angle grids | [`src/pyosv/_orient3d/scanner.py`](../src/pyosv/_orient3d/scanner.py), [`sampling.py`](../src/pyosv/_orient3d/sampling.py) | Scanner entry points, reference-like scoring path, confidence, sampling, and scanner thinning |
| Synthetic scanner selection | [`src/pyosv/evaluation/synthetic_quality/scanner.py`](../src/pyosv/evaluation/synthetic_quality/scanner.py) | Maps report scanner-backend values to scanner API calls |
| Synthetic configuration | [`config.py`](../src/pyosv/evaluation/synthetic_quality/config.py), [`profiles.py`](../src/pyosv/evaluation/synthetic_quality/profiles.py), [`application.py`](../src/pyosv/evaluation/synthetic_quality/application.py) | Configuration fields, effective workflow defaults, overrides, and diagnostics |
| Voter thinning | [`src/pyosv/voting3d.py`](../src/pyosv/voting3d.py) | Stage-specific voter thinning, including `reference` and `hybrid_v2` |
| Skinning | [`src/pyosv/_skinner/reference.py`](../src/pyosv/_skinner/reference.py), [`seeds.py`](../src/pyosv/_skinner/seeds.py) | Reference and quality skinner behavior, adaptive threshold, seed gates, and effective occupancy |
| Current F3 full runner | [`examples/run_3d_f3d_full.py`](../examples/run_3d_f3d_full.py) | Current single full-volume scan/vote pipeline and F3 output comparison |
| Supporting documentation | [3D Orientation Scanning](orient3d.md), [Quality Mode](quality_mode.md), [Controlled Synthetic Quality](synthetic_quality.md), [F3 Validation](f3d_validation.md) | Detailed algorithm, benchmark, and operational context without redefining these canonical terms |
