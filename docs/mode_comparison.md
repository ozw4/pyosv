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
remain in [Quality Workflow Mode](quality_mode.md) and
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
evaluated as separate configuration axes. The library F3 full-volume runner
implements the corresponding 2×2 stage and result contracts, as described in
section 7.

When presenting results, expand the two axes in a caption or table header; do
not infer either axis from a bare `reference` or `quality` value.

### Initial F3 publication controls

The initial F3 full-volume publication protocol permits variation only in the
two named matrix axes. The following controls are fixed before any matrix cell
is run. “Held constant” means that the resolved value, including any explicit
override, is identical in all four cells unless the row explicitly assigns the
value to one scanner backend.

| Control | Fixed contract |
| --- | --- |
| Input | The same `ep.dat` full volume with shape `(420, 400, 100)` and the same file identity and checksum in every cell |
| Scanner thinning | `scanner_thin_mode=reference` in every cell |
| Scanner edge policy | Edge-effect removal is requested and effective: `remove_edge_effects=true` and `effective_remove_edge_effects=true` |
| Common scanner options | Angle-range bounds, `sigma1`, `sigma2`, interpolation, normalization, dtype (`float32`), and every scanner option other than the backend definition are held constant |
| Reference-like backend sampling | The base strike/dip sampling contract described in section 3 is fixed for both `RL-*` cells and is also the base grid refined by both `Q-*` cells |
| Quality backend refinement | The resolved value is `refinement_factor=2` in every cell; the quality backend uses it in both `Q-*` cells, while it has no effect on `RL-*`, and workflow cannot change it |
| Common voting options | Voting radii, seed distance and threshold, strain limits, attribute smoothing, surface smoothing, surface-orientation smoothing, and final normalization are held constant unless a setting is explicitly listed as workflow-owned below |
| Explicit workflow-comparison overrides | Any override added for the comparison is supplied with the same resolved value to all four cells; it must not create another cell-specific difference |

Only these downstream settings may differ because of `workflow_mode` in the
initial protocol:

| Workflow-owned setting | Reference workflow | Quality workflow |
| --- | --- | --- |
| Effective `voter_thin_mode` | `reference` | `hybrid_v2` |
| Skinner method | `reference` | `quality` |
| Minimum likelihood | `0.5` | `None` / adaptive |
| Growth source | `thinned` | `pre_thin` |
| Accepted occupancy radius | `None` (effective `5` in reports) | `1` |
| Boundary fallback and policy | disabled | enabled with `empty_primary` policy |

The scanner backend is the scanner axis, and scanner thinning is a held
constant preprocessing stage; neither is a workflow-owned difference. Settings
not named in the workflow-owned table are held constant, including all resolved
voting settings named above. An explicit override can suppress a workflow
default only when that same override is applied to every cell. The canonical
F3 config exposes this for voter thinning as
`voter_thin_mode_override`.

For each scanner backend, compute raw `ft`, `pt`, and `tt` exactly once. Apply
the fixed scanner reference thinning, including the fixed edge policy, exactly
once to obtain `fet`, `fpt`, and `ftt`. The reference and quality workflows for
that backend must share those same `fet`, `fpt`, and `ftt` volumes. A workflow
comparison must not rerun either scanning or scanner thinning.

The runner manifest records more than the matrix label. It records
the scanner backend, workflow mode, `scanner_thin_mode`, requested and effective
edge policy, refinement factor, the resolved value of every fixed scanner and
voting control above, every workflow-owned resolved value and explicit
override, and the input path/identity/checksum and shape. This evidence is
required to establish that a reported contrast differs only on the intended
axes.

Scanner-thinning comparisons such as `scanner_thin_mode=normal` are separate
ablations, or an explicitly declared third matrix axis. They must not be
encoded or reported with only a 2×2 label such as `RL-REF`; the ablation axis
and its resolved edge policy must appear in its label and manifest.

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

## 7. Current implementation and remaining work

### Current implementation

- The synthetic report can specify `scanner_backend` and `workflow_mode`
  separately. It can therefore run a selected scanner/workflow pairing without
  treating the two choices as one mode.
- `pyosv.evaluation.synthetic_mode_comparison` exposes the immutable
  `SyntheticModeComparisonConfig`, `SyntheticModeComparisonPlan`,
  `ModeCellSpec`, and `SyntheticTrialSpec` contracts. Call
  `build_mode_comparison_plan(config)` to validate case selection, seeded
  trials, fixed scanner controls, resolved reference/quality workflows, and the
  canonical cell order without running scans, metrics, or artifact writers.
  Each trial records its case ID, validated 3D shape, optional case-generation
  seed, and deterministic trial ID.
  The plan uses `current_default`; its cells are `RL-SCAN`, `Q-SCAN`, optional
  `ORACLE-REF` and `ORACLE-QUAL`, followed by the four 2x2 cells in the table
  above. Deterministic cases have one trial with `seed=None`; only registered
  stochastic cases expand in configured `trial_seeds` order.
- `run_mode_comparison(config)` executes that plan sequentially and returns a
  `SyntheticModeComparisonResult` containing JSON-safe cell reports,
  long-format metrics, paired contrasts, case-local aggregates, cache counters,
  and runtime rows. Shared scanner-input and backend scan/thinning costs are
  recorded once per trial rather than copied into workflow-cell runtimes; the
  returned object retains no full-volume arrays or fault skin/cell objects.
  Array nonzero fractions use the legacy synthetic-quality epsilon contract
  (strictly greater than `1e-6` in magnitude), and an empty configured
  truth-surface support mask is rejected after case generation but before any
  scanner work. Non-finite or negative truth-metric scalars are rejected even
  earlier, during configuration and before experiment timing or case
  generation.
- The package CLI and thin example entry point described in [Synthetic Mode
  Comparison](synthetic_mode_comparison.md) run the canonical synthetic plan,
  atomically write its scalar artifact bundle, validate completion, and print
  the completed output path. `validate_completed_bundle()` checks cross-file
  scalar semantics in addition to hashes and syntax. The authoritative writer
  emits artifact schema v3 with independently versioned scalar-evidence and
  runtime contracts (currently scalar-evidence version 5 and runtime version
  4). Each trial has one canonical truth-evidence object, and scanner, voting,
  thinning, and enabled-skin reports bind their fault-band and thin-surface
  truth counts to it. Schema-v1 bundles lack complete scanner evidence, while
  schema-v2 bundles do not uniquely identify runtime coverage; both must be
  regenerated. Enabled-skin component topology persists both duplicate-inclusive
  per-skin truth-component counts and per-skin unique coverage counts for each
  truth component. Its qualification threshold and all dominant, qualifying,
  over-merge, over-split, purity, and recall summaries are recomputed from
  those incidence tables. Truth-component totals are bound to trial truth
  evidence, and covered totals are bound to the exact skin-overlap
  intersection. Scalar-evidence v1-v4 bundles lack part of this current
  evidence contract and are rejected without upgrade. Scanner publication metrics are joined totally to
  the persisted registry-ordered evidence. This scanner evidence is prepared
  once per trial and backend, reused by scanner-only and end-to-end cells, and
  attributed to a shared runtime stage. Runtime and validation derive cache
  hit/miss expectations from the same resolved semantic stage keys. Voting and
  thinning scalar evidence, seed selection, voting volume, base thinning, and
  primary skinning are attributed by trial-local semantic-key reference count.
  Keys referenced by multiple cells use a shared row; successfully built keys
  referenced by one cell use owner-labelled rows. Cache hits add neither calls
  nor elapsed time. `cell_execution` contains only the remaining
  cell-exclusive time. Mode-owned runtime is that residual plus the cell's
  owner-labelled rows; shared rows are not apportioned.
  Runtime rows are one within-experiment breakdown, not isolated-process
  benchmarks. Their disjoint stage sum may not exceed `trial_total`, and the
  sum of trial totals may not exceed `experiment_total`. Validation
  binds top-count selections to truth-surface cardinality, enforces empty-mask,
  radius-zero, fractional-radius, and full-volume buffered-overlap rules, caps
  every mask-derived count at the canonical volume capacity (except validated
  duplicate skin cells), caps every distance summary at the volume diagonal,
  and recomputes largest/small-skin summaries from per-skin arrays and the
  effective configuration. It also validates orientation, edge,
  and component-topology algebra and compares shared scanner, voting, and
  conditionally shared thinning evidence across cells. Passing it demonstrates
  consistency of the recorded scalar evidence, not an independent
  recomputation, proof of any volume stage, or tamper-prevention signature.
  This is separate from F3 full-volume execution.
- `pyosv.evaluation.f3d_mode_comparison` implements the canonical F3
  full-volume plan, checksum-bound run workspace, shared reference-like and
  quality scanner stages, all four workflow cells, exact stage validation and
  resume, public-reference agreement metrics, 2×2 contrasts, regional and
  orientation diagnostics, and runtime/resource rows. Exact resume also
  requires the numerical runtime identity in `run_manifest.json` to match,
  including acceleration, Numba, platform, thread environment,
  `PYTHONHASHSEED`, and the NumPy BLAS/LAPACK build digest. This identity keeps
  numerical generations from being mixed; it is neither a cryptographic
  signature nor proof of bitwise reproducibility on arbitrary hardware. Cell
  references point to shared content-addressed stages instead of duplicating
  full-volume artifacts. Continuous candidate and public-reference volumes use
  `abs(value) > 1e-6` for nonzero metrics. This epsilon-aware magnitude test is
  not strict IEEE nonzero testing: negative values beyond the threshold count,
  while interpolation or smoothing tails at or below it do not. Positive
  candidate masks instead use `value > 1e-6` and exclude negative values.
  Metric evidence and scanner array summaries record the fixed epsilon so
  their nonzero counts and fractions can be reproduced.
- `python -m pyosv.cli.f3d_mode_comparison` runs, resumes, or validates that
  canonical matrix. It always targets all four cells. A run without `--resume`
  requires a new output path; exact resume reuses only matching validated
  stages, and a valid complete bundle is validated without recomputation.
  Root completion is published only after the fixed reports and referenced
  stages pass semantic validation. `--deep-validate` recomputes full-volume
  metric evidence. Runtime rows are within-run attribution, not isolated
  benchmarks.
- [`examples/run_3d_f3d_full.py`](../examples/run_3d_f3d_full.py) is a single
  legacy full-volume scan/vote command. It calls `FaultOrientScanner3.scan()`,
  then performs separately configurable scanner and voter thinning. This
  command does not implement `workflow_mode`, the quality skinner, skinning
  comparisons, or the canonical 2×2 matrix; those capabilities are provided
  by the library APIs above.

### Remaining work

Publication figure generation remains planned separately. The F3 comparison
CLI deliberately does not import visualization dependencies or generate the
PR4 publication figure set. The synthetic CLI remains separate, and the legacy
F3 command does not adopt the canonical workspace or stage cache.

## 8. Source map

| Area | Source paths | Responsibility |
| --- | --- | --- |
| Scanner API and angle grids | [`src/pyosv/_orient3d/scanner.py`](../src/pyosv/_orient3d/scanner.py), [`sampling.py`](../src/pyosv/_orient3d/sampling.py) | Scanner entry points, reference-like scoring path, confidence, sampling, and scanner thinning |
| Synthetic scanner selection | [`src/pyosv/evaluation/synthetic_quality/scanner.py`](../src/pyosv/evaluation/synthetic_quality/scanner.py) | Maps report scanner-backend values to scanner API calls |
| Synthetic configuration | [`config.py`](../src/pyosv/evaluation/synthetic_quality/config.py), [`profiles.py`](../src/pyosv/evaluation/synthetic_quality/profiles.py), [`application.py`](../src/pyosv/evaluation/synthetic_quality/application.py) | Configuration fields, effective workflow defaults, overrides, and diagnostics |
| Synthetic comparison planning | [`src/pyosv/evaluation/synthetic_mode_comparison/`](../src/pyosv/evaluation/synthetic_mode_comparison/) | Canonical cells, validated execution-free plans, and deterministic/seeded trial expansion |
| F3 full-volume comparison | [`src/pyosv/evaluation/f3d_mode_comparison/`](../src/pyosv/evaluation/f3d_mode_comparison/) | Canonical 2×2 plan and stages, exact resume, reference-agreement metrics, contrasts, diagnostics, and resources |
| F3 comparison CLI | [`src/pyosv/cli/f3d_mode_comparison.py`](../src/pyosv/cli/f3d_mode_comparison.py), [`examples/run_3d_f3d_mode_comparison.py`](../examples/run_3d_f3d_mode_comparison.py) | Full-volume four-cell run, exact resume, strict/deep validation, and thin example entry point |
| Voter thinning | [`src/pyosv/voting3d.py`](../src/pyosv/voting3d.py) | Stage-specific voter thinning, including `reference` and `hybrid_v2` |
| Skinning | [`src/pyosv/_skinner/reference.py`](../src/pyosv/_skinner/reference.py), [`seeds.py`](../src/pyosv/_skinner/seeds.py) | Reference and quality skinner behavior, adaptive threshold, seed gates, and effective occupancy |
| Legacy F3 full runner | [`examples/run_3d_f3d_full.py`](../examples/run_3d_f3d_full.py) | Separate single-path full-volume scan/vote pipeline and F3 output comparison |
| Supporting documentation | [3D Orientation Scanning](orient3d.md), [Quality Workflow Mode](quality_mode.md), [Controlled Synthetic Quality](synthetic_quality.md), [Synthetic Mode Comparison](synthetic_mode_comparison.md), [F3 Validation](f3d_validation.md) | Detailed algorithm, benchmark, and operational context without redefining these canonical terms |
