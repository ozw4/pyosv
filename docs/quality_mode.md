# Quality Workflow Mode

This document describes the downstream quality workflow profile. The quality
workflow is distinct from the quality scanner backend: selecting either one
does not select the other. A workflow profile also does not implicitly change
`scanner_thin_mode`; scanner reference thinning and voter reference thinning
are separate stages. See [scanner backends, workflow modes, thinning modes, and
reference targets](mode_comparison.md) for the canonical contract.

In the benchmark history below, **legacy quality-backend** means a historical
run that selected the `quality` scanner backend. It does not name the quality
workflow or imply that the backend produced higher-quality results.

## Workflows

`reference` workflow is for origin-aligned regression comparison. Its defaults
keep reference-like voter thinning and disable support-aware surface voting
(`surface_support_min_fraction=0.0`,
`surface_support_exponent=0.0`). It also keeps
`surface_voting_boundary_policy="reference"`: UVW samples outside the volume
are clamped to the image edge, while `i2` and `i3` face source samples are
excluded from surface vote averaging and accumulation. Use this workflow when
checking that Python behavior remains close to the current reference-oriented
path. It is not the place to evaluate processing-quality improvements.

`quality` workflow is the current quality-first synthetic profile. Its defaults
use `hybrid_v2` voter thinning, disable support-aware surface voting, use the
quality skinner v2 profile, and enable boundary skinner fallback
(`surface_support_min_fraction=0.0`,
`surface_support_exponent=0.0`, `FaultSkinner(method="quality")`,
`growth_source=pre_thin`, `accepted_occupancy_radius=1`,
`boundary_skinner_fallback=true`,
`boundary_skinner_fallback_policy=empty_primary`). The default fallback only
runs when primary skinning returns no skins and the thinned vote volume has
positive samples. It is not a universal production profile; use it while
reviewing the controlled synthetic benchmark matrix and checking that the
candidate set is not over-filtered for the cases under study. The quality
workflow also retains `surface_voting_boundary_policy="reference"`; masked UVW
boundary voting is not part of `current_default`.

Omitting the boundary-fallback CLI options keeps this quality-workflow default.
Pass `--no-skinner-boundary-fallback` to disable it explicitly, or
`--skinner-boundary-fallback` to enable it explicitly. Reference and diagnostic
workflows continue to preserve their configured fallback value.

`diagnostic` workflow keeps the reference workflow defaults and enables
thinning diagnostics. Use it when comparing current behavior, diagnostic
variants, and reference-vs-normal thinning on the same synthetic truth.

Resolution is deterministic: base configuration is filled by the selected
workflow profile, explicit CLI values override those defaults, and the selected
variant's declared patch is applied for that variant. Variant definitions live
in one registry, while promotion thresholds live in one promotion
specification. See [Architecture](architecture.md) for the ownership boundary.
Scanner backend and scanner thinning are resolved independently and are not
filled by the workflow profile.

## Boundary-aware Voter Candidate

`boundary_aware_voter_v1` is an explicit-only synthetic-report variant. It
preserves the scanner backend and scanner thinning chosen by the run/scanner
configuration. It keeps the selected workflow's downstream seed selection,
voter thinning, and skinner settings identical to `current_default`, but calls
`set_surface_voting_boundary_policy("masked_in_bounds")`. It is absent from
the default variant list and the `quality-matrix` preset.

The masked policy is a quality experiment, not a reference-equivalence mode. It
does not clamp out-of-volume UVW samples. It carries an explicit lag mask into
surface extraction, crops tangential support to the deterministic maximum
all-supported rectangle containing the local origin, and maps that crop with
full-box offsets. Invalid DP states cannot be selected; infeasible surfaces are
diagnosed and skipped. After smoothing, mask validity and strain are rechecked
in both tangential directions. A deterministic global feasibility recovery is
used when necessary; it retains a fractional value that is already feasible in
its Java-rounding cell instead of moving it unnecessarily to an integer center.
If no jointly feasible result exists, the seed records
`skip_reason="no_feasible_surface"`. Scoring uses only valid selected samples
and normalizes support against the full tangential patch. Center votes may land
on all six faces, with bounds-checked reinforcement writes.

Per-seed diagnostics report full/cropped support, smoothing projections,
selected-invalid samples, face center votes, orientation source, and skip
reason; their aggregate reports boundary-affected, voted, and skipped seed
counts plus support/projection/vote totals. `surface_projection_count` is the
number of `(w, v)` columns whose value differs between the raw smoothed surface
and the final mask-and-strain-feasible surface, with each changed column counted
once. It is zero when surface smoothing is disabled. A full tangential box uses
the extracted surface orientation. A cropped box deliberately falls back to
the seed orientation and records
`orientation_source="seed_boundary_fallback"`; local boundary-normal
estimation is outside this candidate's scope.

Synthetic JSON stores the aggregate under `pyosv.voting.diagnostic_summary`
beside `surface_voting_boundary_policy` and the existing support settings.
`summary.csv` exposes the policy plus boundary-affected/skipped seed counts,
support-fraction mean/minimum, projection count, selected-invalid count, and
face center-vote count. Reference rows use nonblank neutral/count values.

This candidate is not promoted and must not be described as higher quality or
as a new default until the documented scanner-boundary promotion gate passes.

## Synthetic Truth Benchmark

Use this command as the recommended reproducible benchmark matrix before and
after quality changes:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 33,33,33 \
  --variant-preset quality-matrix \
  --input-mode both \
  --workflow-mode diagnostic \
  --output-dir outputs/3d/synthetic_quality/quality_matrix_001 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

Review `summary.csv` first, then use `metrics.json` and visual overlays for
drill-down.

For the recommended oracle 49^3 quality benchmark, run the quality workflow on
the extended case set with the current default only:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 49,49,49 \
  --workflow-mode quality \
  --variants current_default \
  --input-mode oracle \
  --output-dir outputs/3d/synthetic_quality/oracle_quality_current_49 \
  --pretty
```

For the historical scanner-inclusive quality-workflow evaluation, explicitly
pair the workflow with the refined opt-in quality scanner backend:

```bash
--input-mode scanner \
--scanner-backend quality \
--scanner-refinement-factor 2
```

This benchmark pairing is not an implicit workflow default. The report default
remains `--scanner-backend reference-like` so scanner behavior is not changed
automatically.

### Scanner thinning policy comparison

Evaluate normal scanner thinning as a policy candidate by generating two
separate reports for the same `current_default` variant. The reference-thinning
report is the baseline:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 49,49,49 \
  --workflow-mode quality \
  --variants current_default \
  --input-mode both \
  --scanner-backend quality \
  --scanner-refinement-factor 2 \
  --scanner-thin-mode reference \
  --scanner-downstream-diagnostics \
  --scanner-boundary-stage-diagnostics \
  --output-dir outputs/3d/synthetic_quality/scanner_thin_reference_49 \
  --pretty
```

The normal-thinning report is the candidate:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 49,49,49 \
  --workflow-mode quality \
  --variants current_default \
  --input-mode both \
  --scanner-backend quality \
  --scanner-refinement-factor 2 \
  --scanner-thin-mode normal \
  --scanner-downstream-diagnostics \
  --scanner-boundary-stage-diagnostics \
  --output-dir outputs/3d/synthetic_quality/scanner_thin_normal_49 \
  --pretty
```

Apply the existing scanner-boundary gate with the scanner-policy comparison
profile:

```bash
PYTHONPATH=src python scripts/compare_quality_reports.py \
  outputs/3d/synthetic_quality/scanner_thin_reference_49/summary.csv \
  outputs/3d/synthetic_quality/scanner_thin_normal_49/summary.csv \
  --baseline-metrics outputs/3d/synthetic_quality/scanner_thin_reference_49/metrics.json \
  --candidate-metrics outputs/3d/synthetic_quality/scanner_thin_normal_49/metrics.json \
  --baseline-variant current_default \
  --candidate-variant current_default \
  --comparison-profile scanner-thinning-policy-v1 \
  --promotion-gate scanner-boundary \
  --strict-missing-rows \
  --fail-on-gate-failure \
  --output-json outputs/3d/synthetic_quality/scanner_thin_normal_49/promotion_gate.json \
  --output-markdown outputs/3d/synthetic_quality/scanner_thin_normal_49/promotion_gate.md
```

This is a comparison of one variant across two reports, not a comparison of
two voter/skinner variants. The contract identifies the baseline as
`quality_scanner_reference_v1` and the candidate as
`quality_scanner_thin_normal_v1`. Scanner thinning mode is therefore not part
of the summary-row match key. Instead, the comparison reads both `metrics.json`
files and enforces a run-config contract: the only permitted difference is
`config.scanner.scanner_thin_mode`, directed from `reference` in the baseline
to `normal` in the candidate. All other run configuration, including the
requested `remove_edge_effects=true`, must match.

Each baseline/candidate `summary.csv` must come from its paired `metrics.json`.
For this profile the comparison regenerates the canonical summary CSV v1 in
memory and requires the exact header and all cell values to match, treating
data rows as a multiset so row order is ignored but missing, extra, and
duplicate rows are rejected. It also requires the selected variant to appear
in each report's `config.variants`. Evidence mismatch is an input error, so no
numeric gate report is written. This validation does not apply to the default
`variant` comparison profile.

Edge cleanup applies only to reference thinning. The comparison artifact
therefore records the baseline's requested and effective edge removal as
`true`, while normal thinning records requested edge removal as `true` and
`effective_remove_edge_effects=null` to mean not applicable. A passing gate is
an evaluation result for the normal-thinning policy candidate; it does not
change any quality, reference, diagnostic, or scanner API default.

These 49^3 commands are the formal reproduction procedure. The generated
comparison artifact, rather than the command listing itself, is the evidence
for a measured contract and gate result.

### Reference-like backend scanner thinning policy comparison

The separate `quality-workflow-scanner-thinning-v1` profile evaluates normal
scanner thinning on the current quality-workflow default backend without
changing `scanner-thinning-policy-v1`. Its baseline policy is
`quality_reference_like_scanner_thin_reference_v1`; its candidate policy is
`quality_reference_like_scanner_thin_normal_v1`.

The `scanner-boundary-reference-like` gate is dedicated to this profile and
requires `quality-workflow-scanner-thinning-v1`. The CLI default is the
`variant` profile, so omitting `--comparison-profile` (or explicitly selecting
`variant`) is rejected rather than inferred or corrected automatically. A
formal promotion artifact requires the profile's canonical `summary.csv` /
`metrics.json` pairing, passing policy contract, and complete 14-row coverage;
the numeric gate result alone is not valid evidence.

Generate the reference-thinning baseline:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 49,49,49 \
  --workflow-mode quality \
  --variants current_default \
  --input-mode both \
  --scanner-backend reference-like \
  --scanner-thin-mode reference \
  --scanner-downstream-diagnostics \
  --scanner-boundary-stage-diagnostics \
  --output-dir outputs/3d/synthetic_quality/reference_like_scanner_thin_reference_49 \
  --pretty
```

Generate the normal-thinning candidate:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 49,49,49 \
  --workflow-mode quality \
  --variants current_default \
  --input-mode both \
  --scanner-backend reference-like \
  --scanner-thin-mode normal \
  --scanner-downstream-diagnostics \
  --scanner-boundary-stage-diagnostics \
  --output-dir outputs/3d/synthetic_quality/reference_like_scanner_thin_normal_49 \
  --pretty
```

Compare the two paired reports:

```bash
PYTHONPATH=src python scripts/compare_quality_reports.py \
  outputs/3d/synthetic_quality/reference_like_scanner_thin_reference_49/summary.csv \
  outputs/3d/synthetic_quality/reference_like_scanner_thin_normal_49/summary.csv \
  --baseline-metrics outputs/3d/synthetic_quality/reference_like_scanner_thin_reference_49/metrics.json \
  --candidate-metrics outputs/3d/synthetic_quality/reference_like_scanner_thin_normal_49/metrics.json \
  --baseline-variant current_default \
  --candidate-variant current_default \
  --comparison-profile quality-workflow-scanner-thinning-v1 \
  --promotion-gate scanner-boundary-reference-like \
  --strict-missing-rows \
  --fail-on-gate-failure \
  --output-json outputs/3d/synthetic_quality/reference_like_scanner_thin_normal_49/promotion_gate.json \
  --output-markdown outputs/3d/synthetic_quality/reference_like_scanner_thin_normal_49/promotion_gate.md
```

This is the same `current_default` variant in two separately generated
reports. Scanner thinning mode is not added to the row match key. Instead,
the canonical summary/metrics pairing check validates each evidence pair, and
the config contract fixes the extended 49^3 quality run, reference-like
backend, variant list, scanner settings and input, voting, skinning, truth
metrics, and diagnostics. The only permitted config difference is
`config.scanner.scanner_thin_mode`, directed from `reference` to `normal`.

Both commands retain requested edge cleanup. The baseline therefore records
`requested_remove_edge_effects=true` and
`effective_remove_edge_effects=true`. Normal thinning does not apply edge
cleanup, so the candidate records `requested_remove_edge_effects=true` and
`effective_remove_edge_effects=null` rather than `false`.

`scanner-boundary-reference-like` uses the same numeric limits as the existing
scanner-boundary gate. Boundary skin F1 must be at least 0.90, skin count at
most 3, the skin/FVT-positive cell ratio within [0.75, 1.25], FVT-positive F1
at least 0.90, and FVT-positive distance p95 at most 2.0. Non-boundary skin
and FVT-positive F1 deltas must be at least -0.02, and both distance-p95 deltas
must be at most 2.0. Oracle metrics must remain unchanged; false fallback
replacements and parallel/crossing over-merge or over-split counts must not
increase. The gate also requires all 14 rows, no missing rows, canonical
summary/metrics pairing, and a passing policy contract.

The formal 49^3 evaluation above has been completed and passed. The comparison
contained all 14 required rows with no missing baseline or candidate rows; its
coverage and scanner-policy configuration contracts both passed. On the scanner
`boundary_plane` row, the reference-to-normal policy change produced:

| metric | reference baseline | normal candidate | delta |
|---|---:|---:|---:|
| `skin_buffered_f1_r2` | 0.347853 | 0.993151 | +0.645298 |
| `skin_count` | 29 | 1 | -28 |
| `skin_cell_count / fvt_positive_candidate_count` | 0.265306 | 1.000000 | +0.734694 |
| `fvt_positive_buffered_f1_r2` | 0.497641 | 0.993151 | +0.495509 |
| `fvt_positive_distance_p95` | 5.0 | 1.0 | -4.0 |
| `fvt_positive_candidate_count` | 2401 | 2303 | -98 |

The gate reported no material non-boundary, oracle, fallback-replacement, or
required topology regression. The SHA-256 of the generated
`promotion_gate.json` is
`1b099e06c8900181da68a3c437573c5d83868f727d37675a206380786aca7639`.
Exact values and hashes for both `metrics.json` files, both `summary.csv` files,
and the JSON and Markdown promotion artifacts are retained in
`tests/fixtures/synthetic_quality_refactor/reference_like_scanner_thinning_49_evidence.json`.
The ignored reports did not record a source commit, so the compact evidence
records `source_commit=null` and `source_provenance="not_recorded"` rather than
inferring one.

This pass evaluates the synthetic policy candidate only. A historical F3
64^3-by-3 shared-scan crop diagnostic later failed the public-FVT
sparse-distance p95 check (`+6.193637` samples on crop 1 versus the `+5.0`
limit); the other seven diagnostic checks passed. The prerequisite large-crop
diagnostic was not run and human review remains pending. This is retained as
historical crop evidence, not a current publication gate or a set of
statistical replicates. The quality, reference, and diagnostic workflow
defaults and the public `FaultOrientScanner3.thin()` default remain unchanged.

For the 49^3 scanner-boundary promotion benchmark, run the current default and
the opt-in diagnostic candidates with downstream diagnostics enabled:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 49,49,49 \
  --workflow-mode quality \
  --variants current_default,boundary_aware_voter_v1,boundary_edge_thin_v1,boundary_seed_retention_v1,quality_boundary_skinner_fallback_v5 \
  --input-mode both \
  --scanner-backend quality \
  --scanner-refinement-factor 2 \
  --scanner-downstream-diagnostics \
  --output-dir outputs/3d/synthetic_quality/promotion_candidates_49 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

Then compare each candidate against `current_default`. `compare_quality_reports.py`
can compare two variants from the same `summary.csv`; pass the same file for
the baseline and candidate summaries and select variants explicitly:

```bash
python scripts/compare_quality_reports.py \
  outputs/3d/synthetic_quality/promotion_candidates_49/summary.csv \
  outputs/3d/synthetic_quality/promotion_candidates_49/summary.csv \
  --baseline-variant current_default \
  --candidate-variant quality_boundary_skinner_fallback_v5 \
  --promotion-gate scanner-boundary \
  --output-json outputs/3d/synthetic_quality/promotion_candidates_49/fallback_v5_delta.json \
  --output-markdown outputs/3d/synthetic_quality/promotion_candidates_49/fallback_v5_delta.md
```

Repeat the same command with `--candidate-variant boundary_aware_voter_v1`,
`--candidate-variant boundary_edge_thin_v1`, and
`--candidate-variant boundary_seed_retention_v1`, or run the aggregate checker:

```bash
python scripts/check_synthetic_quality_promotion_gate.py \
  --baseline-summary outputs/3d/synthetic_quality/promotion_candidates_49/summary.csv \
  --candidate-summary outputs/3d/synthetic_quality/promotion_candidates_49/summary.csv \
  --candidate-variants boundary_aware_voter_v1,boundary_edge_thin_v1,boundary_seed_retention_v1,quality_boundary_skinner_fallback_v5 \
  --output-json outputs/3d/synthetic_quality/promotion_candidates_49/promotion_gate.json \
  --output-markdown outputs/3d/synthetic_quality/promotion_candidates_49/promotion_gate.md
```

The aggregate checker requires matched 49^3 extended-case rows using
`scanner_backend=quality` and `scanner_refinement_factor=2` for the boundary
scanner gate, non-boundary scanner regression checks, stable-case fallback
replacement checks, and parallel/crossing topology checks; oracle regression
checks require matched 49^3 oracle rows.
If any required coverage is absent, `promotion_gate.json` records the candidate
as not promotable even when the available boundary row passes.

Then print the concise oracle-vs-scanner comparison table:

```bash
python examples/print_synthetic_quality_comparison.py \
  outputs/3d/synthetic_quality/promotion_candidates_49/summary.csv
```

The helper reports `case_id`, `variant`, oracle/scanner fvt-positive F1,
oracle/scanner skin F1, scanner-minus-oracle deltas, scanner `ft` F1,
scanner downstream fvt-to-ft distance p95, and whether fallback was used.

For diagnostic scanner-inclusive experiments, `--scanner-backend ensemble`
combines the `reference-like`, `quality`, and `fast` scanner outputs. The rule
is deterministic: normalize each backend `ft` to unit range, multiply by fixed
backend priors, apply the quality confidence map as a small extra quality
weight, then select the winning backend's `ft/pt/tt` per voxel. The report
records `scanner.selection_fraction_by_backend`,
`scanner.ensemble.components`, and the
`scanner_ensemble_reference_like_fraction`,
`scanner_ensemble_quality_fraction`, and `scanner_ensemble_fast_fraction`
summary CSV columns. This is a diagnostic backend and is not the F3 or
synthetic report default. Current scanner-inclusive evidence still shows
boundary collapse for the ensemble backend, including an empty
`boundary_plane` FVT/skin result in the 33^3 quality scanner ensemble run, so
it is not a default-promotion candidate.

To compare scanner backend tradeoffs in one run, add
`--scanner-backend-matrix` with `--input-mode scanner` or `--input-mode both`.
The matrix writes `reference-like`, `quality`, and `fast` scanner pipeline
reports plus best-backend summary columns in `summary.csv`.

The `quality-matrix` preset includes `current_default`,
`no_surface_orientation_smoothing`, `final_norm_smoothing_1`,
`voter_thin_normal`, `voter_thin_hybrid`, `voter_thin_hybrid_v2`,
`voter_thin_normal_plateau`, `surface_support_weighted`,
`quality_skinner_v2`, `quality_boundary_skinner_fallback`,
`quality_boundary_skinner_fallback_v2`,
`quality_boundary_skinner_fallback_v3`, and
`quality_boundary_skinner_fallback_v4`. The quality workflow default uses the
`hybrid_v2` voter thinning path. `boundary_aware_voter_v1` is intentionally not
in this preset and remains available only by explicit `--variants` selection.
The hybrid voter thinning variant uses reference-like thinning in
stable-orientation regions and fault-normal thinning where local orientation
changes rapidly. The `voter_thin_hybrid_v2` diagnostic variant
keeps that stable-plane preference, only adopts positive fault-normal
candidates in rough-orientation regions, and uses plateau-aware edge fallback
with the input fault likelihood as the retained-layer tie-breaker. The
`voter_thin_hybrid_v2_recenter_scanner_target` diagnostic variant keeps
`hybrid_v2` thinning but recenters edge-shell positive FVT samples toward the
scanner-thinned `fet` target. It is available by explicit `--variants`
selection for scanner-boundary diagnostics and is not part of the
`quality-matrix` preset. It is not a default candidate unless it also meets the
FVT-positive F1 and distance gates. The `boundary_edge_thin_v1` diagnostic
variant also starts from `hybrid_v2`, but uses the scanner-boundary target
during edge-shell thinning candidate selection instead of moving samples after
thinning. It is available only by explicit `--variants boundary_edge_thin_v1`
selection and is not part of the default or `quality-matrix` preset. The
`boundary_seed_retention_v1` diagnostic variant keeps the normal voter and
thinning path, but adds boundary-shell seeds whose input/scanner target remains
positive before surface voting. Scanner runs use `scanner_fet` as the target;
oracle runs use oracle `ft`. This is a target-aware seed diagnostic, does not
use truth arrays for seed selection, and is available only by explicit
`--variants boundary_seed_retention_v1` selection, not by default or
`quality-matrix`. The
`voter_thin_normal_plateau` diagnostic variant keeps fault-normal thinning
explicit, but collapses normal-direction plateau runs with the input fault
likelihood as the retained-layer tie-breaker.

The `surface_support_weighted` diagnostic variant keeps the default thinning
path but enables support-aware surface voting with
`surface_support_min_fraction=0.5` and `surface_support_exponent=1.0`. This
skips extracted surfaces with low valid support and down-weights the remaining
vote by its valid-support fraction, which is useful for boundary-plane edge
artifact diagnostics. It is a diagnostic experiment in the matrix, not the
quality workflow default. The reference, quality, and diagnostic workflow
default support policy is `0.0, 0.0`, so support-aware voting is inactive
unless the report CLI flags `--surface-support-min-fraction` or
`--surface-support-exponent` are set, or this diagnostic variant is selected.

The `quality_skinner_v2` diagnostic variant keeps the voter path selected by
the workflow, but uses the quality skinner with an adaptive seed threshold and
fixed quality grow threshold,
`growth_source=pre_thin`, and `accepted_occupancy_radius=1`. In
`--workflow-mode quality`, it matches `current_default`; in `reference` and
`diagnostic` workflows, it remains an explicit diagnostic skinning override.

The `quality_boundary_skinner_fallback` diagnostic variant forces
`boundary_skinner_fallback=true`. Under `--workflow-mode quality`, this matches
`current_default`; in `reference` and `diagnostic` workflows, it remains an
explicit diagnostic fallback override.

The `quality_boundary_skinner_fallback_v2` and
`quality_boundary_skinner_fallback_v3` variants are diagnostic degraded-primary
fallback candidates. v2 uses the `degraded_primary` policy and can improve the
scanner-inclusive boundary skin, but it over-includes fallback components and
is not good enough for default promotion. v3 uses the filtered
`degraded_primary_filtered` policy. In the 49^3 scanner-inclusive extended
benchmark with the quality scanner backend and refinement factor 2, v3 kept
`boundary_plane` `fvt_positive_buffered_f1_r2=0.739494` but only reached
`skin_buffered_f1_r2=0.834231`, with `skin_count=1` and
`skin_cell_count/fvt_positive_candidate_count=1.646814`. It also regressed
non-boundary skin F1 by more than 0.02 for `parallel_planes`,
`single_dipping_plane`, and `single_vertical_plane`. The default-promotion
target was boundary skin F1 at least 0.90, skin count at most 3, skin-cell to
positive-fvt-candidate ratio at least 0.75, and no non-boundary skin/FVT
regression beyond the configured tolerances. v3 therefore remains diagnostic.
The `quality_boundary_skinner_fallback_v4` diagnostic variant uses the
`degraded_primary_skeletonized` policy. It requires the boundary-specific
degraded-primary trigger to be supported by scanner-target diagnostics, accepts
the filtered fvt-positive components, then collapses connected runs along each
sample's dominant fault-normal array axis before building fallback skins. Its
diagnostics record the pruning method, raw and pruned component cell counts,
removed-cell count, pruned fraction, largest component size before and after
pruning, and dominant skeletonization axis mode. It remains a diagnostic
variant and is not the default.

The `quality_boundary_skinner_fallback_v5` diagnostic variant uses the
`degraded_primary_topology_guarded` policy on top of the `quality_skinner_v2`
profile. It keeps the v4 filtered component skeletonization path, but only
replaces a degraded non-empty primary skin when fallback topology guardrails
pass: skin count at most 3, fallback coverage of FVT positives between 0.75
and 1.25, small-skin cell fraction at most 0.20, largest-skin fraction at
least 0.50, and removed-by-pruning fraction at most 0.60. Its diagnostics
record `fallback_v5_guardrail` with pass/fail state, reasons, and the measured
guardrail values; `summary.csv` also exposes flat
`skin_fallback_v5_guardrail_*` columns. v5 is available by explicit
`--variants quality_boundary_skinner_fallback_v5` selection only. It is a
default-promotion candidate for investigation, but this version does not make
it the default and does not include it in the `quality-matrix` preset.

The scanner-boundary default-promotion gate is intentionally stricter than the
CI guardrails. On `boundary_plane`, scanner, shape 49, a candidate must satisfy
`skin_buffered_f1_r2 >= 0.90`, `skin_count <= 3`, and
`0.75 <= skin_cell_count / fvt_positive_candidate_count <= 1.25`. If the
candidate changes FVT, it must also satisfy
`fvt_positive_buffered_f1_r2 >= 0.90` and
`fvt_positive_distance_p95 <= 2.0`. Non-boundary scanner shape-49 cases must
show no material skin-F1 regression versus `current_default` and no false
fallback replacement on stable non-boundary cases unless metrics improve.
Oracle shape-49 behavior must not materially regress.
`parallel_planes` and `crossing_planes` also must not worsen the
component-aware over-merge or over-split counts.

The previously documented 33^3 and 49^3 legacy quality-backend
scanner-inclusive gate runs kept the quality workflow's `current_default`
unchanged. In the
49^3 scanner run with `--scanner-backend quality --scanner-refinement-factor 2`,
`current_default` on `boundary_plane` had
`fvt_positive_buffered_f1_r2=0.739494`,
`fvt_positive_distance_p95=4.0`, `skin_buffered_f1_r2=0.453890`,
`skin_count=17`, and
`skin_cell_count/fvt_positive_candidate_count=0.311120`.
`voter_thin_hybrid_v2_recenter_scanner_target` only changed the boundary FVT
metrics to `fvt_positive_buffered_f1_r2=0.740855` and
`fvt_positive_distance_p95=4.0`, so it failed the FVT promotion gate.
`quality_boundary_skinner_fallback_v4` improved scanner boundary skin to
`skin_buffered_f1_r2=0.796428` with
`skin_cell_count/fvt_positive_candidate_count=0.857976`, but it produced
`skin_count=42`, still missed the 0.90 skin-F1 target, and the oracle
`boundary_plane` row collapsed to `skin_buffered_f1_r2=0.0`. v4 therefore
also remains diagnostic. The known boundary issue remains open for this legacy
quality-backend candidate flow: scanner `ft` can be high quality while
downstream FVT and skinning degrade near boundaries. This is distinct from the
passing reference-like scanner backend thinning-policy gate documented above.

For the promotion-candidate flow above, no new 49^3
`promotion_candidates_49` result for `boundary_aware_voter_v1` has been
recorded in this repository update. Adding it to the reproducible command does
not imply that benchmark was run. The quality workflow's `current_default`
profile is therefore unchanged, and `boundary_aware_voter_v1`,
`boundary_edge_thin_v1`,
`boundary_seed_retention_v1`, and `quality_boundary_skinner_fallback_v5` remain
unpromoted until their `promotion_gate.json` shows the scanner-boundary gate
passing without material non-boundary, oracle, fallback-replacement, or
topology regressions.

For skin extraction, `--workflow-mode quality` defaults to
`--skinner-method quality` unless `--skinner-method` is passed explicitly. The
quality skinner reuses reference-like skin growth and reskinning, but uses
adaptive `min_likelihood` when `--skinner-min-likelihood` is omitted and lowers
the seed planarity gate from `ep > 0.8` to `ep > 0.5`. The quality workflow
grows from the pre-thin vote volume and records
`effective_accepted_occupancy_radius=1`. Synthetic reports record the selected
`skinning.method`, whether the likelihood threshold is adaptive, the seed `ep`
threshold, `growth_source`, `effective_accepted_occupancy_radius`,
`boundary_skinner_fallback`, and `seed_planarity_source=fvt` in `metrics.json`.
An explicit `--no-skinner-boundary-fallback` overrides the quality-workflow
fallback default without changing its recorded fallback policy.

Primary metrics to compare:

- `fvt_buffered_f1_r2`
- `fvt_distance_candidate_to_truth_p95`
- `fvt_strike_median_error`
- `fvt_dip_median_error`
- `skin_buffered_f1_r2`
- `skin_distance_candidate_to_truth_p95`
- `edge_false_positive_fraction` columns

## Oracle vs Scanner-Inclusive Quality-Workflow Evaluation

Oracle-input and scanner-inclusive `current_default` runs under the quality
workflow should be reviewed separately. The oracle path at 49^3 is the
stable controlled-truth baseline for the current quality-workflow default,
including the empty-primary fallback on `boundary_plane`. Scanner-inclusive evaluation also
exercises scanner `ft` recovery and downstream fvt/skinning behavior; its
boundary skin can degrade even when scanner `ft` is strong, which is why the
degraded-primary fallback variants remain explicit diagnostics rather than
defaults.

## CI Regression Guardrails

The always-on quality workflow regression test is intentionally synthetic-only:
it does not require F3 data, `reference_osv`, Java/Jython/JTK, or external
downloads. It builds the `extended` synthetic case set at a small shape for both
`reference` and `quality` workflows using only `current_default`.

The guardrails are broad. They assert that key overlap, distance, orientation,
edge false-positive, and skin metrics remain finite, that quality workflow
effective settings are recorded in `metrics.json`, and that the quality workflow has not
clearly regressed relative to the reference workflow on the extended synthetic
cases:
single vertical, single dipping, curved, parallel, crossing, boundary, and
weak/noisy. Boundary-plane guardrails require the quality workflow's
`current_default` to
produce positive fvt candidates with buffered F1 at least `0.98`, distance p95
at most `1.0`, no edge false positives, and a recovered skin via the reported
fallback path with skin buffered F1 at least `0.5`. These thresholds are not
benchmark targets. They are meant to catch obvious workflow breakage while
leaving room for normal tuning changes.

## F3 Full-Volume Real-Data Comparison

Publication-facing F3 comparison uses only the complete `(420, 400, 100)`
volume as one evaluation unit. Public `fl.dat`, `fv.dat`, and `fvt.dat` support
F3 public reference agreement and difference measurements; they are not
independent geological truth, so agreement with them is not quality or
accuracy. Known-truth accuracy, recovery, and topology claims come from the
controlled synthetic `extended` matrix described above. F3 supplies the
complementary truthless real-data reference-agreement review.

The current [`examples/run_3d_f3d_full.py`](../examples/run_3d_f3d_full.py)
runner is a manual, potentially slow, reference-like baseline full-volume
scan/vote path. It exposes scanner and voter thinning separately, but does not
implement `workflow_mode`, the quality scanner backend, the quality skinner, or
the canonical 2×2 scanner-backend/workflow comparison. In particular, a quality
workflow does not imply a quality scanner backend. The planned full-volume
`RL-REF`, `RL-QUAL`, `Q-REF`, and `Q-QUAL` runner is future work; the labels and
separate axes are defined in
[Scanner, Workflow, Thinning, and F3 Reference Comparison](mode_comparison.md).
The complete publication protocol and current/planned boundary are documented
in [F3 3D Reference Data Validation](f3d_validation.md).

## Legacy/Internal F3 Crop Diagnostics

The existing multi-crop report can compare `reference` and `quality` workflows
at the same crop centers for debugging and preservation of historical evidence:

```bash
PYTHONPATH=src python examples/report_3d_f3d_multicrop.py \
  --data-root "$PYOSV_F3D_DATA_ROOT" \
  --count 3 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --compare-workflows \
  --save-figures \
  --write-markdown-index \
  --output-json outputs/3d/f3d/quality_external_smoke_001/metrics.json \
  --pretty
```

This command is an optional legacy/internal crop diagnostic, not the F3
publication comparison path. Its JSON and markdown include
`consensus.workflows` for each workflow and
`consensus.workflow_comparison.quality_minus_reference` in compare mode.
Compare-mode reports also include top-level `quality_validation`, a truthless
external-smoke diagnostic. Its default conservative checks fail on finite metric
failures, quality-workflow FVT density above `2.0x` the reference workflow, FVT
edge-density proxy delta above `0.10`, sparse distance p95 regression above
`5.0` samples, or
extreme crop-to-crop density CV. Use these checks alongside the existing
reference-overlap metrics only to diagnose local behavior. Crop-to-crop
stability is historical diagnostic context, not a publication acceptance
criterion, and crops must not be described as independent samples or
replicates.

This F3 smoke requires external F3 volumes and should not be mandatory in CI.
CI should keep using mock/fixture structure tests for the report and markdown
schema.

The historical reference-like 49^3 scanner-thinning candidate used the
dedicated shared-scan crop diagnostic rather than `--compare-workflows`. The
dedicated path runs `FaultOrientScanner3.scan()` once per crop, changes only
scanner thinning from `reference` to `normal`, and fixes the quality downstream
voter to `hybrid_v2` with each branch's own scanner-thinned `fet` as the plateau
tie-breaker. Its profile is `quality-workflow-scanner-thinning-v1`; the policies
are `quality_reference_like_scanner_thin_reference_v1` and
`quality_reference_like_scanner_thin_normal_v1`. It reports
`policy_validation` with role `truthless_external_smoke`, not a promotion gate,
because public F3 `fv.dat` and `fvt.dat` are not independent truth. The
historical 64^3 multi-crop and large-crop commands, automatic checks, manual
review list, and evidence policy are preserved in
[F3 3D Reference Data Validation](f3d_validation.md). The historical 64^3-by-3
diagnostic completed with three scanner executions and finite, nonempty
outputs, but `policy_validation.passed=false`: crop 1's candidate
public-FVT sparse-distance p95 was `8.429705` versus baseline `2.236068`, a
`+6.193637`-sample regression above the `+5.0` limit. All other automatic
diagnostic checks passed. Per the documented ordering, the large crop was not
run; human geological review was not completed. This failure remains historical
crop diagnostic evidence and is not reused as a full-volume publication gate.

Until controlled synthetic truth evidence and the future full-volume F3
reference-agreement review exist, keep `reference` as the default workflow
value and use `quality` only as an explicit workflow value.
