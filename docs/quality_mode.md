# Quality Workflow Mode

This document defines the downstream workflow profiles used by controlled
synthetic quality evaluation. A workflow profile resolves voting, voter
thinning, skinning, and diagnostic defaults. It does not select a scanner
backend or scanner thinning mode.

The processing stages are:

```text
input -> scanner -> scanner thinning -> voting -> voter thinning -> skinning
```

`scanner_backend`, `scanner_thin_mode`, `workflow_mode`, and
`voter_thin_mode` are independent settings and are recorded separately in
reports. The canonical terminology is defined in
[Mode Comparison Contract](mode_comparison.md).

## Workflow resolution

`resolve_workflow_settings(...)` accepts `reference`, `quality`, or
`diagnostic`.

Resolution follows this order:

1. Select the workflow profile.
2. Apply explicitly supplied voting and skinning configuration.
3. Apply the selected variant's declarative patch.

A supplied `SyntheticVotingConfig`, including its default constructor, is
preserved as supplied. The quality skinning defaults are filled only for fields
that were not explicitly selected by the caller. Scanner configuration is
resolved independently and is never filled by the workflow profile.

Variant definitions live in
`pyosv.evaluation.synthetic_quality.variants`. Promotion thresholds and
coverage requirements live in
`pyosv.evaluation.promotion.specifications`. See
[Architecture](architecture.md) for the ownership boundary.

## Workflow defaults

The effective defaults are:

| Setting | `reference` | `quality` | `diagnostic` |
| --- | --- | --- | --- |
| voter thinning | `reference` | `hybrid_v2` | `reference` |
| surface-support minimum fraction | `0.0` | `0.0` | `0.0` |
| surface-support exponent | `0.0` | `0.0` | `0.0` |
| skinner method | `reference` | `quality` | `reference` |
| skinner minimum likelihood | `0.5` | `None` / adaptive | `0.5` |
| seed planarity threshold | `0.8` | `0.5` | `0.8` |
| skin growth source | `thinned` | `pre_thin` | `thinned` |
| configured accepted-occupancy radius | `None` | `1` | `None` |
| effective accepted-occupancy radius | `5` | `1` | `5` |
| boundary skinner fallback | disabled | enabled | disabled |
| boundary fallback policy | `empty_primary` | `empty_primary` | `empty_primary` |
| thinning diagnostics | when requested | when requested | enabled |

The workflow profiles leave surface voting on
`surface_voting_boundary_policy="reference"`. In that policy, UVW samples are
rounded with Java-style nearest rounding and clamped for local cost sampling,
while surface vote averaging and accumulation require `i2` and `i3` source
samples to be interior. The `masked_in_bounds` policy is available only through
an explicit voter configuration or variant.

### Reference workflow

The `reference` workflow selects the reference-oriented downstream path. It
uses reference-like voter thinning, the reference skinner, fixed
`min_likelihood=0.5`, growth from the thinned vote volume, and no boundary
fallback.

### Quality workflow

The `quality` workflow selects `hybrid_v2` voter thinning and the quality
skinner. When no explicit minimum likelihood is supplied, the quality skinner
uses its adaptive seed threshold and keeps the quality growth threshold
separate. It grows from the pre-thin vote volume, uses an accepted-occupancy
radius of `1`, and enables the `empty_primary` boundary fallback.

The `empty_primary` fallback runs only when primary skinning returns no skins
and the thinned vote volume contains positive samples. It groups the positive
`fvt` support through the fallback implementation and records whether fallback
was enabled, used, and why it ran.

The quality workflow is a downstream synthetic-evaluation profile. It does not
select the quality scanner backend, and selecting the quality scanner backend
does not select this workflow.

The CLI options remain authoritative when explicitly supplied. For example:

```bash
--no-skinner-boundary-fallback
--skinner-boundary-fallback
--skinner-method reference
--skinner-min-likelihood 0.6
--skinner-growth-source thinned
--skinner-accepted-occupancy-radius 3
--voter-thin-mode reference
```

### Diagnostic workflow

The `diagnostic` workflow uses the reference workflow's voting and skinning
defaults and enables thinning diagnostics. It is intended for stage-level
comparisons, not as a scanner selection or an additional scanner/workflow cell.

## Scanner and input independence

`SyntheticScannerConfig` has its own configuration contract. Its defaults are:

| Setting | Default |
| --- | --- |
| backend | `reference-like` |
| strike range | `0.0` to `180.0` degrees |
| dip range | `45.0` to `90.0` degrees |
| `sigma1`, `sigma2` | `2.0`, `2.0` |
| refinement factor | `2` |
| scanner thinning | `reference` |
| requested scanner edge cleanup | `true` |

The refinement factor is used by the `quality` scanner backend. It does not
change a `reference-like` scan merely because it is recorded in the scanner
configuration.

Scanner thinning accepts `none`, `reference`, or `normal`.
`remove_edge_effects` is effective only for `reference` scanner thinning. For
`normal` or `none`, reports use `effective_remove_edge_effects=null` to mean
that the setting is not applicable.

Synthetic input modes are:

- `oracle`: bypasses scanning and evaluates downstream stages from
  truth-derived attributes;
- `scanner`: evaluates the selected scanner and all downstream stages;
- `both`: evaluates both paths on the same truth geometry.

For scanner-inclusive evaluation, pair workflow and scanner settings
explicitly:

```bash
--input-mode scanner \
--workflow-mode quality \
--scanner-backend quality \
--scanner-refinement-factor 2
```

## Variant registry

`current_default` applies no variant patch and therefore represents the
resolved workflow configuration. It is the only member of the `default`
preset.

The `quality-matrix` preset contains:

- `current_default`
- `no_surface_orientation_smoothing`
- `final_norm_smoothing_1`
- `voter_thin_normal`
- `voter_thin_hybrid`
- `voter_thin_hybrid_v2`
- `voter_thin_normal_plateau`
- `surface_support_weighted`
- `quality_skinner_v2`
- `quality_boundary_skinner_fallback`
- `quality_boundary_skinner_fallback_v2`
- `quality_boundary_skinner_fallback_v3`
- `quality_boundary_skinner_fallback_v4`

The current variant patches are:

| Variant | Effective patch |
| --- | --- |
| `current_default` | No patch; use the resolved workflow settings. |
| `boundary_aware_voter_v1` | Set surface-voting boundary policy to `masked_in_bounds`. |
| `no_surface_orientation_smoothing` | Set surface-orientation smoothing to `0.0`. |
| `final_norm_smoothing_1` | Set final vote-map normalization smoothing to `1.0`. |
| `voter_thin_normal` | Select `normal` voter thinning. |
| `voter_thin_hybrid` | Select `hybrid` voter thinning. |
| `voter_thin_hybrid_v2` | Select `hybrid_v2` voter thinning. |
| `voter_thin_hybrid_v2_recenter_scanner_target` | Select `hybrid_v2`, then recenter the configured edge-shell output toward the scanner target. |
| `boundary_edge_thin_v1` | Select `hybrid_v2` with boundary-target-aware edge thinning. |
| `boundary_seed_retention_v1` | Add the boundary seed-retention policy. |
| `voter_thin_normal_plateau` | Select plateau-aware fault-normal thinning. |
| `surface_support_weighted` | Set support minimum fraction to `0.5` and exponent to `1.0`. |
| `quality_skinner_v2` | Select quality skinning, adaptive minimum likelihood, pre-thin growth, and occupancy radius `1`. |
| `quality_boundary_skinner_fallback` | Enable the configured boundary fallback policy. |
| `quality_boundary_skinner_fallback_v2` | Enable `degraded_primary` fallback. |
| `quality_boundary_skinner_fallback_v3` | Enable `degraded_primary_filtered` fallback. |
| `quality_boundary_skinner_fallback_v4` | Enable `degraded_primary_skeletonized` fallback. |
| `quality_boundary_skinner_fallback_v5` | Apply the quality skinner settings and enable `degraded_primary_topology_guarded` fallback. |

The following variants are explicit-only and are not members of
`quality-matrix`:

- `boundary_aware_voter_v1`
- `voter_thin_hybrid_v2_recenter_scanner_target`
- `boundary_edge_thin_v1`
- `boundary_seed_retention_v1`
- `quality_boundary_skinner_fallback_v5`

Version suffixes in these names are current machine-facing identifiers. Their
presence does not imply that a variant is selected by default.

## Boundary-aware surface voting

`boundary_aware_voter_v1` selects
`surface_voting_boundary_policy="masked_in_bounds"` without changing the
selected scanner, scanner thinning, voter thinning, or skinning settings.

The masked policy:

- marks a UVW lag valid only when the lag is admissible and its Java-rounded
  global sample lies inside the volume;
- never clamps an out-of-volume lag into valid evidence;
- selects the deterministic maximum all-supported tangential rectangle that
  contains the local origin;
- carries explicit full-box offsets when a rectangle is cropped;
- excludes invalid states from smoothing, accumulation, and backtracking;
- revalidates mask membership and strain after surface smoothing;
- performs deterministic global feasibility recovery when the smoothed surface
  is not jointly feasible;
- scores only valid selected samples and normalizes support against the full
  tangential patch;
- permits center votes on all six volume faces with bounds-checked
  reinforcement writes.

A full tangential box uses the extracted surface orientation. A cropped box
uses the seed orientation and records
`orientation_source="seed_boundary_fallback"`.

Per-seed diagnostics include selected support, valid-lag counts, projection
counts, invalid selected samples, face center votes, orientation source, and
skip reason. `surface_projection_count` counts `(w, v)` columns whose value
changes between the raw smoothed surface and the final feasible surface, once
per column. It is zero when surface smoothing is disabled.

See [3D Voting Conventions](3d_voting.md) for the complete masked voting
contract.

## Synthetic quality reports

A diagnostic matrix over oracle and scanner inputs can be generated with:

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

A quality-workflow report using only the resolved default is:

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

Scanner-inclusive evaluation must select its scanner explicitly:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 49,49,49 \
  --workflow-mode quality \
  --variants current_default \
  --input-mode scanner \
  --scanner-backend quality \
  --scanner-refinement-factor 2 \
  --scanner-downstream-diagnostics \
  --scanner-boundary-stage-diagnostics \
  --output-dir outputs/3d/synthetic_quality/scanner_quality_current_49 \
  --pretty
```

Review `summary.csv` for comparable scalar rows, then use `metrics.json` and
saved figures for stage-level detail. Reports record scanner, workflow, voting,
thinning, and skinning configuration separately. Skinning configuration records
method, adaptive-threshold state, seed planarity threshold, growth source,
configured and effective occupancy radii, fallback enablement, and fallback
policy.

`--scanner-backend-matrix` is available with `--input-mode scanner` or
`--input-mode both`. It evaluates the `reference-like`, `quality`, and `fast`
scanner backends under the selected downstream configuration.

`--scanner-backend ensemble` is an explicit diagnostic backend. It normalizes
each component likelihood, applies fixed component priors, applies the quality
confidence map as an additional quality weight, and selects one backend's
`ft/pt/tt` at each voxel. Reports record component metadata and selection
fractions. The ensemble backend is not selected by any workflow profile.

## Report comparison profiles

The default `variant` comparison profile compares variant rows. Scanner
thinning policy comparisons use paired `summary.csv` and `metrics.json` files
from separately generated reports.

The supported scanner-policy profiles are:

| Comparison profile | Scanner backend | Baseline policy ID | Candidate policy ID | Promotion gate |
| --- | --- | --- | --- | --- |
| `scanner-thinning-policy-v1` | `quality` | `quality_scanner_reference_v1` | `quality_scanner_thin_normal_v1` | `scanner-boundary` |
| `quality-workflow-scanner-thinning-v1` | `reference-like` | `quality_reference_like_scanner_thin_reference_v1` | `quality_reference_like_scanner_thin_normal_v1` | `scanner-boundary-reference-like` |

Both profiles compare `current_default` under the `quality` workflow on the
`extended` case set at shape `(49, 49, 49)` with `input_mode="both"`. Baseline
and candidate configuration must match recursively except for:

```text
config.scanner.scanner_thin_mode
```

The required direction is `reference` to `normal`. Requested edge cleanup
remains `true` in both reports. Its effective value is `true` for reference
thinning and `null` for normal thinning because normal thinning has no edge
cleanup stage.

For these profiles, comparison first regenerates the canonical summary CSV v1
from each `metrics.json`. It requires:

- exact summary header equality;
- the same complete data rows as a multiset;
- no missing, extra, or duplicate rows;
- a supported `format_version=1` metrics report;
- the selected variant to be present in `config.variants`;
- a passing scanner-policy configuration contract.

Data-row order may differ. Evidence mismatch is an input error and prevents a
numeric promotion artifact from being written.

### Reference-like scanner thinning comparison

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

Compare the paired reports:

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

The quality-scanner profile uses the same procedure with
`--scanner-backend quality`, comparison profile
`scanner-thinning-policy-v1`, and promotion gate `scanner-boundary`.

## Promotion gate contract

Both scanner-boundary gates require the complete extended-case coverage for
scanner and oracle rows at shape `(49, 49, 49)`. Boundary requirements are:

| Requirement | Limit |
| --- | ---: |
| boundary skin buffered F1 | at least `0.90` |
| boundary skin count | at most `3` |
| skin cells / positive FVT candidates | from `0.75` through `1.25` |
| changed boundary positive-FVT buffered F1 | at least `0.90` |
| changed boundary positive-FVT distance p95 | at most `2.0` |

Material-regression limits for applicable non-boundary rows are:

- skin buffered F1 delta must be at least `-0.02`;
- positive-FVT buffered F1 delta must be at least `-0.02`;
- skin distance-p95 delta must be at most `2.0`;
- positive-FVT distance-p95 delta must be at most `2.0`.

Coverage also includes false-fallback checks and component-aware topology checks
for `parallel_planes` and `crossing_planes`.

`scanner-boundary-reference-like` additionally requires:

- comparison profile `quality-workflow-scanner-thinning-v1`;
- unchanged oracle metrics;
- the boundary positive-FVT checks even when FVT counts do not otherwise signal
  a changed row;
- no increase in false fallback replacement;
- no increase in required over-merge or over-split counts.

A gate result evaluates only the compared configuration. It does not alter a
workflow, scanner, thinning, or API default.

## CI regression guardrails

The always-run quality workflow regression suite uses the `extended` synthetic
case set at shape `(21, 21, 21)` with `current_default`. It evaluates
`reference` and `quality` workflows without F3 data or the external Java
reference.

The suite verifies:

- effective workflow settings are recorded correctly;
- key overlap, distance, orientation, edge, and skin metrics are finite;
- non-boundary scanner-inclusive skins remain nonempty;
- reference and quality workflow differences stay within case-specific
  regression limits;
- boundary positive-FVT support remains nonempty, localized, and free of edge
  false positives under the configured tolerance;
- the quality workflow's `empty_primary` fallback produces a nonempty boundary
  skin and records its diagnostics;
- explicit fallback disablement is preserved and may be overridden by an
  explicit variant patch.

The regression thresholds are broad breakage guards. Promotion gates are the
separate 49-cube comparison contracts described above.

## F3 scope

F3 publication comparison uses the complete `(420, 400, 100)` volume as one
evaluation unit. Public `fl.dat`, `fv.dat`, and `fvt.dat` are comparison targets,
not independent geological truth. Scanner backend and workflow remain separate
axes in the full-volume `RL-REF`, `RL-QUAL`, `Q-REF`, and `Q-QUAL` matrix.

Crop and regional diagnostics are views within the same F3 volume. They are not
statistical replicates and do not replace the full-volume protocol. Use
[F3 3D Reference Data Validation](f3d_validation.md) for the current
full-volume runner, runtime contract, artifact validation, and regional
metrics. Use [F3 Visual Diagnostics](f3d_visual_diagnostics.md) for display and
interpretation rules.

## Related specifications

- [Architecture](architecture.md)
- [Mode Comparison Contract](mode_comparison.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [Synthetic Mode Comparison](synthetic_mode_comparison.md)
- [3D Orientation Scanning](orient3d.md)
- [3D Voting Conventions](3d_voting.md)
- [Reference-Like 3D Thinning](reference_like_thinning.md)
- [Skinning](skinning.md)
- [F3 3D Reference Data Validation](f3d_validation.md)
