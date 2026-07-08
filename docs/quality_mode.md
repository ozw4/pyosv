# Quality Mode

This repository separates reference-alignment checks from controlled truth
quality experiments.

## Workflows

`reference` workflow is for origin-aligned regression comparison. Its defaults
keep reference-like voter thinning and disable support-aware surface voting
(`surface_support_min_fraction=0.0`,
`surface_support_exponent=0.0`). Use it when checking that Python behavior
remains close to the current reference-oriented path. It is not the place to
evaluate processing-quality improvements.

`quality` workflow is the current quality-first synthetic profile. Its defaults
use hybrid voter thinning, disable support-aware surface voting, and
`FaultSkinner(method="quality")`
(`surface_support_min_fraction=0.0`,
`surface_support_exponent=0.0`). It is not a universal production profile; use
it while reviewing the controlled synthetic benchmark matrix and checking that
the candidate set is not over-filtered for the cases under study.

`diagnostic` workflow keeps the reference workflow defaults and enables
thinning diagnostics. Use it when comparing current behavior, diagnostic
variants, and reference-vs-normal thinning on the same synthetic truth.

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

For scanner-inclusive quality evaluation, prefer the refined opt-in scanner
path:

```bash
--input-mode scanner \
--scanner-backend quality \
--scanner-refinement-factor 2
```

This recommendation is for quality reports only. The report default remains
`--scanner-backend reference-like` so reference-oriented scanner behavior is not
changed automatically.

The `quality-matrix` preset includes `current_default`,
`no_surface_orientation_smoothing`, `final_norm_smoothing_1`,
`voter_thin_normal`, `voter_thin_hybrid`, and
`surface_support_weighted`, and `quality_skinner_v2`. The hybrid voter thinning
variant uses reference-like thinning in stable-orientation regions and
fault-normal thinning where local orientation changes rapidly.

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
the workflow, but uses the quality skinner with adaptive seed/grow thresholds,
`growth_source=pre_thin`, and `accepted_occupancy_radius=1`. It is a diagnostic
candidate and does not change `current_default` or workflow defaults.

For skin extraction, `--workflow-mode quality` defaults to
`--skinner-method quality` unless `--skinner-method` is passed explicitly. The
quality skinner reuses reference-like skin growth and reskinning, but uses
adaptive `min_likelihood` when `--skinner-min-likelihood` is omitted and lowers
the seed planarity gate from `ep > 0.8` to `ep > 0.5`. The adaptive threshold is
used for seed selection while grow-time likelihood gating remains separately
bounded, so quality mode does not raise the grow threshold just because it
chooses stricter seeds. Synthetic reports record the selected
`skinning.method`, whether the likelihood threshold is adaptive, the seed `ep`
threshold, and `seed_planarity_source=fvt` in `metrics.json`.

Primary metrics to compare:

- `fvt_buffered_f1_r2`
- `fvt_distance_candidate_to_truth_p95`
- `fvt_strike_median_error`
- `fvt_dip_median_error`
- `skin_buffered_f1_r2`
- `skin_distance_candidate_to_truth_p95`
- `edge_false_positive_fraction` columns

## CI Regression Guardrails

The always-on quality workflow regression test is intentionally synthetic-only:
it does not require F3 data, `reference_osv`, Java/Jython/JTK, or external
downloads. It builds the `extended` synthetic case set at a small shape for both
`reference` and `quality` workflows using only `current_default`.

The guardrails are broad. They assert that key overlap, distance, orientation,
edge false-positive, and skin metrics remain finite, that quality workflow
effective settings are recorded in `metrics.json`, and that quality mode has not
clearly regressed relative to reference mode on representative curved,
vertical, and boundary cases. These thresholds are not benchmark targets. They
are meant to catch obvious workflow breakage while leaving room for normal
tuning changes.

## F3 Real-Data Workflow Comparison

The F3 multi-crop report can also compare `reference` and `quality` workflows
on the same real-data crop centers:

```bash
PYTHONPATH=src python examples/report_3d_f3d_multicrop.py \
  --data-root "$PYOSV_F3D_DATA_ROOT" \
  --count 2 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --compare-workflows \
  --output-json outputs/3d/f3d/quality_compare_001/metrics.json \
  --pretty
```

Because F3 has no independent truth labels, reference agreement is a stability
diagnostic rather than direct evidence of higher quality. Review whether the
quality workflow preserves reference geological signal, avoids extra ridges and
boundary artifacts, and keeps crop-to-crop behavior stable.

F3 reference agreement is not quality itself. A quality-mode change should be
promoted only when the controlled synthetic `extended` matrix and F3 multi-crop
comparison both show fewer clear failures than the reference workflow: improved
or preserved synthetic truth recovery, no new boundary or over-filtering
failure, and no obvious loss of real-data geological signal across crops. Until
then, keep `reference` as the default API behavior and use `quality` as an
explicit workflow mode.
