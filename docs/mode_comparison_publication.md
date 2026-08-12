# Mode Comparison Publication Bundle

The publication command is a derived report generator, not an experiment
runner. It accepts a completed and validated synthetic mode-comparison bundle,
a completed and validated F3 full-volume comparison bundle, and the external F3
data root. It reads recorded scalar evidence and validated stage artifacts; it
does not rerun scanner, voting, thinning, or skinning stages.

Synthetic figures use known-truth metrics and paired contrasts. F3 figures use
public-reference agreement terminology. The public `fl.dat`, `fv.dat`, and
`fvt.dat` files are comparison targets rather than geological ground truth.
The F3 data root is required because the F3 source bundle records checksums and
file-role mappings for those read-only public reference volumes. The generator
checks dataset ID, shape, big-endian float32 storage, required roles, file
sizes, and SHA-256 values before it creates any figure.

## Fixed selection and visualization contracts

The curated metric selection is immutable and is derived from the existing
synthetic and F3 metric registries. Synthetic truth semantics and F3
public-reference-agreement semantics remain separate tables and are never
combined into a score.

The four condition order is always `RL-REF`, `RL-QUAL`, `Q-REF`, `Q-QUAL`.
Synthetic `scanner_raw` evidence keeps the source scanner-only labels
`RL-SCAN` and `Q-SCAN` as auxiliary rows; they are not mixed into the
canonical workflow-cell figures.
F3 stage order is `ft`, `fv`, `fvt`. Spatial slice selection is deterministic:
center index, public-reference positive-p99 ridge-count maximum, or the
positive difference peak for `abs(Q-QUAL - RL-REF)` on `fvt`; ties choose the
smallest index. The positive-p99/radius-2 threshold metadata is taken directly
from validated F3 `MetricEvidence` with selection `positive_p99_radius2`; it is
not recalculated by publication generation. Each stage records one shared
public-reference threshold and one candidate threshold for each canonical cell.
`public_reference_peak` uses only the public reference and that stage's
public-reference threshold, so candidate values and candidate thresholds cannot
change its selected slice. `end_to_end_difference_peak` remains the
threshold-free `abs(Q-QUAL - RL-REF)` policy.

For F3 ridge overlays, `selection_threshold` always means the public-reference
threshold. The public-reference mask uses it, while each candidate mask uses
that candidate cell's own source-recorded p99 threshold. Thus exact overlap,
buffered match, public-reference-only, candidate-only, and
`exact_overlap_count` use the same separate reference/candidate mask contract.
The percentile remains 99 and the buffer radius remains 2. Normal panels in
one spatial figure share a scale derived from validated full-volume min/max
evidence. Signed difference panels use a separate finite range centered on
zero. The signed differences themselves are formed only after reading each
selected 2-D slice; publication generation does not materialize a full-volume
candidate-minus-reference array.

Synthetic source bundles are scalar-only, so synthetic figures are metric,
contrast, and runtime figures. Synthetic spatial replay is not performed.
F3 spatial figures use read-only memory maps of the existing stage DAT files
and public reference files. Public DAT files are never copied to the output.

## Output and validation

The output directory contains `publication_manifest.json`, `experiment.json`,
the supplied environment lock, the publication CSV tables, `report.md`,
`figure_data/`, and `figures/`. `publication_manifest.json` is the only root
management and completion file. It records code and environment provenance,
F3 dataset identity, source completion hashes, Synthetic/F3 evaluation
semantics, and each artifact's path, tier, role, size, and SHA-256.

Validation is deliberately bundle-local. It validates the manifest contract
and publication identity, checks every listed artifact's regular-file status,
size, and SHA-256, binds the environment lock and `experiment.json` hashes to
their manifest sections, and rejects unlisted regular files. It does not parse
CSV semantics or PNG dimensions and does not access source bundles, the F3 data
root, Git, Matplotlib, Numba, or BLAS.

`experiment.json` is a deterministic snapshot of the resolved Synthetic and F3
plans, case/trial order, stage order, selected metric keys, and slice-selection
policy. Source paths, host/runtime diagnostics, and artifact hashes are not
duplicated into that snapshot.

The directory is built privately, validated internally, and renamed atomically
only after completion; an existing output directory is an error. Source bundles
and the external F3 data root remain unchanged.

Validate an existing output without source bundles, F3 data access, or
visualization imports:

```bash
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --output-dir outputs/3d/mode_comparison_publication/publication_v1 \
  --validate-only
```

Generate the fixed report:

```bash
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --synthetic-bundle <completed-synthetic-bundle> \
  --f3-bundle <completed-f3-bundle> \
  --f3-data-root "$PYOSV_F3D_DATA_ROOT" \
  --environment-lock uv.lock \
  --output-dir outputs/3d/mode_comparison_publication/publication_v1
```

Figures require the optional visualization extra:

```bash
python -m pip install -e ".[viz]"
```

## Display conventions

The synthetic scanner-orientation figure uses color for the metric (strike or
dip median error) and marker shape for the scanner backend (`RL-SCAN` or
`Q-SCAN`), with both encodings present in the legend and preserved in the
figure-data CSV. The F3 sparse-distance figure retains condition color while
using an explicit hatch for distance direction: candidate-to-reference and
reference-to-candidate p95 are separately identified in its legend and figure
metadata. Regional display units classify overlap fractions (`_fraction`,
`_precision`, `_recall`, `_f1`, `_jaccard`) as fractions, counts as counts,
distance metrics as voxels, normalized correlation as correlation, and ratios
as ratios. In particular, `nonzero_fraction_ratio` is not displayed as a
fraction because it can exceed one.

Related contracts: [mode comparison terminology](mode_comparison.md),
[synthetic comparison](synthetic_mode_comparison.md),
[F3 validation](f3d_validation.md), and
[visualization helpers](visualization.md).
