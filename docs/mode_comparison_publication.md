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
smallest index. The p99 threshold and radius 2 are taken from the validated F3
metric contract. Normal panels in one spatial figure share a scale derived from
validated full-volume min/max evidence. Signed difference panels use a separate
finite range centered on zero. The signed differences themselves are formed
only after reading each selected 2-D slice; publication generation does not
materialize a full-volume candidate-minus-reference array.

Synthetic source bundles are scalar-only, so synthetic figures are metric,
contrast, and runtime figures. Synthetic spatial replay is not performed.
F3 spatial figures use read-only memory maps of the existing stage DAT files
and public reference files. Public DAT files are never copied to the output.

## Output and validation

The output directory contains `manifest.json`, the publication CSV tables,
`figure_manifest.json`, `report.md`, `figure_data/`, `figures/`, and a final
`completion.json`. Publication artifact schema version 2 introduces a semantic
table contract version 1; figure contract version 2 adds fixed figure-slot
coverage, figure-data semantic metadata, and PNG dimensions. Completion and
metric-selection contracts remain version 1. A version-1 publication artifact
or figure contract is not upgraded implicitly: regenerate the publication bundle.

`completion.json` checks exact file bytes, sizes, and SHA-256 values. In
addition, `manifest.json` records a typed semantic contract for every root CSV:
its header, row count, identity fields, ordered identity digest, and ordered
semantic-row digest. The validator reparses CSV values with their declared
types, so nullable blanks remain distinct from numeric zero and boolean values
remain distinct from integers. It rejects duplicate identities and verifies
canonical ordering, selected metric/contrast coverage, recomputed contrasts,
and recomputed descriptive summaries. Supporting regional, orientation, and
runtime tables have equivalent identity and coverage validation rather than
being allowed to pass as header-only files.

The manifest stores source coverage drawn from the validated source metadata:
synthetic case/trial identities and skinning state, plus the one F3 full-volume
evaluation unit, canonical cells, per-cell skinning state, volume shape,
dataset identity, and run fingerprint. Source identity digests are recomputed
from their recorded internal identity fields during validate-only operation.
They are provenance-consistency checks, not cryptographic signatures or a
tamper-proof commitment: a coherent rewrite of all related metadata is outside
their threat model.

Figure validation requires the fixed scalar and spatial slot set, including an
explicit omitted synthetic-skin record when skinning is disabled. Every
non-omitted figure records its typed figure-data row contract, PNG byte hash,
PNG size, and IHDR width/height. The validator reads the PNG signature and
first IHDR chunk with the standard library and rejects invalid, zero, or
unreasonably large dimensions (over 100,000 pixels in either direction)
without requiring Pillow.

The directory is built privately, validated internally, and renamed atomically
only after completion; an existing output directory is an error. Source bundles
and the external F3 data root remain unchanged.

Validate an existing output without source bundles, F3 data access, or
visualization imports:

```bash
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --output-dir outputs/3d/mode_comparison_publication/publication_v2 \
  --validate-only
```

Generate the fixed report:

```bash
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --synthetic-bundle <completed-synthetic-bundle> \
  --f3-bundle <completed-f3-bundle> \
  --f3-data-root "$PYOSV_F3D_DATA_ROOT" \
  --output-dir outputs/3d/mode_comparison_publication/publication_v2
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
