# Mode Comparison Publication Bundle

The publication command builds a derived report from completed, validated
Synthetic and F3 mode-comparison source bundles. It reads their recorded
evidence and F3 stage artifacts; it does not rerun scanner, voting, thinning,
or skinning stages.

Synthetic results use known-truth metrics and paired contrasts. F3 results use
public-reference-agreement semantics: the public `fl.dat`, `fv.dat`, and
`fvt.dat` volumes are comparison targets, not independent geological truth or
an accuracy label. The publication manifest is a file-integrity and provenance
contract, not a cryptographic signature.

## Generate a bundle

Figure generation requires the optional visualization dependencies:

```bash
python -m pip install -e ".[viz]"
```

Run generation from the repository containing the code being published. The
CLI records the current Git commit and dirty state, reads all eight controls
shown below from the process environment, and copies the existing environment
lock byte-for-byte into the output under its basename.

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
NUMBA_DISABLE_JIT=0 \
NUMBA_NUM_THREADS=1 \
PYOSV_ACCEL=auto \
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --synthetic-bundle <completed-synthetic-bundle> \
  --f3-bundle <completed-f3-bundle> \
  --f3-data-root "$PYOSV_F3D_DATA_ROOT" \
  --environment-lock uv.lock \
  --output-dir outputs/3d/mode_comparison_publication/publication_v1
```

`--environment-lock` must name an existing regular, non-symlink file. The
output directory must not already exist and must be outside both source bundles
and the F3 data root. The generator validates the source bundles and F3 dataset
identity before creating publication artifacts. It builds the output in a
private sibling directory, writes `publication_manifest.json` last, validates
the completed directory, and then renames it to the requested output path.

## Output layout

The generated directory has this layout; the lock filename is the basename of
the path passed to `--environment-lock`.

```text
publication_manifest.json
experiment.json
<environment lock>
publication_metrics.csv
publication_contrasts.csv
publication_summary.csv
f3_regional_summary.csv
f3_orientation_summary.csv
runtime_summary.csv
figure_data/
  *.csv
figures/
  *.png
report.md
```

The public F3 DAT volumes are read from the supplied data root and are not
copied into the publication directory.

## `publication_manifest.json`

`publication_manifest.json` is the sole root management and completion file.
Its top level contains:

- `schema`: the `pyosv.publication_manifest.v1` contract identifier.
- `publication_id`: the SHA-256 identity described below.
- `created_at_utc`: the bundle creation time at UTC second precision.
- `code`: repository, Git commit, and recorded dirty state.
- `environment`: Python version, lock path/hash, and the eight explicit
  environment controls.
- `datasets`: the path-independent F3 dataset identity, including shape,
  storage dtype, file roles, sizes, and SHA-256 values.
- `experiment`: the `experiment.json` path/hash and Synthetic/F3 source
  completion hashes.
- `semantics`: Synthetic known-truth and F3 public-reference-agreement
  meanings, including the explicit statement that the F3 reference is not
  geological truth and that the full volume is one evaluation unit.
- `artifacts`: every published artifact's relative path, tier, role, byte size,
  and SHA-256.

The `publication_id` is computed from the schema, code identity, environment
identity (Python version, lock and controls), F3 dataset identity, experiment
snapshot and source completion hashes, semantics, and all primary artifact
records. It does not include `created_at_utc`, derived report/PNG records, the
output path, mtimes, or CPU/BLAS diagnostics.

This distinction keeps a regenerated presentation from changing the
scientific/provenance identity when its `report.md` or PNG bytes change. Those
derived files are still integrity-checked against their own manifest records;
changing one without rebuilding the manifest fails validation.

## `experiment.json`

`experiment.json` is a deterministic, compact snapshot of the resolved public
experiment, not a copy of either source manifest. It records:

- the canonical condition order `RL-REF`, `RL-QUAL`, `Q-REF`, `Q-QUAL`;
- Synthetic shape, meaningful case order, trial identities and seeds, skinning
  state, and resolved plan;
- F3 dataset ID, shape, big-endian float32 storage dtype, and resolved plan;
- publication stage order, sorted selected metric keys, and the fixed slice
  selection policy.

Paths, timestamps, host/runtime diagnostics, artifact hashes, and unrelated
source cache metadata are not duplicated into this snapshot. The resolved-plan
objects preserve the validated source choices without redefining the complete
source schemas in the publication contract.

## Primary and derived artifacts

Primary artifacts are inputs to `publication_id`:

- the environment lock;
- `experiment.json`;
- metric, contrast, summary, diagnostic, and runtime CSV tables;
- all `figure_data/*.csv` files.

Derived artifacts are reproducible presentations and are excluded from
`publication_id`:

- `report.md`;
- all `figures/*.png` files.

Directory validation checks the existence, regular non-symlink status, size,
and SHA-256 of both tiers. Tier controls scientific identity participation, not
whether a file receives integrity validation.

## Validate only

Validate an existing publication directory with:

```bash
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --validate-only \
  --output-dir outputs/3d/mode_comparison_publication/publication_v1
```

This command needs only the completed publication directory. It does not
require the Synthetic or F3 source bundles, the F3 data root, Matplotlib, a Git
repository, or Numba/BLAS/current-runtime inspection. It validates the manifest
schema and `publication_id`; every recorded artifact's path, regular-file
status, size, and SHA-256; the lock and experiment hash links; and the exact set
of regular files in the directory. It does not parse CSV semantics or PNG
dimensions.

## Reproduction boundary

These are separate operations and claims:

- Publication validate-only checks the integrity and recorded provenance of a
  completed publication bundle.
- Source experiment reproduction reruns the Synthetic and F3 source workflows
  and produces new completed source bundles.
- Scientific generalization repeats the study on another survey or against
  independent geological truth.

Validate-only is not an end-to-end experiment replay and does not establish
scientific generalization.

## Fixed selection and visualization behavior

The curated metrics come from the existing Synthetic and F3 metric registry.
Synthetic truth semantics and F3 public-reference-agreement semantics remain
separate tables and are never combined into one score.

F3 stage order is `ft`, `fv`, `fvt`. Spatial slice selection is deterministic:
center index, public-reference positive-p99 ridge-count maximum, or the
positive difference peak for `abs(Q-QUAL - RL-REF)` on `fvt`; ties select the
smallest index. Threshold metadata is reused from validated F3 evidence and is
not recalculated by publication generation. Synthetic source bundles are
scalar-only, so Synthetic figures are metric, contrast, and runtime figures;
Synthetic spatial replay is not performed.

F3 spatial figures use read-only memory maps of existing stage DAT files and
public reference files. Normal panels in one spatial figure share a scale from
validated full-volume min/max evidence. Signed difference panels use a finite
range centered on zero and are calculated after reading each selected 2-D
slice, without materializing a full-volume candidate-minus-reference array.

Related documentation: [mode comparison terminology](mode_comparison.md),
[Synthetic comparison](synthetic_mode_comparison.md),
[F3 validation](f3d_validation.md), and
[visualization helpers](visualization.md).
