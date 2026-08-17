# Q-QUAL operation manual

## Environment setup

Use Python 3.10 or newer in a fresh virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Install optional Numba acceleration with `python -m pip install ".[accel]"`.
NumPy 1.x is the supported NumPy series.

## Input DAT contract

The input must be a regular, non-symlink file containing only finite
big-endian IEEE 754 float32 samples. Storage order is C order. The byte size
must equal `n3 * n2 * n1 * 4`.

## Shape order

Supply the shape as `(n3, n2, n1)`. On the command line it is written without
parentheses, for example `--shape 420,400,100`.

## Q-QUAL CLI

The output directory must not already exist, and its parent must exist:

```bash
pyosv-qqual3d \
  --input /path/to/ep.dat \
  --shape 420,400,100 \
  --output-dir /path/to/new-qqual-output \
  --pretty
```

`--pretty` changes only the formatting of `run.json`. The command has no
scanner, workflow, threshold, or variant selector.

The public profile is resolved by `resolve_qqual3d_profile()` and is fixed to
the quality scanner and quality workflow. Scanner thinning is `reference`,
voter thinning is `hybrid_v2`, and quality skinning is enabled. These values
are recorded with all resolved scanner, voting, and skinning controls in
`run.json`; that profile record is the authoritative per-run settings record.

## Output layout

A default run produces exactly these files:

```text
run.json
ft.dat
fv.dat
fvt.dat
skin_mask.dat
skins.json
```

DAT outputs are C-order big-endian float32 arrays with the input shape. Files
are first written into a private sibling directory. The completed directory is
renamed to the requested path only after every artifact and `run.json` has
been written. A failed run removes its temporary directory.

## Result volumes and skins

- `ft.dat` is the quality-scanner fault-likelihood volume before scanner-stage
  thinning is used as workflow input.
- `fv.dat` is the voted likelihood.
- `fvt.dat` is the voter-thinned ridge volume.
- `skin_mask.dat` marks voxels belonging to returned fault skins.
- `skins.json` records each returned skin and its cells.

`fv` retains the voting response, while `fvt` keeps the ridges selected by
voter thinning. They are different processing stages and should not be treated
as interchangeable attributes.

## run.json

`run.json` uses schema `pyosv.qqual3d.run/v1`. It records:

- the input filename without its absolute path, shape, storage dtype, byte
  size, and SHA-256;
- the complete resolved fixed Q-QUAL profile;
- Python, PyOSV, NumPy, SciPy, and Numba availability/version information;
- each output's role, filename, shape or JSON role, byte size, and SHA-256.

The recorded hashes allow consumers to detect changes to the input identity
and generated artifacts after the run.

## Compact bundle validation

Verify the release archive checksum before extracting it. Then validate the
single extracted publication directory:

```bash
sha256sum -c compact-publication.tar.gz.sha256
pyosv-validate-compact /path/to/extracted/compact-publication
```

The validator checks manifest identity, artifact sizes and hashes, safe paths,
regular non-symlink files, the exact file set, and provenance links. It does
not regenerate the publication and does not require the numerical stack.

## Troubleshooting

- **Input size mismatch:** confirm shape order and the four-byte `>f4` storage
  contract.
- **Output already exists:** choose a new path; the command never overwrites an
  existing filesystem entry.
- **Non-finite input:** replace or reject NaN and infinity values before the
  run.
- **Archive validation failure:** verify the release checksum, extract into a
  new directory, and ensure no artifact was added, removed, or changed.
- **Slow execution:** install the optional acceleration extra and verify Numba
  availability in `run.json`.

See the [README](../README.md) for scope and the
[comparison guide](public_reference_comparison.md) for interpretation.
