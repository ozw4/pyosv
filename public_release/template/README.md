# PyOSV Q-QUAL PoC

PyOSV Q-QUAL PoC is a focused Python distribution for running the fixed Q-QUAL
3D fault-interpretation workflow and validating its compact F3 publication
bundle. It provides an in-memory library API and two command-line entry points
without exposing evaluation-mode selectors.

The canonical source repository is
<https://github.com/ozw4/pyosv-qqual-poc>.

## Scope

The distribution contains the Q-QUAL runtime, its required PyOSV dependency
closure, the compact-publication validator, examples, and focused tests. The
fixed runtime uses the quality scanner, quality voter thinning, and quality
skinning. It does not include raw F3 DAT volumes, the internal comparison
runner, the full Synthetic evaluation, or the full publication generator.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Optional Numba acceleration can be installed with `python -m pip install
".[accel]"`.

## Five-minute quick start

Prepare a regular, non-symlink, C-order DAT file containing big-endian float32
samples. Supply its shape in `(n3, n2, n1)` order and choose an output directory
that does not exist:

```bash
pyosv-qqual3d \
  --input /path/to/ep.dat \
  --shape 420,400,100 \
  --output-dir qqual-output \
  --pretty
```

The command prints the new output directory after a successful atomic write.
The input file is not modified.

To validate an extracted compact publication:

```bash
pyosv-validate-compact /path/to/extracted/compact-publication
```

The release assets for `v0.1.0-poc` provide the compact publication archive and
its SHA-256 file. The archive is not part of the Git tree.

## Output bundle

The default Q-QUAL run writes:

- `run.json`: input identity, resolved fixed profile, software versions, and
  output size and SHA-256 records.
- `ft.dat`: quality-scanner fault likelihood.
- `fv.dat`: voted likelihood.
- `fvt.dat`: voter-thinned ridge volume.
- `skin_mask.dat`: mask of cells in the returned skins.
- `skins.json`: serialized fault-skin cells.

DAT outputs use big-endian float32 storage and the input shape. See the
[operation manual](docs/manual.md) for the complete file contract.

## Compact publication

The compact publication contains six atlases, their six machine-readable data
tables, a summary table, provenance, and a manifest. It reports agreement with
the F3 public reference; that reference is not geological truth. See
[PUBLIC-REF comparison and interpretation](docs/public_reference_comparison.md)
and [reproducibility boundaries](docs/reproducibility.md).

## Limitations

This is a proof of concept. Runtime and memory use scale with the input volume,
and results require domain interpretation. Only the fixed Q-QUAL workflow is a
public runtime. Users must supply input data they are authorized to use.

## License, attribution, and citation

The source code is provided under the Common Public License Version 1.0
(`CPL-1.0`); see [LICENSE](LICENSE). Upstream implementation attribution is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and data rights are addressed
separately in [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). Citation metadata is
available in [CITATION.cff](CITATION.cff).
