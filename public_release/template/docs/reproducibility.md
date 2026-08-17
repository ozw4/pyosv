# Reproducibility and data boundary

## Publicly reproducible

The public snapshot supports these operations:

- install the Q-QUAL distribution in a fresh Python environment;
- run fixed Q-QUAL on an authorized big-endian float32 input volume;
- verify generated output identities using `run.json` sizes and SHA-256 values;
- validate an extracted compact publication using the standalone validator;
- verify the Release archive against its SHA-256 file.

The compact validator verifies an existing publication. It does not recreate
its figures or metrics.

## Not included

The public distribution and Release assets do not include raw F3 DAT volumes.
The public source snapshot does not include the internal four-cell comparison
runner, the full Synthetic evaluation, or the full publication generator.

## Provenance

- Public export source identity: recorded in `SOURCE_SNAPSHOT.json`.
- Compact generation commit/build identifier:
  `47f81b72a7bfab3ce259b821548ad8e6156e74cb`
- Compact publication ID:
  `c20a3a4195fb5598a9661d16cf368610ba7081c28ece2e63549706fec6a35322`
- Compact archive SHA-256:
  `872fa183e1016b70ccd41449e689b545adb49e70cb29e563bf6f35133834c13d`
- Source F3 completion SHA-256:
  `3cc8818b27c9ea68d7fc4f5c9fc8d072aaaeb81cfd672c1c79d54c9fe8c1ae72`

The exact internal repository commit used to export the public source snapshot
is recorded in `SOURCE_SNAPSHOT.json`. This file is the sole authority for the
export source identity, even when the corresponding internal repository is not
publicly accessible.

The public repository commit is identified by the Git tag and Release notes.
It is not embedded in this tracked document.

## Verification

After obtaining the two Release assets, verify and extract the archive:

```bash
sha256sum -c compact-publication.tar.gz.sha256
tar -xzf compact-publication.tar.gz
pyosv-validate-compact compact-publication
```

The actual asset basename and extracted top-level directory are recorded by
the Release. See the [operation manual](manual.md) for validator behavior and
[comparison interpretation](public_reference_comparison.md) for evidence
semantics.
