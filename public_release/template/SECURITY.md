# Security

## Status

PyOSV Q-QUAL PoC is proof-of-concept software and has no security-support SLA.

## Untrusted files

Run the software with least privilege when handling untrusted DAT files or
compact publication archives. Inspect archive provenance before extraction,
avoid extracting into sensitive locations, and verify the supplied SHA-256
before validation. The Q-QUAL input reader requires a regular non-symlink file,
and the compact validator rejects symlinks and unsafe relative paths in an
extracted bundle.

## Reporting

Report security concerns through this repository's private vulnerability
reporting channel. If private reporting is unavailable, open a minimal public
issue without sensitive details and request a private communication channel.
