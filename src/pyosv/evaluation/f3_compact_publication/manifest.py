"""Internal compatibility names for compact publication manifest operations."""

from pyosv.compact_publication_validation import (
    F3_COMPACT_PUBLICATION_MANIFEST_SCHEMA,
    PUBLICATION_MANIFEST_FILENAME,
    build_manifest,
    compute_publication_id,
    validate_compact_publication,
    validate_manifest,
    write_manifest,
)

validate_publication_directory = validate_compact_publication

__all__ = [
    "F3_COMPACT_PUBLICATION_MANIFEST_SCHEMA",
    "PUBLICATION_MANIFEST_FILENAME",
    "build_manifest",
    "compute_publication_id",
    "validate_manifest",
    "validate_publication_directory",
    "write_manifest",
]
