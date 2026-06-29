from pathlib import Path

import numpy as np
import pytest

from pyosv.io import read_dat
from pyosv.reference import (
    REFERENCE_DATASETS_2D,
    ReferenceDataset,
    reference_root,
    resolve_reference_file,
)


def require_reference_root() -> Path:
    root = reference_root()
    if not root.exists():
        pytest.skip(f"reference_osv mount not available: {root}")
    return root


def require_reference_2d_file(dataset: ReferenceDataset, file_name: str) -> Path:
    path = resolve_reference_file(dataset, file_name, root=require_reference_root())
    if not path.exists():
        pytest.skip(f"reference fixture not available: {path}")
    return path


def load_reference_2d_array(dataset_name: str, file_name: str) -> np.ndarray:
    dataset = REFERENCE_DATASETS_2D[dataset_name]
    path = require_reference_2d_file(dataset, file_name)
    return read_dat(path, dataset.shape, endian=dataset.endian)
