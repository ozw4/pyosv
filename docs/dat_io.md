# DAT I/O

`pyosv.io` provides small helpers for the raw binary `.dat` files used by the
reference OSV data. The helpers do not parse headers; callers provide the target
shape and endian convention explicitly or through reference dataset metadata.

## Shape and dtype conventions

- 2D arrays use shape `(n2, n1)`.
- 3D arrays use shape `(n3, n2, n1)`.
- The default dtype is `np.float32`.
- `reference_osv` `.dat` files are treated as big-endian `float32` by default.

## Reading

`read_dat(path, shape, *, endian="big", dtype=np.float32)` reads raw binary
scalar values, validates that `shape` is a non-empty tuple of positive integers,
checks the file size, reshapes the values in C order, and returns a
C-contiguous array with native-byte-order dtype.

Example:

```python
from pyosv.io import read_dat
from pyosv.reference import REFERENCE_DATASETS_2D, resolve_reference_file

dataset = REFERENCE_DATASETS_2D["f3d2d"]
path = resolve_reference_file(dataset, "ft.dat")
ft = read_dat(path, dataset.shape, endian=dataset.endian)
```

If the file size does not exactly match `prod(shape) * dtype.itemsize`,
`read_dat` raises `ValueError` and includes the expected and actual byte counts.
Invalid shapes also raise `ValueError`.

## Writing

`write_dat(path, array, *, endian="big", dtype=np.float32, create_parents=True)`
writes an array as raw binary scalar values in C order. It converts the output
to the requested storage dtype and endian, creates parent directories by
default, writes the file with `numpy.ndarray.tofile`, and returns the written
`Path`.

## Endian values

Accepted endian values are:

- `"big"` or `">"` for big-endian storage
- `"little"` or `"<"` for little-endian storage

Unknown endian values raise `ValueError`.

## Reference data policy

`reference_osv/` is a read-only bind mount and is not committed. Do not add
reference binary `.dat` files or generated fixtures to this repository.

By default, reference paths resolve under `./reference_osv`. Set
`PYOSV_REFERENCE_OSV=/absolute/path/to/osv-master` when the reference checkout
or bind mount is elsewhere.

Optional smoke tests for reference fixture I/O skip when the reference root does
not exist. Individual cases also skip when a required reference `.dat` file is
missing from an otherwise available mount.
