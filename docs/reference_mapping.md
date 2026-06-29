# Reference Mapping

This table records the initial target modules for the Python reimplementation. It is a planning aid, not an implementation commitment for the bootstrap issue.

| Reference file | Python target | Priority |
| --- | --- | --- |
| `FaultGeometry.java` | `pyosv.geometry` | high |
| `FaultCell2.java` | `pyosv.cells.FaultCell2` | high |
| `FaultCell.java` | `pyosv.cells.FaultCell` | medium |
| `OptimalPathVoter.java` | `pyosv.voting2d` | highest |
| `OptimalSurfaceVoter.java` | `pyosv.voting3d` | high |
| `FaultOrientScanner2.java` | `pyosv.orient2d` | medium |
| `FaultOrientScanner3.java` | `pyosv.orient3d` | later |
| `FaultSkin.java` | `pyosv.skin` | later |
| `FaultSkinner.java` | `pyosv.skinner` | later |
| `SincInterpolator` use sites | `pyosv.interp` | approximate |
| `RecursiveExponentialFilter` use sites | `pyosv.filters` | approximate |
| `RecursiveGaussianFilterP.java` | `pyosv.filters` | approximate/later |

## Dynamic-programming path kernel

The 2D dynamic-programming path kernel used by
`OptimalPathVoter.java` is mapped to `pyosv.dp` helpers:

| Reference method | Python helper | Notes |
| --- | --- | --- |
| `findPath` | `find_path_2d` | high-level path extraction with optional attribute and path smoothing |
| `accumulateForward` | `accumulate_forward_2d` / `accumulate_2d(direction=1)` | forward accumulated cost image |
| `backtrackReverse` | `backtrack_reverse_2d` | reverse backtracking through a forward accumulation |

DP cost images use shape `(ni, nl)`, where `ni` is the path direction
sample count and `nl` is the lag-axis sample count. This is a specialized 2D
array under the repository-wide `(n2, n1)` convention. Paths returned by
`find_path_2d` and `backtrack_reverse_2d` have shape `(ni,)` and `float32`
lag values.

## Approximation policy

`pyosv` does not attempt bit-exact reproduction of Mines JTK interpolation or
recursive filters. Reference uses of `SincInterpolator` in scanner and voting
code are mapped to SciPy interpolation primitives, typically
`scipy.ndimage.map_coordinates`. Reference uses of `RecursiveExponentialFilter`
and `RecursiveGaussianFilterP` are mapped to SciPy Gaussian smoothing or explicit
separable smoothing in `pyosv.filters`.

`reference_osv/` remains a read-only bind mount used for comparison and
inspection only. It is not part of the Python package and must not be included in
distribution artifacts.
