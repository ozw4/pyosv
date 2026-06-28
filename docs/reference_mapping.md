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
| `RecursiveGaussianFilterP.java` | `pyosv.filters` | approximate/later |
