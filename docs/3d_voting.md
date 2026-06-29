# 3D Voting Conventions

`pyosv` stores global 3D image volumes with shape `(n3, n2, n1)`. Sample
indices are addressed as `(i3, i2, i1)` in array indexing order, while vector
components follow the OSV component order `(x1, x2, x3)`.

Local 3D voting samples use a seed-centered `(w, v, u)` coordinate system with
array shape `(nw, nv, nu)`. The local axes are:

- `u`: fault-normal lag direction.
- `v`: fault-dip direction.
- `w`: fault-strike direction.

For orientation vectors, `u` is the fault normal, `v` is the dip vector, and
`w` is the strike vector. The component arrays for these vectors use
`(x1, x2, x3)` order even when they are sampled from global volumes stored as
`(n3, n2, n1)`.

`reference_osv/` is a read-only reference implementation. It is not a runtime
dependency of `pyosv`, and generated outputs should not be written under that
directory.

## Minimal Usage

`OptimalSurfaceVoter.apply_voting` runs the current 3D voting MVP on fault
likelihood, strike, and dip volumes:

```python
import numpy as np

from pyosv.voting3d import OptimalSurfaceVoter

ft = np.zeros((64, 96, 128), dtype=np.float32)
pt = np.zeros_like(ft)
tt = np.full_like(ft, 90.0)

voter = OptimalSurfaceVoter(ru=6, rv=8, rw=8)
voter.set_strain_max(0.25, 0.25)
fv, vp, vt = voter.apply_voting(d=4, fm=0.3, ft=ft, pt=pt, tt=tt)
```

The returned `fv`, `vp`, and `vt` arrays have the same `(n3, n2, n1)` shape as
the inputs. `fv` is a normalized `float32` vote volume in `[0, 1]`; `vp` and
`vt` store the strike and dip angles associated with the strongest local vote at
each sample.

## MVP Limitations

This is the 3D voting MVP, not the complete Java 3D fault interpretation
pipeline. The implementation is sequential and currently covered by synthetic
regression tests only. It does not include 3D thinning.

`FaultOrientScanner3` and `FaultSkinner` equivalents are later milestones, so
current 3D workflows must provide `ft`, `pt`, and `tt` volumes directly.
