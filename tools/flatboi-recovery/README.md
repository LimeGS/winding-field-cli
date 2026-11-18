# Flatboi recovery: reproducible native build and corrections

The shipped SLIM binary (`/srv/jobs/175A-slim-ct-trial-20261113/runtime/usr/bin/flatboi`,
sha256 `d724cf0cc6940d7c959e36c73ed1a9a977d3b90bab60fc97d519fe370cc63f38`) comes from
ScrollPrize/villa revision `fc25b4d1f160b494eee5e5f69726df2c1008bf26`, built by that
repository's `volume-cartographer/libs/flatboi/CMakeLists.txt` (libigl `ae8f959` plus
the local `libs/libigl_changes` overlay, PaStiX `v6.4.0`, `SLIM_CACHED PASTIX6`).

## Build

```
git clone --filter=blob:none https://github.com/ScrollPrize/villa villa
git -C villa checkout fc25b4d1f160b494eee5e5f69726df2c1008bf26
cmake -S tools/flatboi-recovery -B build -G Ninja -DVILLA_SOURCE=$PWD/villa -DCMAKE_BUILD_TYPE=Release
ninja -C build flatboi native_regression
```

`CMakeLists.txt` fetches libigl at `ae8f959` and PaStiX `v6.4.0` with the same options
and patches the same PaStiX CMake quirks as the historical file; it only drops the Qt
application tree. Host packages beyond a C++ toolchain: `gfortran`,
`libopenblas-dev liblapack-dev liblapacke-dev libscotch-dev libptscotch-dev`,
`libopencv-dev`.

Artifacts (this host, GCC 13.3, Release):

| binary | sha256 |
| --- | --- |
| unmodified reference `build/flatboi` | `fec002cd8073f437d8ca9584fd2838408b5d456b6bd78623f23c4c72262106f4` |
| patched `build-patched/flatboi` | `644dbd14ffbd3c43d9127574fe742b4a9e513daa10b59c45fa39742a921f3228` |
| `build-patched/native_regression` | `cf0d4920fbf293f63e56f3042042d15448ba86ec0e69e348b65570a38c8b7d67` |
| shipped (AppImage, for comparison) | `d724cf0cc6940d7c959e36c73ed1a9a977d3b90bab60fc97d519fe370cc63f38` |

The rebuilt binary cannot be byte-identical to the shipped one (different compiler,
BLAS and PaStiX build); equivalence is established functionally on identical inputs
(see below), never by hash.

## Corrections (`flatboi-recovery.patch`, applied by `apply_patch.py`)

1. **Every PaStiX stage return code is checked.** `pastix_task_analyze`,
   `pastix_task_numfact` and `pastix_task_solve` were fire-and-forget, so a failed
   factorization was solved and published as if it had succeeded.
2. **The symbolic factorization is keyed on the full pattern.** It was reused whenever
   `(rows, cols, nnz)` matched; two different graphs with the same nnz scattered
   coefficients into the wrong rows. The identity is now an FNV-1a hash of the complete
   CSC/CSR pattern.
3. **Coefficients and diagonals are validated before factorization** (finite, positive
   diagonal) and failures dump the exact system, rhs and pattern as lossless doubles.
4. **Symmetric equilibration**: `(D L D) y = D b`, `x = D y` with `D = diag(1/sqrt(L_ii))`.
   Same equations and same solution; it stops the assembled operator's 18-decade
   coefficient range from destroying the Cholesky pivots.
5. **Acceptance is the backward error of the original system**,
   `||Lx-b|| / (||L|| ||x|| + ||b||) <= 1e-10`, recorded per iteration. A bad step is
   never written out.
6. **The final UV is validated before publication**: a map with zero or inverted faces
   exits non-zero instead of writing energies and a checkpoint.

## Native regressions

`build-patched/native_regression` (all five must pass):

* non-SPD diagonal rejected with an explicit cause;
* two distinct patterns with identical `n`/`nnz` re-analyse and each solve its own
  equations;
* repeated factorization of one pattern is stable;
* the solved vector satisfies the original equations;
* a system whose diagonal spans `1e18..1e-18` is solved to backward error `9.5e-41`
  and relative error `1.4e-16`.

## Known unresolved family

The shipped binary prints `Negative diagonal term` (PaStiX `core_dpotrf2sp`) and then
continues. Reproduced on W-41's saved input, it is not deterministic: two-iteration
reruns stay silent, ten-iteration runs print it and then slow to a crawl. The
instrumented build shows why: the assembled SLIM operator has a diagonal range of
`1.5 .. 1.8e18` because it is built from gradients of physical micrometre coordinates
over near-degenerate triangles, so the `lambda I` term is lost in double precision and
the operator stops being numerically SPD. Equilibration alone does not restore
definiteness once the damage happened during assembly.

Fixing that requires per-element assembly scaling or a documented regularization of the
SLIM step, both of which change the objective's numerics and need their own validation
on the original equations. They are deliberately **not** included here: the corrections
above make the failure visible, attributable and non-publishable, but do not recover
those cases.
