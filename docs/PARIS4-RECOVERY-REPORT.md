# Paris4 SLIM recovery — final status

Frozen census: `CASES.json`, snapshot 2026-11-18T02:41:36Z, **63 failed of 172 bundles**
(109 ready). This report compares that census with every receipt produced since, using
`tools/paris4-recovery/compare.py`:

```
compare.py --baseline CASES.json --evidence <receipts> --output COMPARISON-NOW.json
```

## Counts

| bucket | cases | meaning |
| --- | --- | --- |
| cause removed, verified | 6 | the failing stage was rerun and now passes |
| cause removed, unverified | 9 | the same measured cause is gone from the code, those bundles were not rerun |
| unresolved, cause established | 32 | reproduced and attributed; not repairable inside this change |
| pending, never rerun | 16 | admission/limit families not re-executed end to end |

**No bundle has a published, validated UV map yet.** "Cause removed" is a statement
about the stage that failed, not about the mesh.

## By family

| family | cases | state | evidence |
| --- | --- | --- | --- |
| GMRES did not converge | 13 | cause removed: the 300-iteration ceiling. W−40/−38/−30/−24 rerun: 13/13, 6/6, 13/13, 16/16 domains initialized, 0 non-positive faces | `init-w*/INITIALIZATION.json`, `TASK3-W40-CONTINUATION.json` |
| overlap narrow-test budget / candidate materialization | 2 | fixed: complete audits; W10 needed 42,890,715 narrow tests (2.1× the old ceiling), W−73 2,512,208 with 0 pairs | `TASK2-AUDITS.json`, `wm60-audit-v4`, `wp10-audit-v4`, `wm73-audit-v4` |
| bounded arrays / 6 GiB admission (W0, W1…W9) | 10 | limits removed; W0 and W1 verified at read+cut (2,102,206 and 1,946,910 faces, identical domains); not rerun through SLIM | `task1-read.log`, `task1-cut-w*.log` |
| `inverted UV triangle` | 9 | 3 rerun and still failing (W−20, W11, W12); 6 not rerun | `init-wm20`, `init-wp11`, `init-wp12` |
| `collapsed uniform initialization` | 1 | W−129 rerun, still failing | `init-wm129` |
| `native negative diagonal` | 22 | not recovered; cause established below | `native-w41-patched`, `native-w41-dump`, `TASK3-EXPERIMENT.json` |
| checkpoint inverted/degenerate | 5 | not recovered; exact negative face per case | `wm14-exact.json`, `exact-batch-*.json` |
| warm-start energy differs | 1 | not addressed (task 5 was not executed) | `wm102-resume` |

## Established causes of what remains

**22 × `negative diagonal`.** The assembled SLIM operator has a diagonal range of
`1.5 … 1.8e18` because it is built from gradients of physical micrometre coordinates
over near-degenerate triangles; `lambda I` is lost in double precision and the operator
stops being numerically SPD. Symmetric equilibration is implemented and validated (a
`1e18`-range system solves to backward error `9.5e-41`), but it cannot undo damage done
during assembly. The shipped binary is unchanged; the corrected one is built and
documented in `tools/flatboi-recovery/` and refuses to publish a bad step.

**10 × geometry.** The circle-boundary harmonic parameterization collapses thin
filaments: with a *direct* solve at `2.6e-15` residual, W−20 leaves 7 faces at area
`0/−1e-21` and W−129 puts 7010 of 9687 interior vertices on a single point. This is the
boundary correspondence, not the solver; changing it changes every UV in the sheet.

**5 × checkpoint.** Each inspected case has exactly one genuinely inverted face
(exact decimal arithmetic: `−1.42e-08` for W−14, `−1.92e-17` for W−100 batch 030).
The rejection is correct.

## Measured cost (initialization-only reruns)

| case | wall | peak RSS |
| --- | --- | --- |
| W−40 | 206.3 s | 0.92 GB |
| W−38 | 178.4 s | 0.92 GB |
| W−24 | 130.6 s | 1.05 GB |
| W−30 | 123.9 s | 1.00 GB |
| W11 | 123.1 s | 1.22 GB |
| W−20 | 95.8 s | 1.09 GB |
| W12 | 71.1 s | 0.98 GB |
| W−60 (control) | 59.0 s | 0.76 GB |

W−40's mean-value system needed 750/780 inner GMRES iterations, where the delivered
budget stopped at 300; the same factorization converges to a true residual of `8.6e-11`
with continuation, and the solved vector differs from a direct solve by `5.7e-08`
relative.

## What this change does not claim

- No anatomical or physical acceptance: every receipt stays `physical_acceptance=false`.
- No recovery of the 22 native, 10 geometric, 5 checkpoint or 1 warm-start bundles.
- No end-to-end rerun of the 16 pending bundles.
