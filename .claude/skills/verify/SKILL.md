---
name: verify
description: Verify a winding-field-cli change end to end before committing or opening a PR. Runs the real CLI on synthetic data (demo, native SLIM via Flatboi, stop/resume), compares flags and numerical outputs with the base branch as the oracle, checks that runs made by the previous version resume or are refused cleanly, and lists the REVIEW.md items the diff triggers. Use it for any change under src/, before every commit you will push.
---

# Verify

Run from the repository root with the project venv active:

```bash
python tools/verify.py
```

Default baseline is `origin/main` (fetch it first). Use `--baseline <ref>` for another
oracle, `--steps a,b` for a subset while iterating, `--out DIR` to keep run directories.
About 1 minute. Receipt: `.verify/VERIFY.json`.

## Steps

| Step | Proves | Typical failure |
|---|---|---|
| `review` | Lists REVIEW.md items triggered by the diff (from the baseline's REVIEW.md) | — informational: check each listed item |
| `flags` | Every public command, option and choice of the baseline still exists | a removed/renamed flag (as in 3556a4b) |
| `golden` | Flattening paths, native charts (default + SLIM) and field checkpoints are byte-identical to the baseline | numerics changed by a refactor or optimization |
| `e2e-demo` | `winding-field demo` completes and `resume` leaves it byte-identical (except RUN.json bookkeeping) | broken pipeline, resume rewriting outputs |
| `e2e-native` | native run stopped after `field` and resumed to COMPLETE, SLIM recorded as `flatboi-slim` | stop/resume or Flatboi integration broken |
| `cross-version` | a baseline-made run resumes or is refused cleanly (unchanged, no traceback); baseline Lasagna/AFV stages are reusable with `--reuse-from` | resume identity or reuse broken |

Flatboi: SLIM needs the pinned runtime (`python -m winding_field.flattening.flatboi
--provision`, or `WINDING_FLATBOI_APP_RUN`). Without it `e2e-native` and the SLIM golden
key are skipped with a WARN; `WINDING_REQUIRE_FLATBOI=1` turns that into a FAIL (CI).

## When a step fails

1. Read the evidence in `.verify/VERIFY.json` (removed flags, changed golden keys, stderr).
2. Fix the code, not the check. If the difference is intended (a deliberate numerical
   change, a removed flag with a migration path), add it to `tools/verify-approved.json`
   with a reason and say so in the PR body (REVIEW.md R10).
3. Three rounds on the same failure without progress: stop and ask the user, with the
   evidence.

## Before committing

- Run the fast checks for what changed (`ruff format` on touched files, never
  `surface_render.py`; the related tests), then `python tools/verify.py`.
- Stamp the commit message with a trailer, including when nothing ran:
  `Verified: tools/verify.py PASS (baseline <sha7>)` or `Verified: none (<reason>)`.
