# winding-field-cli

## Verification

- Before every commit you will push: run the `verify` skill (`python tools/verify.py`) and
  stamp the commit with `Verified: tools/verify.py PASS (baseline <sha7>)` or
  `Verified: none (<reason>)`.
- Check the REVIEW.md items the `review` step lists; they come from past incidents.
- A failing check gets a root cause. Never skip, xfail, quarantine, add sleeps or retries,
  or raise timeouts to get green. If a flaky test has no root cause, delete it and write
  down why in the commit message.
- Three rounds on the same failure without progress: stop and bring in the user.
- Fix the code, not the check; intended differences go in `tools/verify-approved.json`
  with a reason.

## Repository rules

- Python is ruff-formatted (line length 100). Format only the files you touched.
- `src/winding_field/surface_render.py` is hash-pinned: never edit or format it.
- `src/winding_field/_vendor` is vendored: do not format it.
- Flattening lives in `src/winding_field/flattening/` (see docs/FLATTENING.md, Code
  layout). LSCM only through `flattening/lscm.py`; SLIM only through Flatboi.
- Do not merge pull requests unless the user asks explicitly.
- After a history rewrite on GitHub, re-clone or reset to `origin/main`; never push from
  an old clone.
