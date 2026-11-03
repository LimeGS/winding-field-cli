# Explicit partial TIFXYZ publication

User approved “implementalo end to end asi”: publish useful geometry with explicit
incomplete coverage, preserving all delivered-geometry, integrity/resource gates.
The old runs, their identities, limits and receipts remain unchanged.

## Contract

- Add `coverage_policy`: `strict` (default) or `allow_partial`, frozen in request
  and effective settings. No implicit downgrade or policy change on resume.
- Geometry means all delivered-node/root generation, float32 bounds, support,
  orientation and local/nonlocal intersection gates. Empty surfaces, ambiguous
  roots, bad geometry, numerical uncertainty, resource/integrity failures still block.
- Partial publication is allowed only for `SOURCE_UNCOVERED` or
  `COVERAGE_DEPTH_LIMIT` after geometry completed. Record the first uncertified
  source cell/subcell, depth, and source_scan_complete=false. Remaining coverage
  is UNMEASURED, not an assertion that only one region is missing.
- Audit flags explicitly identify geometry completion and blocked phase. Catch
  qualified coverage failures inside the backend context so resource/identity
  cleanup is checked normally before a partial result can be published.
- Keep XYZ and -1 masks unchanged; never fill gaps or discard additional points.
- Per-level VERIFIED.json and level descriptors record `status` READY/PARTIAL and
  `coverage_complete`. TIFXYZ.json aggregates: READY iff every requested level is
  complete; otherwise PARTIAL. All requested levels must have valid geometry.
- Final publication marker is exclusively READY.json or PARTIAL.json, written
  last and SHA-bound. A partial result must never create READY.json. Interrupted
  or failed work withdraws either publication marker under the owned lock.
- `verify-tifxyz` reprojects/rebuilds/audits in a fresh worker and reports
  TIFXYZ_VERIFIED or TIFXYZ_PARTIAL_VERIFIED. It does not modify the publication.
  Exact audit/region/quality receipt comparison prevents relabelling or hiding loss.
- Exporting partial geometry does not itself claim independent reprojection;
  that is claimed only by the successful independent verifier.
- CLI PARTIAL and TIFXYZ_PARTIAL_VERIFIED return2 (incomplete coverage), with usable
  files and explicit JSON status. Existing complete0/resource75/integrity1 behavior
  remains. Integrated RUN status is PARTIAL, never COMPLETE, and snapshots/records
  its TIFXYZ stage and PARTIAL marker. Resume revalidates it as partial.

## Delivery and limits

Use the existing writer, settings/CLI and workflow. Keep strict compatibility and
all masks/units/flags/budgets. No new dependency, source solver change, automatic
installation, old-identity bypass, commit or cleanup. Native in-place execution.
Tests cover natural incomplete fields, hard failures, mixed levels, pause/resume,
corruption/relabel attacks, source immutability, marker-last and real reader use.
Real historical data still needs its bound adapter; never fabricate FIELD.json.
Stored work from another implementation cannot be silently relabelled as new work.
