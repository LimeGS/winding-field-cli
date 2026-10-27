# Independent review and fixes

A fresh, read-only whole-repository review covered `163367a..b7221c0`. The first
review attempt failed with an infrastructure authentication error and delivered
no report. Its replacement completed the probes; a stalled final response was
interrupted and resumed solely to return the existing findings.

The report requested changes: **eight important findings, no critical or minor
findings**. The implementer reproduced each issue with a failing test, applied the
fix and ran the full suite: **39 passed**. There was no second review pass.

| Finding | Fix and regression |
| --- | --- |
| Stale or incorrect local mirrors bypassed original content | Online content verification; `test_online_mirror_resume_detects_changed_authoritative_chunk`, `test_online_mirror_rejects_wrong_content_with_matching_metadata` |
| Oversized remote update retained an obsolete cache entry | Always discard the superseded entry; `test_remote_update_larger_than_cache_cannot_revive_old_payload` |
| Stride allocated a dense native envelope | Native-chunk-bounded Cartesian sampling; `test_strided_preparation_reads_native_chunks_not_dense_envelope`, `test_sampled_reads_preserve_duplicates_and_cross_chunk_coordinates` |
| Per-tile Otsu caused seams | One persisted run-wide threshold; `test_classical_threshold_is_shared_across_tiles_and_persisted` |
| Partial atlas could complete with missing companions | Expected hashes plus deterministic companion recovery; `test_partial_atlas_resume_restores_missing_and_corrupt_companions` |
| Field initialization ignored stop/reserve | Block-boundary stop and reserve propagation; `test_field_initialization_stops_at_block_boundary_and_resumes` (two cases), `test_field_initialization_preserves_disk_reserve` |
| Resume recharged all products | Remaining-product accounting and completed-run admission; `test_resume_admits_remaining_products_and_completed_runs` |
| Local fibre queries scanned the global archive | Index `curves(tile)` on new/resumed archives; `test_resumed_fibre_archive_uses_indexed_tile_lookup` |

No numerical-core rewrite was needed. [Decisions and unvalidated boundaries](DECISIONS.md)
include everything the reviewer declined to judge.
