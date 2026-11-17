# Local field-constraint boundary rounding

## Fix and qualification

The ordinary field constraint provider must query, clip and construct its
trilinear rows against one local geometry:

```
block_origin = global_origin + lo[::-1] * spacing
block_shape  = hi - lo
block_upper  = block_origin + (block_shape[::-1] - 1) * spacing
```

Computing the upper corner from a global end index instead can differ by one
representable float from the core's local bound. The physical support check then
correctly rejects a generated endpoint. Conversely, a smaller global bound can
drop a valid boundary segment. This is independent of a support mask.

The fix changes only the CLI provider, including its nonspatial inside test.
It neither modifies core 1.5.2 nor adds tolerances, clamps arbitrary external
constraints, discards invalid rows, or changes the already-correct bounded path.

Tests exercise the actual `field.solve` constraint consumer, real SQLite/R-tree,
clipping/sampling and the published core. They cover nonzero/negative origins,
isotropic/anisotropic spacing, all six box faces, exact nodes, one-ULP inward/
outward points, different halos, truncated final blocks and support rejection.
The numerical iterator is replaced only to invoke the provider at a selected
block; separate existing field tests still exercise actual solves and resume.

An owned local replay of the PHerc0483A diagnostic reads all 1,193 original
selected curves. The baseline provider fails; the fixed provider constructs
all 26,943 rows with 305,184 columns. The original invalid-pair archive remains
invalid to the unchanged core, as it should: the fix regenerates samples from
curves using coherent geometry rather than modifying that archive.

Local qualification: 208 focused tests passed from source and the same 208
passed using a wheel installed outside the checkout, with the published
core 1.5.2 and its source hashes verified. The complete real-curve replay also
passed through the installed consumer. A fresh focused review found no
Critical/Important or Minor issues. The protected preparation, fibre, RUN and
partial-checkpoint receipt hashes remained identical to the diagnosis.
The full suite, optional benchmarks and Docker were not run for this hotfix.

## Recovery proposal — not an automatic migration

Do not run normal CLI `resume` with a changed implementation. Do not edit
`implementation_sha256`, source pins, saved identities or old receipts. An old
checkpoint lacking a provider-code binding is not proof of compatibility.

**Recommended for the reported partial first sweep: preserve it and restart
only field in a new, explicitly authorized recovery output.** Already solved
blocks may have included a subtly different boundary set, so adopting the
partial checkpoint has not been qualified.

After separate authorization, use one reviewed, case-specific field-only launcher:

1. Freeze an inventory of the protected run, its original runtime/core hashes,
   configuration and PREPARED/FIBRES/checkpoint receipts. Ensure no writer is
   active; do not stop unrelated services or remove the old STOP marker.
2. Obtain adequate approved storage for a private snapshot/reflink copy. Never
   point a writable SQLite handle, generated receipt or output at the original.
   Include the prepared maps, their index and metadata, fibre database and
   required provenance/sealed input ledgers. Scientific payloads and source pins
   must be byte-identical; do not rerun prepare, prediction, import or AFV export.
3. Independently validate the private snapshot: complete tile inventory and
   checksums match PREPARED; fibre bytes, curve/bounds inventory and annotations
   match FIBRES; geometry/calibration/configuration and consumed source/metadata
   hashes match the frozen original records. Reject absent or changed inputs;
   do not repair/relabel old records. If verification cannot establish reuse,
   stop and report it rather than recomputing preparation without permission.
4. Create a new recovery manifest, not a rewritten RUN. Record the old receipt
   pins as provenance, new qualified wheel/core/implementation hashes, unchanged
   scientific settings, the fresh-field decision and all validation results.
   Preserve the old RUN and checkpoint unchanged. Assert the new output contains
   no adopted `field/`, FIELD receipt, pair cache or partial solver state.
5. Invoke the existing `winding_field.field.solve(new_output, config)` API in
   that launcher under the approved resource limits. This is a new field-only
   computation, **not** a bypass of the old workflow's resume guard. Allow normal
   initialization and all blocks/sweeps to run from the copied prepared inputs.
   Record its output separately; do not present it as an in-place resumed run.
6. Recheck original inventory hashes and validate the new field receipts,
   stored payloads, geometry and constraints. Operational completion remains
   distinct from global convergence and physical/scientific acceptance.

A synthetic copy-and-restart test validates reuse without modifying the source
preparation, fibres or partial checkpoint. It does **not** establish production
snapshot capacity, complete real-data source verification, or successful
completion of the reported field. No production recovery was executed.

## Publication proposal

Commit this focused fix separately from config/flag parity. Review and qualify
the installed wheel with the published pinned core; keep old tags immutable.
After explicit release authorization, use the next free CLI patch version
(candidate 1.5.7 if still free), leave core 1.5.2 pinned, and publish notes
describing the changed boundary arithmetic and separate recovery requirement.
Do not wait for CI if the user maintains that instruction, but report CI's actual
state and exactly which local checks ran.

Release publication, deployment to Vast and launching the recovery each require
their own explicit authorization. No generic migration command is introduced.
