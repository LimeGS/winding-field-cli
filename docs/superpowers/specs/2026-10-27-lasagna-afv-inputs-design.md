# Lasagna observations and external AFV inputs

Status: approved by the user after builder review on 2026-10-27.

## Intent and user decisions

The automatic CT pipeline must use Lasagna for surface normals and target
winding density, and Qual V2 for fibres/splines. The user also requires importing
an external `.afv` and performing the necessary processing automatically.
Lasagna inference belongs inside the CT pipeline: a user should not need to
prepare its predictions manually or choose a model for each scroll.

This replaces the current automatic route that estimates normals and density
from Qual fibre presence. Reading Lasagna's exported observations in the original
research workflow and producing Lasagna-compatible exports in this CLI are
different operations; the latter is not evidence of using the Lasagna model.

## Public model and upstream integration

Builder choice: default to the public `scrollprize/lasagna/model_current.pt`,
pinned to Hugging Face revision `de45f01147adea9f4ad0f621b1ccd885a1f03760`
and SHA-256 `4cc0c40a846b296769b9f7982470559078662a1a0b0e1f74ff41ed9341c0d600`.
The published file is 558,287,587 bytes. Download and verify it automatically
using the existing checkpoint acquisition pattern; support the existing offline
contract and a local path for the same pinned checkpoint.

The public repository lists one checkpoint and its model card only declares a
license. It does not establish accuracy on every scroll or document a separate
checkpoint per scroll. Do not manufacture a universal physical-resolution range
or inherit the Qual input contract as proof of Lasagna's scientific validity.

Reuse the official Lasagna inference implementation and decoding, preserving its
normalization, model architecture, density scale and tiled inference behavior.
Do not substitute a locally invented density estimator. The already inspected
Villa revision is `b29f310957c7b29e7d332f9876e0e487819cfaa3`.
Its Lasagna distribution requires Python 3.14; the CLI currently supports 3.11/3.12.
Builder choice: orchestrate a pinned official Lasagna subprocess in its own
environment rather than force the CLI and numerical core onto Python 3.14.
Provide reproducible one-time runtime setup instructions and a preflight that
reports a missing/incompatible runtime before creating a partially started run.
Once installed, an ordinary CT run invokes Lasagna automatically.

The builder must inspect checkpoint metadata and upstream inference before
choosing compatible tile/context defaults. Confirm the exact official invocation
and public-model compatibility in an integration check; a mock alone is insufficient.

## CT preparation and solver inputs

An ordinary run obtains two distinct observation products:

- Lasagna: predicted `grad_mag`, surface-normal directions, and their validity.
- Qual: fibre probabilities, tangents and official traced splines.

Decode Lasagna's normals with its actual encoding. Convert `grad_mag` to the
solver's physical units using the model/export encoding and the selected input
grid's transform, exactly once. Persist source values, transforms, support and
model/code identities so later stages do not infer units from filenames.
Surface normals remain unsigned until the existing explicit orientation step;
using Lasagna does not itself establish the global sign or sheet calibration.

Preserve strict output ROI, acquisition-invalid gaps, bounded source reads,
source-content verification, disk admission and independent field spacing.
Any staged CT required by the official subprocess must be bounded by declared
input context, included in `plan` disk estimates and recoverable after interruption.
Inference context may extend beyond output ROI; output support must not.

Keep surface evidence separate from fibre evidence. A hole in fibre prediction
must not silently erase valid Lasagna surface support or become measured zero
density. Conversely, imported curves must not bridge unsupported surface gaps.
Field construction consumes Lasagna-derived density and oriented normals plus
the selected fibre constraints. Automatic level selection must use appropriate
surface evidence rather than treating fibre presence as a surface measurement.

Persist reusable Lasagna products and include a real density product in relevant
exports; `grad_mag_*` manifest fields without an exported density array are not
sufficient. Retain existing CT, field, mesh and atlas command ownership.
The synthetic offline classical demo remains an explicitly labelled baseline.
Preserve explicit classical/SD2 selections; document exactly which surface
observation path they use instead of silently changing them into another model.

## External AFV

Builder choice: expose `--fibres /path/input.afv` on `run` and `plan`. Without it,
use the normal Qual inference/tracing path. With it, import the file instead of
running Qual inference or tracing; Lasagna surface inference still runs.

Support AFV v1 SQLite files compatible with the current exporter and official
Qual tracer. Validate schema, completion, finite coordinates, block continuity,
inventory and source identity. Stream curves/blocks instead of loading the
entire archive into memory. Open the input read-only and hash-bind it.

Read trustworthy coordinate metadata and transform native/source/ROI coordinates
to physical XYZ exactly once. A calibrated external AFV must not be implicitly
reinterpreted as ROI-local or source level zero. If geometry is missing or
ambiguous, require an explicit small geometry descriptor with frame, scale and
origin instead of guessing. Precise descriptor/flag names are the builder's call.

Clip curves to ROI and split at unsupported gaps while retaining intersecting
edges even if both endpoints lie outside a solve block. Preserve source IDs and
available family labels; ordinary fibres with no H/V label remain usable with
their unknown family recorded. Reuse the existing spatial index, per-block
constraint construction and AFV exporter after normalized import.

Do not smooth, refit or retrace imported curves unless required by their format;
the existing clipping/resampling for numerical constraints is sufficient.
Report input/imported counts and reasons for exclusions. Invalid geometry must
fail explicitly rather than create a scientifically misregistered result.

## Recovery, scope and acceptance

Keep preparation and fibre-import work resumable. Bind retained products to
model, upstream implementation, source bytes, transforms and settings. Changing
an AFV, metadata or Lasagna products must be detected even after a completed
stage. Older implementation-bound runs retain their existing receipts and need
their original environment; do not rewrite them to bypass validation.

Work reaches CLI/config validation, model acquisition and provenance, preflight
and planning, observation preparation, fibre dispatch/import, solver adapters,
exports, resume validation, packaging/setup instructions, tests and README/formats.
The numerical solver itself does not need replacement.

Required checks:

- Lasagna values/normal decoding and physical scaling against upstream reference.
- External AFV transforms for shifted origins, source levels and ROI-local data;
  unknown families, cross-boundary edges, unsupported gaps and malformed inputs.
- No Qual download/inference/tracing when external fibres are selected.
- STOP/resume, input/product mutation and completed-stage integrity checks.
- Strict ROI, field-spacing independence and realistic storage planning.
- Existing synthetic regressions and installed-package invocation.
- A small real-CT end-to-end run with official Lasagna weights and both fibre
  paths, documenting runtime and platform coverage without claiming physical
  accuracy or whole-scroll throughput.

A scientific A/B comparison, universal model selection, arbitrary mesh import
changes and a general cross-run cache system are outside this request. Reuse
within a resumable run is required; broad cache functionality is not.

## Review checklist

- [x] Confirm requested Lasagna/Qual responsibilities and external AFV goal.
- [x] Inspect existing integration, original Lasagna consumers and public weights.
- [x] Obtain fresh builder review and resolve substantive design gaps.
- [x] Present design and obtain user approval.
- [x] Capture the initiative and write the implementation plan before code.

Builder review found no important unanswered questions. Live upstream inference
and checkpoint/runtime compatibility were not exercised during design review;
they remain explicit implementation acceptance checks.

## References

- https://huggingface.co/scrollprize/lasagna/tree/de45f01147adea9f4ad0f621b1ccd885a1f03760
- https://github.com/ScrollPrize/villa/blob/b29f310957c7b29e7d332f9876e0e487819cfaa3/lasagna/preprocess_cos_omezarr.py
- https://github.com/ScrollPrize/villa/blob/b29f310957c7b29e7d332f9876e0e487819cfaa3/lasagna/setup.py
- `src/winding_field/predict.py`, `splines.py`, `field.py`, `exports.py`, `workflow.py`
- `docs/FORMATS.md`, `docs/PREPARED.md`, `provenance/villa.json`
