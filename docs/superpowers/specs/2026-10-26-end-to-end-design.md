# Public end-to-end winding-field CLI

Approved in conversation: an independent public repository, installable CLI and
automatic original CT -> prerequisites -> fibres -> field -> meshes -> flattening,
for a complete scroll or an exclusive output ROI. A morning demo must be reproducible.

Local CT takes priority. Missing data comes from the original S3 source. Download
the required domain when it fits after reserving intermediate/output space;
otherwise stream through a bounded cache. ROI output is strict; input halos are
allowed and recorded. Never confuse an incomplete local mirror with known zero.

Reuse the existing winding-field numerical core and upstream inference tools.
Preserve physical geometry, unsigned normals versus fibre directions, model
identity, calibration, support and reproducibility. No manual existing mesh,
umbilicus or fitting annotations are prerequisites of the automatic route.

Every stage is independently runnable and resumable. Checkpoints and products
are durable; CT cache is disposable. Failures retain progress. Changes to the
source or numerical problem cannot silently reuse old results.

Outputs include a persistent field, all supported mesh components, flat charts
with 3D correspondence, quality/coverage and optional CT previews. Numerical
completion is distinct from physical sheet correctness. Unsupported or ambiguous
regions remain explicit. Full-volume operation must not silently become a crop.

Implementation choices: Python 3.11/3.12, Linux/CUDA for production, CPU reference
and demos; argparse CLI; existing numerical modules namespaced under winding_field.
vc_reader is the preferred native reader with a portable Zarr fallback. Use
pretrained inference when selected; a separately labelled classical CT baseline
supports an offline demo and comparison, never impersonating a neural model.

Validation: local/HTTP parity, bounded cache, source mutation, ROI transforms,
interruption/resume, synthetic known layers, actual original CT, clean wheel
installation, and fresh-context review. Whole-scroll physical reconstruction is
a scientific milestone, not something inferred from successful file exports.
