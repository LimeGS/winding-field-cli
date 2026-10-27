# Lasagna and external AFV acceptance — October 27, 2026

Official Lasagna supplies CT surface normals and density; Qual V2 or imported
AFV curves supply fibre constraints. The numerical library remains the pinned
external dependency. Setup and public commands are in [LASAGNA.md](../LASAGNA.md).
The compact [receipt](lasagna-afv-inputs-receipt.json) records revisions, versions,
geometry, runtime fingerprints and product hashes for code revision `07f5add`.

## Regression and installation

- Source suite: **323 passed in 102.00 s**.
- Clean installed wheel, outside the checkout, with `PYTHONPATH` removed and
  pytest's `pythonpath` empty: **323 passed in 137.72 s**. This includes the
  installed CLI demo and completed resume from another directory.
- All **39 Python/JSON package files**, four installed provenance records and
  `THIRD_PARTY.md` match the checkout. Installed `run`/`plan` help exposes all four
  input/runtime options; `models --help` includes Lasagna; `doctor` succeeds.
- The clean installation obtains core revision
  `397ad750e1ee7f7aa5cb84ca575e750db3e3c232` and spline-generator revision
  `3c8b73628ca56cb28042b9b2e79439b2e1c693d2` from their public archives.
- Coverage includes calibrated/anisotropic decoding, unknown families, shifted
  frames, sparse crossing edges, support gaps, malformed input, transactional
  recovery, child termination, changed inputs/products and completed-stage checks.
  External-AFV tests forbid fibre-model acquisition, inference and tracing.

Commands used (paths are local verification environments):

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .runtime/separation-venv/bin/python -m pytest -q
.runtime/separation-venv/bin/python -m build --outdir .runtime/lasagna-dist
uv venv --python .runtime/separation-venv/bin/python .runtime/lasagna-install-venv
uv pip install --python .runtime/lasagna-install-venv/bin/python \
  'winding-field-experiments @ https://github.com/LimeGS/winding-field/archive/397ad750e1ee7f7aa5cb84ca575e750db3e3c232.tar.gz' \
  .runtime/lasagna-dist/winding_field_cli-0.1.0-py3-none-any.whl pytest
# From an external temporary directory containing copied tests/examples/tools:
env -u PYTHONPATH OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /absolute/path/to/lasagna-install-venv/bin/python -m pytest tests -o pythonpath= -q
```

Host: macOS 27.0.1 arm64. CLI Python 3.11.13, NumPy 2.4.6, Torch 2.14.1,
SciPy 1.17.1, scikit-image 0.26.0 and numcodecs 0.16.6. The separate official
runtime uses Python 3.14.10, Torch 2.14.1, NumPy 2.5.4 and Zarr 3.4.2;
899 runtime source files are fingerprinted. Python 3.12/CUDA/Linux were not
exercised in this acceptance.

## Real upstream boundary

The installed official Villa revision is
`b29f310957c7b29e7d332f9876e0e487819cfaa3`. Its strict loader recovered all
**306 checkpoint tensors** from the public 558,287,587-byte model, SHA-256
`4cc0c40a846b296769b9f7982470559078662a1a0b0e1f74ff41ed9341c0d600`.
The adapter invokes official `predict3d` with 64³ tiles, overlap 32, border 16,
full-input-lattice output and upstream normalization/blending/decoding.

A direct bridge run completed a 48³ ROI with 80³ staged context in **44.28 s**.
All 110,592 decoded densities matched official `grad_mag / 1000 / 8.64`
(relative tolerance 2e-7); supported normals were unit length.

The untouched upstream CPU BF16 probe processed eight tiles in **2017.4 s**;
the FP32 worker probe reported **8.9 s** of inference. A controlled convolution
confirmed software-emulated BF16 on this host (oneDNN unavailable). These are
backend-specific measurements, not whole-scroll throughput estimates or proof of
BF16/FP32 byte equivalence.

## Real CT → atlas: both fibre paths

Source: `s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr`.
Selected level 0; ROI `1488 1536 4032 4080 6048 6096`; observations 48³ at
8.64 µm. Origin XYZ is `(52254.72, 34836.48, 12856.32)` µm. The independent
field is 24³ at **exactly 17 µm**. One solver sweep, field/storage chunks 16,
four CPU threads, maximum 2000 iterations and relative tolerance 1e-6.
Zero remained valid because no acquisition-invalid sentinel was asserted.

The acceptance driver invoked `workflow.run` with these equivalent CLI settings:

```bash
CT='s3://vesuvius-challenge-open-data/PHerc0175A/volumes/20250521115057-8.640um-1.2m-116keV-masked.zarr'
winding-field run --source "$CT" --voxel-um 8.64 \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --chunk 16 --field-chunk 16 --sweeps 1 --threads 4 \
  --ct-mode stream --cache-gib .25 --reserve-gib .25 \
  --lasagna-python .runtime/lasagna-venv/bin/python \
  --lasagna-model-path .runtime/lasagna-model.pt --model-path .runtime/qual-v2.pt \
  --output .runtime/lasagna-acceptance-final/qual
```

It created `STOP` after the actual Lasagna child reported model loading, asserted
`PAUSED` with a sealed work inventory, then resumed through flattening. A copied
exported AFV fed the second run using the same settings, replacing `--model-path`
with `--fibres .runtime/lasagna-acceptance-final/external.afv` and a new output.

| Result | Lasagna + Qual | Lasagna + external AFV |
|---|---:|---:|
| Wall time, including interruption/resume where exercised | 65.19 s | 43.31 s |
| Retained run bytes at measurement | 11,915,224 | 8,234,856 |
| Curves | 7 | 7 |
| Fibre constraint rows | 143 | 143 |
| Mesh faces / components | 2,283 / 7 | 2,283 / 7 |
| Linked charts | 13 | 13 |
| Extracted-face coverage | 100% | 100% |

Surface maps are exactly equal between the two runs. Import preserves all source
IDs, vertex counts and families (**4 V, 3 H**), with maximum world-coordinate
round-trip difference **7.28e-12 µm**. Fibre and mesh vertices satisfy their strict
ROI bounds. The two fields need not be bit-identical after floating-point coordinate
round-tripping; container metadata also has distinct provenance.

Changing the copied AFV was rejected on completed resume without changing product
hashes. Restoring it allowed completed resume with stable hashes. Both runs used
previously verified cached CT bytes (zero new CT download bytes); reported disk
sizes exclude shared cache, models and runtime environments.

An earlier acceptance exposed artificial collinear vertices during import
(143 versus 448 constraint rows). A failing regression reproduced it; support
traversal now retains original vertices unless clipping/splitting is necessary.
The table and receipt are the corrected rerun, not that earlier experiment.
Large outputs remain under `.runtime/lasagna-acceptance-final`; local verification
drivers/logs are retained under `.runtime/lasagna-evidence`.

## Recorded implementation decisions and limits

- CPU inference uses FP32; CUDA retains upstream BF16. Rounding can differ from
  unmodified CPU upstream, so no byte-equivalence claim is made.
- Lasagna shares the existing labelled child supervisor, with process-group
  cleanup. Connected spline/bridge tests cover that shared behavior.
- The private worker disables upstream's global training-pause service, which
  stalled concurrent CPU work. GPU sharing requires external scheduling.

Scientific accuracy, superiority to the previous estimator, universal checkpoint
coverage, global sheet orientation and whole-scroll performance remain unvalidated.
This establishes bounded integration, coordinate handling and recovery on the
recorded CPU environment. A hard-killed unsealed parent requires a new output.
