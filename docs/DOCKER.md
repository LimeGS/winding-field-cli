# CPU and CUDA containers

Both images expose the complete installed `winding-field` CLI on **Linux x86-64**.
CPU and CUDA select different PyTorch builds, not different workflow stages.
The CLI uses Python 3.12.12; official Lasagna has its own Python 3.14.0 environment.

## Build

From this checkout, with Docker Buildx/Bake:

```sh
docker buildx bake cpu --load       # winding-field:cpu
docker buildx bake cuda --load      # winding-field:cuda, CUDA 12.8.1
```

Builds download pinned source archives, hash-locked Python dependencies and the
[official Villa AppImage](../provenance/flatboi.json). Base images and uv are
digest-pinned; Ubuntu dependencies come from the 2026-11-01 snapshot. The full
AppImage runtime is extracted at `/opt/vc3d`; FUSE is unnecessary.

The July Villa archive contains exactly the existing `historical` Flatboi binary.
November releases removed OBJ input support and cannot replace it in these
source-preserving workflows. The default `installed-2f89bfc9` recipe still
requires its original binary; explicitly select `historical` in these images.

Models download separately through the CLI with the repository's pinned revision
and SHA256 checks. Installed inventories, source provenance, dependency locks,
wheel hash and runtime self-check output live in `/opt/runtime-provenance`.
Use an immutable image ID/digest and the same mount paths when resuming a run.
Rebuilding can produce different wheel or image hashes even with fixed inputs;
retain the original image for binary-bound checkpoints.

## Persistent paths and resource limits

Create host directories owned by the user running the job:

```sh
mkdir -p data outputs models cache scratch
IMAGE=winding-field:cpu
wf() {
  docker run --rm --init --platform linux/amd64 \
    --user "$(id -u):$(id -g)" --memory=21g --memory-swap=85g \
    --mount "type=bind,src=$PWD/data,dst=/data,readonly" \
    --mount "type=bind,src=$PWD/outputs,dst=/outputs" \
    --mount "type=bind,src=$PWD/models,dst=/home/worker/.cache/winding-field/models" \
    --mount "type=bind,src=$PWD/cache,dst=/cache" \
    --mount "type=bind,src=$PWD/scratch,dst=/scratch" \
    "$IMAGE" "$@"
}
wf doctor
wf --help
```

`--memory-swap` is **RAM plus swap**, so 21g/85g permits at most 64 GiB of swap.
Use lower limits on smaller machines (for example 6g/6g permits no swap).
Native checkpointed SLIM requires cgroup v2 with finite RAM ≤21 GiB and swap
≤64 GiB. Existing per-command memory, disk reserve and topology limits still
apply; container limits do not make an oversized scroll feasible.

| Container path | Purpose |
| --- | --- |
| `/data` | Read-only CT, external fibres/predictions, fields and configuration |
| `/outputs` | Persistent runs, checkpoints, meshes, atlases and exports |
| `/home/worker/.cache/winding-field/models` | Persistent verified model weights |
| `/cache` | Default CT and patch CT cache; use `--cache /cache/NAME` or `cache_dir` for separate caches |
| `/scratch` | Temporary files (`TMPDIR`) |
| `/opt/lasagna` | Immutable Lasagna environment; linked at the CLI's default location |

Mount the **models subdirectory**, not the whole home/cache tree. The image's
default Lasagna symlink must remain visible. The default container user is
10001:10001; the example overrides it with the host UID/GID. Mounted directories
must already be writable by that UID. The entrypoint runs the CLI directly;
`--init` forwards signals and reaps orphaned child processes.

## CUDA

On Linux with a compatible NVIDIA driver and NVIDIA Container Toolkit, use
`IMAGE=winding-field:cuda`, add `--gpus all` to the `docker run` function above,
and choose `--device cuda:0` in inference commands. Both Python environments
contain PyTorch 2.10.0+cu128. CPU uses 2.10.0+cpu.

Build/import checks do not establish GPU execution. This check requires a real
GPU and runs CUDA operations in both environments:

```sh
docker run --rm --gpus all --entrypoint python winding-field:cuda \
  /opt/runtime-provenance/docker/verify-runtime.py --variant cu128 --gpu
```

### Validated GTX 1660 / driver 550 configuration

On `gpu-1`, the CUDA image passed both PyTorch environments, the native TIFXYZ
GPU kernel and real Lasagna/Qual inference on a GTX 1660 with driver 550.163.01.
The default NVIDIA image constraint rejects this older driver. The verified run
used CUDA 12 minor-version compatibility with these explicit runtime options:

```sh
--gpus 'device=1' \
-e 'NVIDIA_REQUIRE_CUDA=cuda>=12.0,driver>=525' \
-e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
```

The library path selects the actual host driver: the image's `cuda-compat`
forward-compatibility library fails on this GeForce card. This qualification
covers the tested kernels and models; newer PTX or driver-dependent features
may still require a newer driver. See [NVIDIA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
With only GPU 1 exposed, the CLI device inside the container is `cuda:0`.

## Workflows

All existing arguments retain their meaning. Use absolute **container paths**
inside configuration files, including geometry descriptors and reused inputs.
Physical scale comes from the scan/descriptor, not from Docker.
Keep source files immutable on the host too: a read-only bind mount does not
prevent host writes. Metadata-based change checks cannot distinguish same-size
writes with restored mtime within one filesystem ctime tick (about 4 ms on the
tested Linux host); concurrent source writers are unsupported.

```sh
# Download pinned weights once; reuse the same models mount in later jobs.
wf models --prediction lasagna
wf models --prediction qual-v2

# No ROI means the whole selected CT level. Substitute the actual voxel scale.
wf plan --source /data/scan.zarr --voxel-um 8.64 --device cpu \
  --field-spacing-um 300 --cache /cache/full --until field --output /outputs/full
wf run --source /data/scan.zarr --voxel-um 8.64 --device cpu \
  --field-spacing-um 300 --cache /cache/full --until field --output /outputs/full

# ROI bounds are selected-level voxels, ZYX, upper bounds exclusive.
wf run --source /data/scan.zarr --voxel-um 8.64 --device cpu \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --cache /cache/roi --until field --output /outputs/roi

wf status /outputs/full
wf field /outputs/full
wf resume /outputs/full
```

Qual V2 requires uint8, isotropic 8.64–9.6 µm observations. Choose ROI, scan level,
stride and field pitch using the existing [planning instructions](../README.md).

| Workflow | Container usage / reference |
| --- | --- |
| Custom fibres | Add `--fibres /data/fibres.afv` and its `--fibres-geometry` descriptor |
| Custom Lasagna | Independently add `--lasagna-input /data/surface.lasagna.json` and, when needed, `--lasagna-geometry`; see [Lasagna imports](LASAGNA.md) |
| Reused fields / prepared workflows | Existing `prepared`, `extract`, and `extract-range` commands; see [README](../README.md) |
| Level/range extraction | Preserve the existing field config and supported holes; use the [mesh/range recipes](../README.md) |
| LSCM or SLIM, patches or sheet | `export-sheets --flattening-solver lscm\|slim --scope patches\|sheet`; see [matrix settings](FLATTENING.md) |
| Original mapping / historical patches | Existing `--flattening-method original\|patches` routes; do not combine with solver/scope selection |
| Constant-W stages | Existing [constant-sheet](CONSTANT-SHEET.md) commands, with `--native-recipe historical` for flatten/import |
| Atlas, TIFXYZ, probes and verification | Existing commands are installed in both images; run `wf --help` |

For matrix SLIM settings, include:

```json
{
  "app_run": "/opt/vc3d/AppRun",
  "native_recipe": "historical"
}
```

These are additions to the required output-frame configuration, not a complete
configuration file. Historical patch export already defaults to this recipe.
Constant-sheet example:

```sh
wf constant-sheet flatten --prepared /outputs/prepared \
  --app-run /opt/vc3d/AppRun --native-recipe historical --output /outputs/native
```

CPU TIFXYZ acceleration is built at `/opt/accelerators/cpu/accelerator.json`.
The CUDA image also includes `/opt/accelerators/cuda/accelerator.json`. Select
the backend and manifest explicitly as described in [formats](FORMATS.md).
Compilers are retained for the existing `python -m winding_field.tifxyz_build`
command; its strict numerical build flags remain authoritative.

## Stop and resume

Write `outputs/NAME/STOP`, or send SIGTERM with a sufficiently long Docker stop
grace period. Let the process exit before replacing its container. Retain the
complete output tree, input/model/cache mounts, image digest and path mapping.
Use the existing command's resume option; `run` itself requires a new directory.
See [matrix resume](FLATTENING.md#outputs-and-resume) for its stop-marker rules.
The historical `--flattening-method patches` CLI rejects `--resume`; use its
documented reuse configuration with a fresh output instead.

Checkpoints bind source, settings, code and native binary identity. Selecting
another Flatboi recipe or another image is not an implicit migration.
`import-native` requires the original runner hash and the same native binary.
Reader-bound checkpoints from before the native-only cutover require their
original v1.5.0 runtime; the new reader does not migrate them.

## Smoke checks

Run the image's check with a new writable output path and the same mounts:

```sh
# Override the default CLI entrypoint to invoke the self-contained script.
docker run --rm --init --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD/outputs,dst=/outputs" \
  --mount "type=bind,src=$PWD/scratch,dst=/scratch" \
  --entrypoint python winding-field:cpu \
  /opt/runtime-provenance/docker/smoke.py --output /outputs/smoke
```

Add the models mount and `--models` to exercise real pinned Lasagna and Qual
inference; for GPU use the CUDA image, `--gpus all`, and `--device cuda:0`.
The default smoke is offline: complete classical demo, ROI field, exported
products and checkpoint-preserving resume. Small fixtures test runtime wiring;
they do not establish scientific acceptance or full-scroll resource feasibility.

The `containers` workflow builds CPU and runs the installed suite on pull requests.
Manual dispatch can also build CUDA on a self-hosted runner labelled `linux`,
`x64`, `nvidia`, with Docker access and at least 60 GiB free disk. Enable its
separate `gpu` option to run GPU kernels and real model inference. Test tooling
is installed only in the derivative `docker/testing.Dockerfile` image.

### CT cache fix end-to-end qualification

Commit `1ec8c1d` was built as a wheel and checked against all 85 installed CLI
source modules. Qualification used an isolated macOS environment and derivative
images of the retained reviewed Linux CPU/CUDA runtimes, not clean Docker builds.

- Full installed macOS suite: **1781 passed, 36 skipped**.
- Selected Linux CPU suites: **399 passed, 2 skipped**, plus **1 passed** after
  correcting a missing `/pyproject.toml` test mount. The initial packaging-test
  failure required no production change. The two skips require optional frozen
  native geometry witnesses.
- Selected CUDA-image suites: **288 passed**; real pinned Lasagna/Qual inference
  passed on GPU 1, exposed as `cuda:0`.
- Actual CLI processes completed prepare, fibres, field, mesh and flatten for
  the atlas branch; mesh, flatten, assemble and render for native surfaces; and
  the original global recipe. Stage receipts, export hashes and checkpoint-stable
  completed resume were checked. These used the small synthetic CT fixture.
- Prepared field/surface examples passed supervised build, pause/resume and
  rendering, including CT visualizations.
- Direct-field TIFXYZ passed export, independent verification and resume on a
  planar field. **The curved demo field was blocked by `COVERAGE_DEPTH_LIMIT`**
  at W=2 with 8 µm pitch/tolerance. The same request also blocked on the pre-fix
  runtime: this is an existing coverage-certification limitation, not a successful
  curved-field export. No audit threshold or acceptance policy was relaxed.

The results establish execution of these stages and supported fixtures, not
scientific accuracy, every input combination or production-scale CT feasibility.
Detailed commands/results are retained in `.runtime/crop-fix-e2e-plan.md`,
`.runtime/crop-fix-e2e-suite.log` and `.runtime/crop-fix-{linux,cuda}-results/`.

### Native-only reader qualification (2026-11-06)

Qualification used derivative images of the retained v1.5.0 runtimes, replacing
the hash-pinned reader and CLI wheel. Regenerated locks change only `vc-reader`;
this is not a clean rebuild of the complete Dockerfile.

- Installed macOS suite: **1777 passed, 36 skipped**.
- Installed Linux CPU suite: **1806 passed, 6 skipped**. After the final bounded
  snapshot-recheck fix, the reviewed CPU image passed **68 affected tests**.
- Reviewed CUDA image: **285 integration tests passed**, plus runtime GPU checks.
- Both reviewed images passed offline real Lasagna/Qual inference and workflow
  STOP/resume checks. CUDA used host GPU 1, exposed as `cuda:0`; its first model
  attempt exhausted VRAM occupied by another workload, then passed after it exited.
- SIGTERM/container replacement preserved the native solver checkpoint and source
  hashes while resuming from iteration 1 to 200; a different binary was rejected.

Reviewed local image IDs:

```text
cpu  sha256:f4dbec2cc8a6df86fc953845daccc3a9dc034d50e5789cfd3cde067813c0e0d0
cuda sha256:95d83dd8c4aa9268470a40d49145ce48115e97e8cc6986676759edc277c14903
```

See [crop, sampling and render measurements](RASTER-PERFORMANCE.md) for the
numerical comparison and its limits.

### Retained v1.5.0 qualification on gpu-1 (2026-11-06)

These pre-cutover images passed the installed suite: **1785 passed, 7 skipped** each.
The official Lasagna zero-chunk writer control was then enabled separately and
passed in both images; CI enables it through `WINDING_LASAGNA_PYTHON`.
Both images also passed offline Lasagna/Qual inference as host UID 1000, default
UID 10001 path checks, and runtime/dependency checks. CUDA additionally passed
actual GPU operations in both Python environments and the TIFXYZ GPU kernel.
SIGTERM/container replacement resumed native checkpoints from iteration 1 to 200
while retaining source and prior output hashes; the wrong recipe was rejected.

Validated local image IDs (retain these images for their checkpoints):

```text
cpu  sha256:4edc826f6622b53fbf11d5bb7954f53fb998447978503c1515cb1e2b2e8e8696
cuda sha256:718ad442657fd4013ff30da033b6d48bd5f363d2bb030f911f73d30e77b47b93
```

The full suites omit the macOS-only case and four external frozen W50 controls.
The optional external VC3D reader control was also initially skipped; enabling
it separately on CPU **failed**: the bundled July `vc_tifxyz2obj` retained 66,305
of 66,821 valid fixture nodes. The archive is qualified for Flatboi workflows,
not that converter's TIFXYZ round trip. Do not use it as proof of lossless TIFXYZ
consumption. The CLI's own exporter and verification tests passed. Both suites
also report the existing multithreaded `fork()` deprecation warning in a test.
The new container CI workflow has been validated locally but has not run on GitHub.

## Updating locks

Use uv **0.9.17**. Update source pins deliberately, then regenerate and verify
both variants rather than editing generated hashes:

```sh
for backend in cpu cu128; do
  uv pip compile pyproject.toml docker/cli.in --extra plot --extra tifxyz \
    --python-version 3.12.12 --python-platform x86_64-manylinux_2_39 \
    --torch-backend "$backend" --generate-hashes \
    --build-constraints docker/build-constraints.txt -o "docker/cli-$backend.txt"
  uv pip compile docker/lasagna.in --python-version 3.14.0 \
    --python-platform x86_64-manylinux_2_39 --torch-backend "$backend" \
    --generate-hashes --build-constraints docker/build-constraints.txt \
    -o "docker/lasagna-$backend.txt"
done
```
