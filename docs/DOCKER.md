# CPU and CUDA containers

Both images expose the complete installed `winding-field` CLI on **Linux x86-64**.
CPU and CUDA select different PyTorch builds, not different workflow stages.
The CLI uses Python 3.12.12; official Lasagna has its own Python 3.14.0 environment.

Use this page if you want an isolated runtime with the native tools already
installed. Start with CPU for an offline smoke test; choose CUDA for NVIDIA model
inference. Build from the checkout, keep outputs outside the container, and start
with a small ROI. ARM hosts are not qualified for these x86-64 images.

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

# ROI bounds are selected-level voxels, ZYX, upper bounds exclusive.
wf plan --source /data/scan.zarr --voxel-um 8.64 --device cpu \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --cache /cache/roi --until field --output /outputs/roi
wf run --source /data/scan.zarr --voxel-um 8.64 --device cpu \
  --roi 1488 1536 4032 4080 6048 6096 --field-spacing-um 17 \
  --cache /cache/roi --until field --output /outputs/roi

wf status /outputs/roi
wf field /outputs/roi
wf resume /outputs/roi
```

Qual V2 requires uint8, isotropic 8.64–9.6 µm observations. Choose ROI, scan level,
stride and field pitch using the existing [planning instructions](../README.md).
Replace the scan path and ROI with your data. Omit `--roi` only after the
whole-volume plan fits your resources. The README's public full-resolution scan
needs about 134 TB for the default preparation/products; a coarse field alone
does not remove that cost.

| Workflow | Container usage / reference |
| --- | --- |
| Custom fibres | Add `--fibres /data/fibres.afv` and its `--fibres-geometry` descriptor |
| Custom Lasagna | Independently add `--lasagna-input /data/surface.lasagna.json` and, when needed, `--lasagna-geometry`; see [Lasagna imports](LASAGNA.md) |
| Reused fields / prepared workflows | `prepared surfaces`, `export-sheets --field-config`, or `extract-range`; see [README](../README.md) |
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

Historical checks on 2026-11-06 exercised retained CPU/CUDA images, native CT
reads, model inference and STOP/resume. They used small fixtures and derivative
images, not a clean build of the current checkout. Old test counts and image IDs
are not a current CI result; retain the exact image used by each old checkpoint.
See [dated raster measurements](RASTER-PERFORMANCE.md) for timings and scope.

Two limitations matter when choosing an export:

- The curved direct-field fixture was blocked by `COVERAGE_DEPTH_LIMIT`; planar
  success does not establish support for arbitrary curved fields. Inspect the
  current export's result and use [its verification protocol](FORMATS.md#direct-tifxyz-experimental).
- The bundled July `vc_tifxyz2obj` converter retained only 66,305 of 66,821 valid
  fixture nodes in its external readback test. The archive is used for Flatboi,
  not as proof of a lossless TIFXYZ round trip. UVW's float64 TIFFs also face
  [downstream float32 conversion](UVW.md#products).

The current container job mounts tests, examples, tools and the configuration/
provenance files used by tests. This is test-only setup; it does not mount a
source package over the installed CLI. A source-tree test pass does not replace
a fresh container CI run.

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
