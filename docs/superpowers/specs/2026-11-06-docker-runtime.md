# Reproducible full-CLI containers

## Approved scope

Build Linux x86-64 CPU and NVIDIA CUDA images from one shared Dockerfile.
Both expose the installed `winding-field` CLI, including full-scroll/ROI runs,
independent custom Lasagna/fibre inputs, reused fields, level/range extraction,
LSCM/SLIM × patches/sheet, original/historical exporters, constant-sheet,
atlas, TIFXYZ, inspection, verification and resume commands.

Package the CLI and core in Python 3.12.12 and official Lasagna in Python
3.14.0. Resolve separate CPU/cu128 dependency locks for both environments,
using PyTorch 2.10.0 and torchvision 0.25.0 where required. Retain the official
Lasagna archive URL/subdirectory in its installed `direct_url.json`. Core,
Qual, Lasagna, native reader and model source revisions remain those recorded
by the repository. Downloads must be verified; a digest mismatch stops the build.

## Official native recipe

Five inspected November Villa AppImages do not contain the installed Flatboi
hash currently required by the matrix and constant-sheet commands. The user
approved qualifying an official recipe, preserving previous identities. November
also removed OBJ support; the July archive matches the existing `historical`
binary exactly. Reuse that identity and qualify its checkpointed routes.

Candidate recipe: `historical` from the archived July stable release:

- URL: `https://github.com/ScrollPrize/villa/releases/download/vc3d-archive-stable-363877681/VC3D-fc25b4d-2026-07-31-linux-x86_64.AppImage`
- Archive SHA256: `5da27c3df44c0958390ea752b550756f9d7e2c104840a95c0ba8b8651c7aee9f`
- AppRun SHA256: `c1e0ca2ea1616967f6da5907c60ea3fe114e840834154ccde758856f85510517`
- Flatboi SHA256: `d724cf0cc6940d7c959e36c73ed1a9a977d3b90bab60fc97d519fe370cc63f38`

Extract the complete runtime and libraries to `/opt/vc3d`; use AppRun directly
without FUSE. Add explicit `native_recipe` selection to SLIM matrix settings
and `--native-recipe` to constant-sheet flatten/import. Preserve old defaults.
Keep the existing batch size, energy, tolerance, thread limits, geometry gates,
checkpoint identity, warm-start checks and cgroup guards. Validate constant-sheet
against the two qualified checkpoint recipes only.
Changing a binary cannot resume or import an old binary's checkpoint.

## Container contract

- One Dockerfile, CPU/CUDA build targets selected by a small Bake definition.
- Digest-pinned Ubuntu 24.04 CPU and CUDA 12.8.1 devel/Ubuntu 24.04 bases;
  digest-pinned uv 0.9.17. Freeze Ubuntu packages through a dated snapshot.
- Installed environments at `/opt/winding-field` and `/opt/lasagna`; avoid
  editable installs or runtime dependency installation.
- Non-root default user; allow host UID/GID overrides without a root entrypoint.
- Stable paths: read-only inputs `/data`, persistent outputs `/outputs`, model
  cache `/home/worker/.cache/winding-field/models`, CT cache `/cache`, scratch
  `/scratch`. Mounting models must not hide the Lasagna runtime.
- Link the existing default Lasagna location to `/opt/lasagna` in the image.
- `ENTRYPOINT ["winding-field"]`; `CMD ["--help"]`. Use Docker `--init` and
  explicit memory/swap bounds for native jobs; preserve signal delivery.
- Include CPU TIFXYZ accelerator and CUDA accelerator in the CUDA image;
  retain supported explicit compiler commands and numerical flags.
- Store locks, source/binary provenance and installed environment inventories
  under `/opt/runtime-provenance`; keep model acquisition in the existing CLI.
- Images carry runtime identity. Resume requires the same image and mount paths.

## Acceptance

Build both images for `linux/amd64`; run installed-package checks outside the
source checkout as a non-root user. Check native reader/libigl imports, Lasagna
identity and real inference, model hash validation, demo/resume, small full/ROI
and custom-input workflows, mesh level/range routes, all four solver/scope
combinations, native checkpoint continuation and changed-identity rejection.
Verify mounted outputs survive container replacement and signal cancellation.

Local hardware is macOS ARM64 with an 8 GiB Docker VM. x86-64 emulation works.
The user provided gpu-1 for CUDA verification: two GTX 1660 6GiB GPUs, driver
550.163.01. Use the idle GPU 1 and an approved isolated Docker daemon on
/mnt/fast2. CUDA build/import checks do not establish CUDA execution: record
real inference separately and provide a GPU smoke command.
Small fixtures establish packaging compatibility, not feasibility for a full
fine-resolution scroll or numerical equivalence with the historical Flatboi.
