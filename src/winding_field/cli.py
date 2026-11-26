"""Installed command entry point."""

import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import sys

# OpenMP threads (PyTorch on CPU) that wait for work sleep instead of spinning, so several
# runs on one host do not starve each other: five concurrent field stages with 4 threads
# took 133-205 s with the default spin and 9 s with PASSIVE. Must be set before torch loads;
# an explicit user setting wins. Worker processes inherit it.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from . import __version__
from .io import Volume, resolve_source
from .flattening.selection import EXPORT_METHODS, RUN_METHODS, SCOPES, SOLVERS

TOOLS = {
    "field-advice": (
        "field_advice",
        "read-only consultative field progress; never triggers scientific work",
    ),
    "field-converge": (
        "field_convergence",
        "opt-in global field convergence; legacy routes remain unchanged",
    ),
    "constant-sheet": (
        "flattening.constant_sheet",
        "prepare, flatten, validate and render exact constant-W sheets",
    ),
    "sheet-atlas": (
        "flattening.sheet_atlas",
        "package and query side-by-side constant-W CT sheets",
    ),
    "surface-export": ("surface_export", "export a level set, flatten and sample CT"),
    "surface-atlas": ("surface_atlas", "build bounded surface charts and previews"),
    "surface-probe": ("surface_probe", "freeze a prior seed and inspect its surface"),
    "surface-geometry-check": (
        "surface_geometry_check",
        "export TIFXYZ and invoke a pinned geometry gate",
    ),
    "ct-visuals": ("ct_visuals", "render frozen CT planes and field contours"),
    "patch-visuals": ("flattening.patch_visuals", "evaluate and plot human-patch residuals"),
    "run-watch": ("run_watch", "supervise a job with timeout and storage limits"),
    "scroll": ("scroll", "whole-scroll fields from overlapping tiles; full-resolution CT renders"),
}


def resolve_device(device: str) -> str:
    """Freeze automatic selection before starting; explicit and saved devices are never replaced."""
    if device != "auto":
        return device
    import torch

    for name, available in (
        ("cuda", torch.cuda.is_available),
        ("mps", torch.backends.mps.is_available),
    ):
        if not available():
            continue
        try:
            # The core's norm reduces in FP64 even for float32 fields (unsupported by current MPS).
            float(torch.linalg.vector_norm(torch.ones(1, device=name), dtype=torch.float64))
        except (RuntimeError, TypeError):
            continue
        return name
    return "cpu"


def tifxyz_exit_code(result):
    status = result.get("status")
    if status in ("READY", "TIFXYZ_VERIFIED", "COMPLETE", "PAUSED"):
        return 0
    if status in ("PARTIAL", "TIFXYZ_PARTIAL_VERIFIED"):
        return 2
    if status == "BLOCKED":
        reasons = result.get("tifxyz_result", result).get("reasons", [])
        if not reasons:
            return 1
        resource = {
            "RUNTIME_UNSUPPORTED",
            "RSS_OBSERVATION",
            "CPU_AFFINITY",
            "RTREE_UNAVAILABLE",
            "BUDGET_ENVELOPE",
            "WORKER_RESOURCE_EXIT",
            "ACCELERATOR_UNAVAILABLE",
        }
        return (
            75
            if any(
                r.get("code", "") in resource
                or r.get("code", "").endswith(("_LIMIT", "_BUDGET", "_HEADROOM"))
                for r in reasons
            )
            else 2
        )
    return 1


def dependency_preflight(config):
    """Inspect selected inference dependencies without installation or model execution."""
    from . import lasagna, models
    from .config import validate_config

    config = dict(config, prediction=config.get("prediction") or "qual-v2")
    validate_config(config)
    checks = {
        name: dict(status="NOT_REQUIRED")
        for name in ("lasagna_runtime", "lasagna_checkpoint", "fibre_checkpoint")
    }
    weights = []
    if not config.get("lasagna_input") and (
        config.get("fibres") or config["prediction"] != "classical"
    ):
        python = str(
            Path(config.get("lasagna_python") or lasagna.DEFAULT_PYTHON).expanduser().absolute()
        )
        runtime = dict(executable=python)
        try:
            lasagna.runtime_identity(python)
            runtime["status"] = "READY"
        except FileNotFoundError as error:
            runtime.update(status="MISSING", error=str(error))
        except (ValueError, OSError, RuntimeError) as error:
            runtime.update(status="INCOMPATIBLE", error=str(error))
        checks["lasagna_runtime"] = runtime
        weights.append(("lasagna_checkpoint", "lasagna", config.get("lasagna_model_path")))
    if not config.get("fibres") and config["prediction"] != "classical":
        weights.append(("fibre_checkpoint", config["prediction"], config.get("model_path")))
    for name, prediction, path in weights:
        check = dict(prediction=prediction)
        try:
            checkpoint = models.checkpoint(path, offline=True, prediction=prediction)
            check.update(status="READY", path=str(checkpoint))
        except FileNotFoundError as error:
            check.update(status="MISSING", error=str(error))
        except (ValueError, OSError) as error:
            check.update(status="INVALID", error=str(error))
        checks[name] = check
    return dict(
        ready=all(check["status"] in ("READY", "NOT_REQUIRED") for check in checks.values()),
        checks=checks,
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "prepared":
        from .field_workflow import main as prepared_main

        return prepared_main(argv[1:])
    if argv and argv[0] in TOOLS:
        module = importlib.import_module("." + TOOLS[argv[0]][0], __package__)
        try:
            return module.main(argv[1:])
        except (ValueError, OSError, RuntimeError) as error:
            print(f"winding-field {argv[0]}: {error}", file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(prog="winding-field")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "prepared", help="build/resume prepared fields, mesh and render (prepared --help)"
    )
    for command, (_, help_text) in TOOLS.items():
        sub.add_parser(command, help=help_text)
    info = sub.add_parser("inspect", help="inspect exact original CT geometry")
    info.add_argument("source", help="CT Zarr path or URL")
    info.add_argument(
        "--voxel-um",
        type=float,
        help="micrometre scale for CT geometry lacking units; not field/output spacing",
    )
    info.add_argument(
        "--level",
        type=int,
        default=0,
        help="CT Zarr pyramid level (default: 0); not a winding W value",
    )
    for command in ("run", "plan"):
        p = sub.add_parser(
            command,
            help="run the whole pipeline"
            if command == "run"
            else "inspect source and space estimates",
            description="Start a new CT run and build the selected field/surface products. Use a fresh output directory; continue existing work with resume."
            if command == "run"
            else "Estimate source geometry and disk use without creating the output or running model inference. Use the same options with run after checking the plan.",
        )
        p.add_argument("--source", help="CT Zarr path or URL; alternatively select --scroll/--scan")
        p.add_argument("--scroll", help="public catalog scroll identifier")
        p.add_argument("--scan", help="public catalog scan identifier")
        p.add_argument(
            "--output",
            required=True,
            help="run directory for configuration, checkpoints and products",
        )
        p.add_argument(
            "--reuse-from",
            help="completed imported preparation/fibres from a prior run; new output required",
        )
        p.add_argument(
            "--reuse-from-sha256",
            help="SHA256 of the selected reuse source RUN.json; required with --reuse-from",
        )
        p.add_argument("--local", help="local mirror of source CT chunks")
        p.add_argument("--cache", help="CT chunk cache directory")
        p.add_argument(
            "--cache-gib",
            type=float,
            default=2.0,
            help="CT chunk cache budget in GiB (default: 2); not total RAM",
        )
        p.add_argument(
            "--reserve-gib",
            type=float,
            default=1.0,
            help="free disk reserve in GiB (default: 1); not RAM",
        )
        p.add_argument(
            "--ct-mode",
            choices=["auto", "download", "stream"],
            default="auto",
            help="CT access: auto chooses by disk estimate, download prefetches, stream reads on demand",
        )
        p.add_argument(
            "--offline",
            action="store_true",
            help="disable CT network access; provision model weights and runtime separately",
        )
        p.add_argument(
            "--voxel-um",
            type=float,
            help="micrometre scale for CT geometry lacking units; not field/output spacing",
        )
        p.add_argument(
            "--level",
            type=int,
            default=0,
            help="CT Zarr pyramid level (default: 0); not a winding W value",
        )
        p.add_argument(
            "--roi",
            nargs=6,
            type=int,
            metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"),
            help="half-open CT voxel bounds at the selected pyramid level; default: whole volume",
        )
        p.add_argument(
            "--stride",
            type=int,
            default=1,
            help="CT observation sampling step: 1..8; 16 or 32 require both --fibres and --lasagna-input; keep 1 for fine inference",
        )
        p.add_argument(
            "--field-spacing-um",
            type=float,
            help="solve-grid pitch in micrometres: exact normally, maximum in bounded preparation; default: observation pitch",
        )
        p.add_argument(
            "--bounded-preparation",
            action="store_true",
            help="recompute Lasagna/Qual windows; requires Qual V2, stride 1, --field-spacing-um and no imported inputs",
        )
        p.add_argument(
            "--bounded-gib",
            type=float,
            default=12.0,
            help="bounded producer retained/scratch allowance, excluding CT cache and disk reserve (default: 12 GiB)",
        )
        p.add_argument(
            "--fibre-downsample",
            type=int,
            choices=[2, 4],
            default=2,
            help="post-prediction fibre grid reduction per axis (default: 2); 4 requires bounded preparation; model input and CT resolution unchanged",
        )
        p.add_argument(
            "--chunk",
            type=int,
            default=64,
            help="observation tile edge in sampled voxels (default: 64)",
        )
        p.add_argument(
            "--prediction",
            choices=["classical", "sd2", "qual-v2"],
            default="qual-v2",
            help="fibre predictor (default: qual-v2); learned predictors use Lasagna surfaces, classical uses CT intensity",
        )
        p.add_argument("--model-path", help="local checkpoint for the selected fibre predictor")
        p.add_argument(
            "--device",
            default="auto",
            help="PyTorch prediction/field device: auto (default) selects usable CUDA, then MPS, then CPU; or an explicit device such as cpu or cuda:0",
        )
        p.add_argument("--fibres", help="external AFV; replaces fibre inference/tracing")
        p.add_argument("--fibres-geometry", help="explicit AFV XYZ-to-world geometry JSON")
        p.add_argument(
            "--import-workers",
            type=int,
            default=1,
            help="local AFV/Lasagna compute workers: 1 serial (default), 0 resource-admitted auto; parent writes in order",
        )
        p.add_argument(
            "--afv-clip-backend",
            choices=["auto", "native", "numpy", "reference"],
            default=argparse.SUPPRESS,
            help="AFV support clipping: auto (native when buildable, default), native (required), "
            "numpy, or reference (per-tile cache, no snapshot); all keep exact fragments",
        )
        p.add_argument("--lasagna-python", help="isolated official Python 3.14 runtime")
        p.add_argument("--lasagna-model-path", help="local pinned Lasagna surface checkpoint")
        from .config import LASAGNA_DOWNSAMPLE

        for name, default in LASAGNA_DOWNSAMPLE.items():
            p.add_argument(
                "--lasagna-" + name.replace("_", "-") + "-downsample",
                type=int,
                choices=[1, 2, 4, 8],
                default=argparse.SUPPRESS,
                help=f"{name} prediction output reduction per axis (default: {default}); nx/ny must match; model input and CT unchanged",
            )
        p.add_argument(
            "--lasagna-input",
            help="local/HTTP/S3 Lasagna v2 manifest; skips surface inference; explicit physical geometry uses calibrated interpolation, otherwise binds to the selected CT base grid",
        )
        p.add_argument(
            "--lasagna-geometry", help="explicit Lasagna base-grid XYZ-to-world geometry JSON"
        )
        from .input_settings import import_arguments

        import_arguments(p)
        p.add_argument(
            "--threshold",
            type=float,
            help="classical predictor CT intensity threshold; default: calibrated Otsu; unused by learned predictors",
        )
        p.add_argument(
            "--invalid-value",
            type=float,
            help="acquisition fill value to exclude; zero remains valid unless declared",
        )
        p.add_argument(
            "--field-chunk",
            type=int,
            default=64,
            help="field solver block edge in field-grid samples (default: 64); at least the "
            "largest grid dimension solves the whole field as one block",
        )
        p.add_argument(
            "--sweeps",
            type=int,
            default=4,
            help="field solver sweep budget (default: 4); not global convergence",
        )
        p.add_argument(
            "--field-advice",
            action="store_true",
            default=argparse.SUPPRESS,
            help="opt-in consultative diagnostics only; requires observer-capable development core",
        )
        p.add_argument(
            "--prepared-cache-bytes",
            type=int,
            default=0,
            help="retained field CPU/device workspaces; 0 disables (not a peak-memory cap)",
        )
        p.add_argument(
            "--constraint-cache-bytes",
            type=int,
            default=0,
            help="retained static fibre restrictions; 0 disables; retained bytes are not a total RSS cap",
        )
        p.add_argument(
            "--observation-cache-bytes",
            type=int,
            default=argparse.SUPPRESS,
            help="retained static per-block field observations across sweeps (default: 0, disabled); not a total RSS cap",
        )
        p.add_argument(
            "--field-cache-bytes",
            type=int,
            default=argparse.SUPPRESS,
            help="decoded field-checkpoint chunk cache for block solves (default: 64 MiB); not a total RSS cap",
        )
        p.add_argument(
            "--field-pair-scope",
            choices=("block", "global"),
            default=argparse.SUPPRESS,
            help="experimental, ordinary fields with several blocks: block samples fibre pairs "
            "inside each block crop; global samples them once over the whole grid and restricts "
            "them per crop with exterior terms, so the blocks solve the global problem "
            "(default: global with --field-coarse-factor, else block)",
        )
        p.add_argument(
            "--field-memory",
            choices=("auto", "disk", "field", "all"),
            default=argparse.SUPPRESS,
            help="host RAM use of the field solve (default auto: measured at start). disk: read and "
            "write every block on disk; field: keep the whole field in RAM and checkpoint once per "
            "sweep in the background, spending what is left on observations; all: field and every "
            "block's observations in RAM. Explicit --field-cache-bytes/--observation-cache-bytes win",
        )
        p.add_argument(
            "--field-cpu-workers",
            type=int,
            default=0,
            help="opt-in ordinary-field CPU preparation processes; solves/writes stay ordered and serial; 0 disables",
        )
        p.add_argument(
            "--field-cpu-buffer-bytes",
            type=int,
            default=0,
            help="explicit CPU preparation/IPC buffer budget; requires --field-cpu-workers; not a total RAM or RSS cap",
        )
        p.add_argument(
            "--maxiter",
            type=int,
            default=2000,
            help="iteration limit per local field solve (default: 2000)",
        )
        p.add_argument(
            "--rtol",
            type=float,
            default=1e-6,
            help="relative tolerance of local field solves (default: 1e-6)",
        )
        p.add_argument(
            "--preconditioner",
            choices=["jacobi", "spectral", "amg"],
            default=argparse.SUPPRESS,
            help=(
                "default: jacobi; amg (algebraic multigrid) needs far fewer iterations on large "
                "blocks; spectral is rejected for block workflows"
            ),
        )
        p.add_argument(
            "--nonconvergence-policy",
            choices=("fail", "continue"),
            default="fail",
            help="fail at the first nonconverged block or continue the full field budget; numerical breakdowns always fail",
        )
        p.add_argument(
            "--seed-stride",
            type=int,
            default=8,
            help="fibre seed grid spacing in observation samples (default: 8)",
        )
        p.add_argument(
            "--fibre-weight",
            type=float,
            default=1.0,
            help="field solver fibre constraint weight (default: 1)",
        )
        p.add_argument(
            "--levels",
            nargs="+",
            type=float,
            help="raw field isovalues (W * sheet_step), e.g. --levels -31; not CT pyramid levels",
        )
        p.add_argument(
            "--mesh-tile",
            type=int,
            default=64,
            help="mesh extraction tile edge in field-grid cells (default: 64)",
        )
        p.add_argument(
            "--chart-faces",
            type=int,
            default=1024,
            help="native/atlas local chart face budget (default: 1024); not original global flattening",
        )
        p.add_argument(
            "--field-prior",
            default=argparse.SUPPRESS,
            help="completed field run whose W seeds and gauges this field (e.g. a coarse run "
            "for a fine --roi tile); same frame required",
        )
        p.add_argument(
            "--field-halo",
            type=int,
            default=argparse.SUPPRESS,
            help="field nodes each solver block shares with its neighbours (default: 2); "
            "a wider overlap reaches the global solution in fewer sweeps",
        )
        p.add_argument(
            "--field-coarse-factor",
            type=int,
            default=argparse.SUPPRESS,
            help="experimental, 0 (default) or >= 2: open every field sweep with a global coarse correction on "
            "every F-th node (two-level Schwarz); large-scale structure then crosses all blocks "
            "in each sweep instead of one overlap per sweep",
        )
        p.add_argument(
            "--field-prior-calibration",
            choices=["own", "inherit"],
            default=argparse.SUPPRESS,
            help="sheet orientation and spacing calibration with --field-prior: own (default) "
            "estimates it from this run's ROI; inherit uses the prior's, so a tile orients "
            "normals from the scroll centre like its coarse",
        )
        p.add_argument(
            "--field-orientation",
            choices=["radial", "prior-gradient"],
            default=argparse.SUPPRESS,
            help="how sheet normals are signed outward: radial (default) from the calibration "
            "centre; prior-gradient from --field-prior's gradient where the prior is supported "
            "(radial elsewhere), for flattened scrolls whose sheets near the pressed core are "
            "nearly perpendicular to the radial direction",
        )
        p.add_argument(
            "--field-scope",
            choices=["all", "supported"],
            default=argparse.SUPPRESS,
            help="field chunks to solve and store: all (default) or supported, which skips "
            "chunks without observations and holds them at --field-prior",
        )
        p.add_argument(
            "--level-workers",
            type=int,
            default=argparse.SUPPRESS,
            help="native: compute this many W levels in parallel processes (default: 1); "
            "each needs its own memory; outputs equal a serial run",
        )
        p.add_argument(
            "--threads",
            type=int,
            default=4,
            help="PyTorch CPU thread count (default: 4); not a global worker limit",
        )
        p.add_argument(
            "--previews",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="legacy atlas chart previews; final CT renders are retained",
        )
        p.add_argument(
            "--surface-output",
            choices=["native", "atlas", "tifxyz"],
            default="native",
            help="pipeline branch: native surfaces (default), legacy atlas, or experimental direct-field tifxyz; not an output file format",
        )
        p.add_argument(
            "--flattening-method",
            choices=list(RUN_METHODS),
            help="slim: Flatboi SLIM (default for new native runs on Linux x86_64), default: tangent projection/LSCM charts, original: global LSCM; independent of field resolution",
        )
        p.add_argument(
            "--flattening-config",
            help="original-mapping JSON: output_frame, optional pitch and memory_gib; requires --flattening-method original",
        )
        p.add_argument(
            "--coverage-target",
            type=float,
            default=1.0,
            help="native mapped mesh-area fraction in (0,1] (default: 1); not observed CT coverage",
        )
        p.add_argument(
            "--surface-time-limit",
            dest="surface_time_limit_s",
            type=float,
            default=1800.0,
            help="cumulative active seconds per native W (default: 1800)",
        )
        # 2.0: mesh-derived TIFXYZ is always written; --export-tifxyz/--no-export-tifxyz
        # were removed. The config key stays so 1.x run identities and resumes are unchanged.
        p.set_defaults(export_tifxyz=True)
        p.add_argument(
            "--no-ct-texture",
            dest="ct_texture",
            action="store_false",
            default=argparse.SUPPRESS,
            help="native only: flatten and write mesh-derived TIFXYZ without sampling CT for the "
            "flattened texture",
        )
        p.add_argument(
            "--tifxyz-config",
            help="Linux-only direct-export JSON; requires --surface-output tifxyz; frozen by content",
        )
        from .export_settings import tifxyz_arguments, auto_float

        tifxyz_arguments(p, prefix="tifxyz-", include_levels=False)
        p.add_argument(
            "--pitch",
            type=auto_float,
            default=argparse.SUPPRESS,
            help="original mapping WORLD pitch, or auto",
        )
        p.add_argument("--original-memory-gib", type=float, default=argparse.SUPPRESS)
        p.add_argument(
            "--until",
            choices=[
                "prepare",
                "fibres",
                "field",
                "mesh",
                "flatten",
                "assemble",
                "render",
                "tifxyz",
                "original",
            ],
            help="default: terminal stage of the selected branch",
        )
    for command in (
        "resume",
        "status",
        "prepare",
        "fibres",
        "field",
        "mesh",
        "flatten",
        "assemble",
        "render",
    ):
        p = sub.add_parser(
            command,
            help="continue or inspect an existing run",
            description="Resume an existing run with its saved inputs, settings and runtime. Remove RUN/STOP first; published levels are unchanged."
            if command == "resume"
            else None,
        )
        p.add_argument("run", help="existing run directory containing RUN.json")
        if command != "status":
            p.add_argument(
                "--offline",
                action="store_true",
                help="disable network for this continuation without changing saved identities",
            )
        if command == "resume":
            p.add_argument(
                "--field-sweeps",
                type=int,
                help="absolute total sweep budget: continue a field (also a finished ordinary one) from its checkpoint; retries do not add sweeps",
            )
            p.add_argument(
                "--extra-surface-seconds",
                type=float,
                default=0.0,
                help="explicit extra native budget; original elapsed/target retained",
            )
    p = sub.add_parser(
        "tifxyz",
        help="experimental direct export of an existing field; Linux runtime",
        description="Export XYZ directly from a saved field in an admitted Linux worker. This experimental route does not sample CT; it is separate from mesh-derived TIFFs. Supply typed request flags or optional --config.",
    )
    p.add_argument("field_run", help="existing run directory containing the field")
    p.add_argument("--output", required=True, help="direct-field TIFXYZ export directory")
    p.add_argument(
        "--config",
        help="optional direct-field export request JSON; explicit request flags override",
    )
    p.add_argument(
        "--resume", action="store_true", help="continue the saved export request in --output"
    )
    from .export_settings import tifxyz_arguments

    tifxyz_arguments(p)
    p = sub.add_parser(
        "verify-tifxyz",
        help="read-only verification in an admitted Linux worker",
        description="Verify a direct-field READY or PARTIAL publication in an admitted Linux worker. Rechecks geometry, coverage and file integrity; does not modify the export or establish physical sheet correctness.",
    )
    p.add_argument(
        "output", help="direct-field export directory, not a native/original/UVW TIFF directory"
    )
    p = sub.add_parser("extract-range", help="extract a calibrated continuous W range")
    p.add_argument("--config", help="optional immutable field/geometry/umbilicus recipe JSON")
    from .cli_settings import field_arguments
    from .export_settings import flattening_arguments, auto_float

    field_arguments(p)
    group = p.add_argument_group("range overrides", argument_default=argparse.SUPPRESS)
    group.add_argument(
        "--umbilicus-point", nargs=3, type=float, action="append", help="WORLD XYZ in increasing Z"
    )
    group.add_argument("--handedness", type=int, choices=(-1, 1))
    group.add_argument("--w-ref", type=float)
    group.add_argument("--theta-ref", type=float)
    group.add_argument("--samples-per-w", type=int)
    group.add_argument("--max-edge-world", type=auto_float)
    p.add_argument(
        "--w-range",
        nargs=2,
        required=True,
        metavar=("A", "B"),
        help="exact decimal bounds on the global W lattice",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("join-ranges", help="join compatible contiguous continuous ranges exactly")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", required=True)
    p = sub.add_parser(
        "export-sheets",
        help="LSCM/SLIM on patches or whole sheets; historical recipes also available",
        description="Export flattened CT from a field or mesh. The solver/scope matrix and historical patches require Linux; original mapping also supports macOS. Choose --flattening-solver and --scope together, or a historical --flattening-method. LSCM needs no Flatboi; SLIM requires the pinned binary and cgroup limits. Outputs remain diagnostic.",
    )
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument(
        "--run",
        help="completed field run; infer saved inputs and original mapping defaults without manual JSON",
    )
    inputs.add_argument(
        "--field-config",
        help="field path/hash and geometry JSON; requires --levels; not supported by historical patches",
    )
    inputs.add_argument(
        "--mesh-bundle",
        help="immutable mesh bundle or verified RANGE wrapper; original mapping rejects continuous ranges",
    )
    field_arguments(p)
    flattening_arguments(p)
    p.add_argument(
        "--levels",
        nargs="+",
        type=float,
        help="raw field isovalues (W * sheet_step); required with --field-config, forbidden with --mesh-bundle",
    )
    p.add_argument("--source", help="original CT Zarr; required except with --run")
    p.add_argument(
        "--level",
        type=int,
        help="CT Zarr pyramid level (default: 0, or saved run level); not a winding W value",
    )
    p.add_argument(
        "--voxel-um",
        type=float,
        help="micrometre scale for CT geometry lacking units; not field/output spacing",
    )
    p.add_argument("--output", required=True, help="sheet export directory")
    p.add_argument(
        "--flattening-config",
        help="optional JSON: output_frame and render settings; explicit flags override; slim requires app_run",
    )
    p.add_argument(
        "--flattening-method",
        choices=list(EXPORT_METHODS),
        help="historical recipe; cannot combine with --flattening-solver/--scope",
    )
    p.add_argument(
        "--flattening-solver",
        choices=list(SOLVERS),
        help="lscm-slim (default on Linux x86_64: global LSCM seed, then Flatboi SLIM), "
        "lscm, or slim (Flatboi from a mean-value seed); default lscm elsewhere",
    )
    p.add_argument(
        "--scope",
        choices=list(SCOPES),
        help="independent patches or whole connected sheets (default: sheet)",
    )
    # 2.0: mesh-derived float32 XYZ TIFFs and mask are always written alongside CT renders.
    p.set_defaults(export_tifxyz=True)
    p.add_argument(
        "--stages",
        choices=("all", "uv", "ct"),
        default="all",
        help="all: solve and render; uv: publish validated UV without opening the CT; ct: render a published UV without solving",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="repeat identical inputs/settings; unavailable for historical patches",
    )
    p = sub.add_parser(
        "export-uvw", help="aligned diagnostic W band from one original global UV map"
    )
    p.add_argument("--field-config", help="optional SHA-bound field descriptor JSON")
    field_arguments(p)
    p.add_argument(
        "--reference-sheet",
        required=True,
        help="original/level-NNNN with independent publication ledger",
    )
    p.add_argument(
        "--levels",
        nargs="+",
        type=float,
        required=True,
        help="finite W targets; band must contain reference W",
    )
    p.add_argument("--source", required=True, help="same physical CT source as reference")
    p.add_argument("--voxel-um", type=float, help="CT units when metadata lacks them")
    p.add_argument("--level", type=int, default=0, help="CT pyramid level, not winding W")
    p.add_argument(
        "--canvas-samples",
        type=int,
        default=512,
        help="shared square UV grid edge, 2..1024 (default: 512)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="read-only geometry processes, 1..8; CT/publication stay in coordinator",
    )
    p.add_argument(
        "--cache-mib",
        type=int,
        default=256,
        help="total decoded CT + field-region RAM cache budget, 0..8192 MiB (default: 256)",
    )
    p.add_argument("--output", required=True)
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume identical source/settings; remove output/STOP to continue",
    )
    p = sub.add_parser("demo", help="offline synthetic original-CT demonstration")
    p.add_argument("--output", required=True, help="synthetic demo run directory")
    p.add_argument(
        "--prediction",
        choices=["classical", "sd2"],
        default="classical",
        help="demo observation predictor (default: classical)",
    )
    p.add_argument("--model-path", help="local SD2 checkpoint path")
    p.add_argument("--device", help="PyTorch device (default: CUDA when available, otherwise CPU)")
    p = sub.add_parser("doctor", help="report CLI runtime; selected inference checks are read-only")
    p.add_argument(
        "--prediction",
        choices=["classical", "sd2", "qual-v2"],
        help="check dependencies for this fibre predictor without downloads/inference",
    )
    p.add_argument(
        "--lasagna-python",
        help="explicit isolated Lasagna Python; preserve the venv executable path",
    )
    p.add_argument("--lasagna-model-path", help="local pinned Lasagna checkpoint to verify")
    p.add_argument("--model-path", help="local pinned fibre checkpoint to verify")
    p.add_argument(
        "--lasagna-input",
        help="imported surfaces; no Lasagna inference runtime/checkpoint required",
    )
    p.add_argument("--fibres", help="imported AFV; no fibre inference checkpoint required")
    p = sub.add_parser("models", help="download and verify a pinned checkpoint")
    p.add_argument("--path", help="checkpoint destination; default: model cache")
    p.add_argument(
        "--prediction",
        choices=["qual-v2", "sd2", "lasagna"],
        default="qual-v2",
        help="checkpoint to download (default: qual-v2)",
    )
    from .cli_settings import parse_options

    args = parse_options(
        parser,
        argv,
        paths=(
            "source",
            "output",
            "run",
            "local",
            "cache",
            "model_path",
            "fibres",
            "fibres_geometry",
            "lasagna_model_path",
            "lasagna_geometry",
            "lasagna_input",
            "lasagna_python",
            "path",
            "field_config",
            "flattening_config",
            "tifxyz_config",
            "reference_sheet",
            "mesh_bundle",
            "config",
            "field_run",
            "inputs",
            "fibres_source",
            "lasagna_source",
            "lasagna_ct_source",
            ("lasagna_channel", 2),
            ("lasagna_metadata_pin", 0),
        ),
        lexical_paths=("lasagna_python",),
    )
    direct = False
    native = False
    try:
        if args.command == "inspect":
            volume = Volume(args.source, voxel_um=args.voxel_um, level=args.level)
            try:
                result = volume.identity
            finally:
                volume.close()
        elif args.command == "doctor":
            import platform
            import torch

            result = dict(
                version=__version__,
                python=platform.python_version(),
                platform=platform.platform(),
                torch=torch.__version__,
                cuda=torch.cuda.is_available(),
                reader="vc_reader-zarr-v2-checked-v1",
            )
            if any(value is not None for name, value in vars(args).items() if name != "command"):
                result["preflight"] = dependency_preflight(vars(args))
        elif args.command == "models":
            from .models import checkpoint, model_sha

            result = dict(
                path=str(checkpoint(args.path, prediction=args.prediction)),
                prediction=args.prediction,
                sha256=model_sha(args.prediction),
            )
        elif args.command in ("run", "plan"):
            from .workflow import run, plan, stage_path
            from .config import validate_config
            from .export_settings import tifxyz_request, pop_tifxyz_options, output_frame_overrides
            from .cli_settings import merge_explicit, read_settings
            from .util import digest

            config = vars(args).copy()
            command = config.pop("command")
            until = config.pop("until")
            from .input_settings import import_requests, IMPORT_OPTIONS

            declarations = import_requests(args)
            for key in IMPORT_OPTIONS:
                config.pop(key, None)
            config.update(declarations)
            config["device"] = resolve_device(config["device"])
            if config["flattening_method"] is None:
                from .flattening.flatboi import supported_platform

                # SLIM is Flatboi (Linux x86_64 only); elsewhere native runs default to LSCM.
                config["flattening_method"] = (
                    "slim"
                    if config["surface_output"] == "native" and supported_platform()
                    else "default"
                )
            original = config.pop("flattening_config")
            original_flags = {
                key: config.pop(flag)
                for flag, key in (("pitch", "pitch"), ("original_memory_gib", "memory_gib"))
                if flag in config
            }
            if config["flattening_method"] == "original":
                from .flattening.original_pipeline import settings

                if any(
                    name in config
                    for name in (
                        "output_frame",
                        "output_shape_zyx",
                        "output_source_identity_sha256",
                    )
                ):
                    raise ValueError(
                        "direct TIFXYZ frame selection/shape/source pin is not an original-mapping setting"
                    )
                frame = output_frame_overrides(args)
                if frame:
                    original_flags["output_frame"] = frame
                config["original_request"] = settings(
                    merge_explicit(read_settings(original) if original else {}, original_flags)
                )
            elif original or original_flags:
                raise ValueError("--flattening-config requires --flattening-method original")
            if config.get("ct_texture") is False:
                # The mesh-only output is the TIFXYZ, so the flag implies it.
                config["export_tifxyz"] = True
            request = config.pop("tifxyz_config")
            has_tifxyz = any(name.startswith("tifxyz_") for name in config) or request is not None
            if config["surface_output"] == "tifxyz":
                config["tifxyz_request"] = tifxyz_request(args, path=request, prefix="tifxyz-")
                config["tifxyz_request_sha256"] = digest(config["tifxyz_request"])
            elif has_tifxyz:
                raise ValueError("--tifxyz settings require --surface-output tifxyz")
            elif config["flattening_method"] != "original" and any(
                name.startswith("output_") for name in config
            ):
                raise ValueError("output frame flags require original mapping or direct TIFXYZ")
            pop_tifxyz_options(config, prefix="tifxyz-")
            direct = config["surface_output"] == "tifxyz" and command == "run"
            native = (
                config["surface_output"] == "native"
                and config["flattening_method"] != "original"
                and command == "run"
            )
            validate_config(config)
            stages = stage_path(config)
            until = until or stages[-1][0]
            if until not in dict(stages) and not (
                config["surface_output"] == "native"
                and until in ("mesh", "flatten", "assemble", "render")
            ):
                raise ValueError(
                    "--until " + until + " is incompatible with the selected surface output"
                )
            config["output"] = str(Path(config["output"]).expanduser().absolute())
            if config.get("reuse_from"):
                config["reuse_from"] = str(Path(config["reuse_from"]).expanduser().absolute())
            for key in (
                "local",
                "cache",
                "model_path",
                "fibres",
                "fibres_geometry",
                "lasagna_model_path",
                "lasagna_geometry",
            ):
                if config.get(key) and not isinstance(config[key], dict):
                    config[key] = str(Path(config[key]).expanduser().resolve())
            if config.get("lasagna_input") and not isinstance(config["lasagna_input"], dict):
                from .lasagna_stream import location

                config["lasagna_input"] = location(config["lasagna_input"])
            if config.get("reuse_from"):
                config["reuse_from"] = str(Path(config["reuse_from"]).expanduser().absolute())
            if config.get("lasagna_python"):
                # Resolving the venv executable symlink would select the base interpreter.
                config["lasagna_python"] = str(
                    Path(config["lasagna_python"]).expanduser().absolute()
                )
            for key in ("cache", "reserve", "bounded"):
                value = config.pop(key + "_gib")
                if not 0 <= value <= 1e6:
                    raise ValueError("invalid storage budget")
                config[key + "_bytes"] = int(value * 2**30)
            roi = config.pop("roi")
            if roi:
                config.update(roi_lo=roi[::2], roi_hi=roi[1::2])
            validate_config(config)
            selected = resolve_source(
                config.pop("scroll"),
                config.get("source"),
                config.pop("scan"),
                cache=config.get("cache"),
                offline=config["offline"],
            )
            config["source"] = selected["source"]
            if config.get("voxel_um") is None:
                config["voxel_um"] = selected.get("voxel_um")
            if not config["source"].startswith(("http://", "https://", "s3://")):
                config["source"] = str(Path(config["source"]).expanduser().resolve())
            result = plan(config) if command == "plan" else run(config, until=until)
        elif args.command == "export-uvw":
            from .uvw_pipeline import export_existing
            from .export_settings import field_request
            from .cli_settings import field_overrides

            if field_overrides(args):
                args.resolved_field = field_request(args)
            interrupted = False

            def handle_uvw(signum, frame):
                nonlocal interrupted
                interrupted = True

            previous = {
                sig: signal.signal(sig, handle_uvw) for sig in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                result = export_existing(args, stop=lambda: interrupted)
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        elif args.command == "extract-range":
            from .range_surface import extract_range
            from .export_settings import range_request

            result = extract_range(
                range_request(args), args.w_range, args.output, resume=args.resume
            )
        elif args.command == "join-ranges":
            from .range_surface import join_ranges

            result = join_ranges(args.inputs, args.output)
        elif args.command == "export-sheets":
            stages = getattr(args, "stages", "all")
            if stages != "all" and args.run:
                raise ValueError("--stages requires an explicit mesh bundle or field source")
            if stages != "all" and args.flattening_method:
                raise ValueError("--stages requires --flattening-solver and --scope")
            if not args.run:
                if stages == "uv":
                    if (
                        args.source is not None
                        or args.level is not None
                        or args.voxel_um is not None
                    ):
                        raise ValueError(
                            "UV-only publication must not depend on the CT: omit --source, --level and --voxel-um"
                        )
                else:
                    if not args.source:
                        raise ValueError("--source is required without --run")
                    if args.level is None:
                        args.level = 0
            if args.run:
                from .flattening.run_sheet_export import export_run as export_existing
            elif args.flattening_method:
                if args.flattening_solver is not None or args.scope is not None:
                    raise ValueError(
                        "--flattening-method cannot combine with --flattening-solver/--scope"
                    )
                if args.flattening_method == "patches":
                    from .flattening.patches import export_existing
                else:
                    from .flattening.original_pipeline import export_existing
            else:
                from .flattening.selection import DEFAULT_SCOPE, default_solver

                args.flattening_solver = args.flattening_solver or default_solver()
                args.scope = args.scope or DEFAULT_SCOPE
                from .flattening.matrix_pipeline import export_existing
            from .export_settings import field_request, flattening_request
            from .cli_settings import field_overrides

            if (args.run or args.mesh_bundle) and field_overrides(args):
                raise ValueError("saved run/mesh sources cannot combine with field overrides")
            args.resolved_field = field_request(args)
            if not args.run:
                args.resolved_flattening = flattening_request(args)
                if (args.resolved_field is None) == (args.mesh_bundle is None):
                    raise ValueError("exactly one field or mesh source required")
                if args.resolved_field is not None:
                    from .surface_sources import FieldSource

                    with FieldSource(
                        args.resolved_field["field"], args.resolved_field["geometry"]
                    ) as source:
                        source.validate(hashes=True)
            result = export_existing(args)
        elif args.command in ("tifxyz", "verify-tifxyz"):
            from .tifxyz import _path, export_field, verify_export
            from .config import tifxyz_settings
            from .export_settings import tifxyz_request

            direct = True
            out = _path(Path(args.output).expanduser())
            interrupted = False

            def handle(signum, frame):
                nonlocal interrupted
                interrupted = True

            previous = {sig: signal.signal(sig, handle) for sig in (signal.SIGINT, signal.SIGTERM)}
            stop = lambda: interrupted or (out / "STOP").exists()
            try:
                if args.command == "verify-tifxyz":
                    result = verify_export(out, stop=stop)
                    if result.get("status") not in (
                        "TIFXYZ_VERIFIED",
                        "TIFXYZ_PARTIAL_VERIFIED",
                        "PAUSED",
                        "BLOCKED",
                        "FAILED",
                    ):
                        raise ValueError("unexpected tifxyz verification result")
                else:
                    source = _path(Path(args.field_run).expanduser())
                    if out == source or source in out.parents:
                        raise ValueError(
                            "standalone output must be outside the read-only field run"
                        )
                    request = tifxyz_request(args)
                    settings = tifxyz_settings(out, request, resume=args.resume)
                    result = export_field(source, out, settings, resume=args.resume, stop=stop)
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            result = dict(
                result,
                experimental=True,
                comparison_status="UNMEASURED",
                comparison_reason="reference OOM",
            )
        elif args.command == "demo":
            from .demo import synthetic
            from .workflow import run

            output = Path(args.output).expanduser().resolve()
            if output.exists():
                raise FileExistsError("demo output already exists; use resume")
            if args.device is None:
                import torch

                args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            source = synthetic(output.with_name(output.name + "-ct.zarr"))
            config = dict(
                source=source,
                output=str(output),
                prediction=args.prediction,
                model_path=args.model_path,
                device=args.device,
                chunk=64,
                field_chunk=64,
                sweeps=1,
                seed_stride=4,
                chart_faces=1024,
                cache=str(output.with_name(output.name + "-cache")),
                reserve_bytes=0,
            )
            result = run(config)
        else:
            from .workflow import run

            root = Path(args.run).expanduser().resolve()
            state = json.loads((root / "RUN.json").read_bytes())
            direct = (
                state["config"].get("surface_output", "atlas") == "tifxyz"
                and args.command != "status"
            )
            native = (
                state["config"].get("surface_output", "atlas") == "native"
                and state["config"].get("flattening_method") != "original"
                and args.command != "status"
            )
            if args.command == "status":
                result = state
                progress = root / "surfaces/STATUS.json"
                if progress.is_file():
                    result = dict(state, native_progress=json.loads(progress.read_bytes()))
                afv_progress = root / "fibres/STATUS.json"
                try:
                    detail = json.loads(afv_progress.read_bytes())
                    if (
                        isinstance(detail, dict)
                        and detail.get("schema") == "afv-import-progress-v1"
                    ):
                        json.dumps(detail, allow_nan=False)
                        result = dict(result, afv_progress=detail)
                except (OSError, ValueError):
                    pass  # Advisory progress must not hide the authoritative run state.
            else:
                result = run(
                    state["config"],
                    resume=True,
                    until=("fibres" if state.get("preparation_only") else None)
                    if args.command == "resume"
                    else args.command,
                    extra_surface_seconds=getattr(args, "extra_surface_seconds", 0.0),
                    field_sweeps=getattr(args, "field_sweeps", None),
                    offline=args.offline,
                )
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "doctor" and not result.get("preflight", {}).get("ready", True):
            return 1
        if native:
            native_result = result.get("native_result", result)
            if result.get("status") == "PAUSED" and native_result.get("reason") in (
                "TIME_LIMIT",
                "RSS_LIMIT",
                "RSS_OBSERVATION",
                "DISK_HEADROOM",
                "SCRATCH_LIMIT",
            ):
                return 75
            return (
                2
                if result.get("status") == "PARTIAL"
                else 1
                if result.get("status") == "FAILED"
                else 0
            )
        return tifxyz_exit_code(result) if direct else 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"winding-field: {error}", file=sys.stderr)
        return 1
