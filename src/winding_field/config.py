"""Shared preflight contracts; no source or model I/O."""

import math
from pathlib import Path

import numpy as np
from .flattening.selection import RUN_METHODS


def tifxyz_settings(out, request=None, *, resume=False):
    """Reuse the worker's frozen effective settings; never resolve geometry here."""
    from .tifxyz_contract import Settings, validate_request
    from .tifxyz_field import read_metadata
    from .util import digest

    if not resume:
        if request is None:
            raise ValueError("--config is required for a new tifxyz export")
        return validate_request(request)
    out = Path(out)
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("symlink in export path")
    saved = read_metadata(out / "EXPORT.json")[0]
    try:
        settings = Settings(**saved["effective_settings"])
    except (KeyError, TypeError) as error:
        raise ValueError("invalid saved tifxyz settings") from error
    if request is not None and digest(validate_request(request)) != settings.request_sha256:
        raise ValueError("tifxyz request changed; use a new output")
    return settings


def validate_observations(config, volume, shape):
    if (
        not config.get("lasagna_input")
        and (config.get("fibres") or config.get("prediction", "classical") != "classical")
        and volume.dtype not in (np.dtype("uint8"), np.dtype("uint16"))
    ):
        raise ValueError("Lasagna requires uint8 or uint16 CT")
    if config.get("prediction") == "qual-v2" and not config.get("fibres"):
        spacing = np.asarray(volume.spacing_xyz) * config.get("stride", 1)
        if (
            volume.dtype != np.uint8
            or not np.allclose(spacing, spacing[0], rtol=1e-6, atol=0)
            or spacing.min() < 8.64 - 1e-6
            or spacing.max() > 9.6 + 1e-6
        ):
            raise ValueError(
                "Qual V2 requires uint8 CT and isotropic 8.64–9.6 µm observations; "
                "select --level/--stride to match, independently of --field-spacing-um"
            )
        if min(shape) < 8:
            raise ValueError("Qual V2 splines need at least 8 observation samples per axis")


def field_geometry(meta, spacing_um=None):
    geometry = {k: meta[k] for k in ("shape_zyx", "origin_xyz", "spacing_xyz", "frame")}
    if spacing_um is not None:
        if not math.isfinite(spacing_um) or spacing_um <= 0:
            raise ValueError("field spacing must be positive finite micrometres")
        extent = (np.asarray(meta["shape_zyx"])[::-1] - 1) * meta["spacing_xyz"]
        shape = []
        for length in extent:
            ratio = float(length) / spacing_um
            if not math.isfinite(ratio) or ratio >= np.iinfo(np.int64).max - 2:
                raise ValueError("field grid is too large; increase --field-spacing-um")
            last = math.floor(ratio)
            # Division can round below an integer; coordinates decide ownership.
            last += int((last + 1) * spacing_um <= length)
            last -= int(last * spacing_um > length)
            shape.append(last + 1)
        geometry.update(spacing_xyz=[float(spacing_um)] * 3, shape_zyx=shape[::-1])
    if min(geometry["shape_zyx"]) < 2:
        raise ValueError(
            "field grid needs at least two nodes per axis; enlarge ROI or reduce --field-spacing-um"
        )
    return geometry


LASAGNA_DOWNSAMPLE = {"cos": 2, "grad_mag": 4, "nx": 4, "ny": 4}


def lasagna_downsamples(config):
    factors = {
        name: config.get("lasagna_" + name + "_downsample", default)
        for name, default in LASAGNA_DOWNSAMPLE.items()
    }
    if any(type(v) is not int or v not in (1, 2, 4, 8) for v in factors.values()):
        raise ValueError("Lasagna downsample factors must be1,2,4 or8")
    if factors["nx"] != factors["ny"]:
        raise ValueError("Lasagna nx and ny must share the same downsample factor")
    return factors


def validate_field_cpu(config, *, convergence=False):
    workers = config.get("field_cpu_workers", 0)
    budget = config.get("field_cpu_buffer_bytes", 0)
    if (
        type(workers) is not int
        or workers < 0
        or type(budget) is not int
        or budget < 0
        or bool(workers) != bool(budget)
    ):
        raise ValueError(
            "CPU workers require an explicit positive CPU buffer budget; both default to zero"
        )
    if workers and (convergence or config.get("bounded_preparation")):
        raise ValueError("CPU preparation pool supports only ordinary static providers")


# Two-level Schwarz as measured on the 175A 384 box (RTX 5090, 2026-11-26): blocks of 128
# with halo 8 reach the one-block solution in about 5 sweeps, and Jacobi local solves with
# the fused kernels make a steady sweep 2.7x faster than AMG with the eager loop.
SCHWARZ_FIELD_DEFAULTS = dict(field_chunk=128, field_halo=8, field_kernel="fused")


def field_defaults(config):
    """Fill the field solver settings a new run's command line left unset.

    Resolved once, when the run is created, so the saved configuration names them and a
    resume never depends on the defaults of the version that resumes it.
    """
    if config.get("field_coarse_factor", 0):
        for key, value in SCHWARZ_FIELD_DEFAULTS.items():
            config.setdefault(key, value)
    config.setdefault("field_chunk", 64)
    return config


def validate_config(config):
    validate_field_cpu(config)
    if type(config.get("field_advice", False)) is not bool:
        raise ValueError("field_advice must be boolean")
    if config.get("field_coarse_factor", 0) == 1:
        raise ValueError("field_coarse_factor must be 0 (off) or at least 2")
    if config.get("field_pair_scope", "block") not in ("block", "global"):
        raise ValueError("field_pair_scope must be block or global")
    if config.get("field_memory", "auto") not in ("auto", "disk", "field", "all"):
        raise ValueError("field_memory must be auto, disk, field or all")
    if config.get("preconditioner", "jacobi") not in ("jacobi", "amg"):
        raise ValueError("block workflows require preconditioner='jacobi' or 'amg'")
    if config.get("field_kernel", "eager") not in ("eager", "fused"):
        raise ValueError("field_kernel must be eager or fused")
    if config.get("field_status_policy", "all") not in ("all", "checkpoints"):
        raise ValueError("field_status_policy must be all or checkpoints")
    prior = config.get("field_prior")
    if prior is not None and (not isinstance(prior, str) or not prior):
        raise ValueError("field_prior must be a completed field run directory")
    if config.get("field_prior_calibration", "own") not in ("own", "inherit"):
        raise ValueError("field_prior_calibration must be own or inherit")
    if config.get("field_prior_calibration") == "inherit" and not prior:
        raise ValueError("field_prior_calibration inherit requires field_prior")
    if config.get("field_scope", "all") not in ("all", "supported"):
        raise ValueError("field_scope must be all or supported")
    if config.get("field_scope") == "supported" and not prior:
        raise ValueError("field_scope supported requires field_prior")
    if config.get("field_scope") == "supported" and config.get("field_coarse_factor", 0):
        # The coarse correction needs every chunk; refuse before any stage runs.
        raise ValueError("field_scope supported cannot be combined with field_coarse_factor")
    if config.get("afv_clip_backend", "auto") not in ("auto", "native", "numpy", "reference"):
        raise ValueError("afv_clip_backend must be auto, native, numpy or reference")
    if config.get("nonconvergence_policy", "fail") not in ("fail", "continue"):
        raise ValueError("invalid nonconvergence_policy")
    lasagna_downsamples(config)
    if config.get("lasagna_input") and any(
        "lasagna_" + n + "_downsample" in config for n in LASAGNA_DOWNSAMPLE
    ):
        raise ValueError("--lasagna-input conflicts with Lasagna generation downsample flags")
    factor = config.get("fibre_downsample", 2)
    if type(factor) is not int or factor not in (2, 4):
        raise ValueError("fibre_downsample must be2 or4")
    if factor != 2 and not config.get("bounded_preparation"):
        raise ValueError("4x fibre downsampling requires --bounded-preparation")
    if type(config.get("bounded_preparation", False)) is not bool:
        raise ValueError("bounded_preparation must be boolean")
    if config.get("bounded_preparation"):
        if config.get("lasagna_geometry"):
            raise ValueError(
                "--bounded-preparation requires base-grid imports without --lasagna-geometry"
            )
        if (not config.get("fibres") and config.get("prediction") != "qual-v2") or config.get(
            "stride", 1
        ) != 1:
            raise ValueError("bounded preparation requires Qual V2 or imported AFV with stride1")
        if config.get("fibres") and not config.get("lasagna_input"):
            raise ValueError("bounded imported AFV requires imported Lasagna")
        if config.get("field_spacing_um") is None:
            raise ValueError("bounded preparation requires --field-spacing-um")
        if not config.get("fibres") and any(
            v % (2 * factor) for v in config.get("roi_lo", [0, 0, 0])
        ):
            raise ValueError(
                f"bounded ROI origin must be a multiple of{2 * factor} to preserve global seed parity"
            )
        budget = config.get("bounded_bytes", 12 << 30)
        if type(budget) is not int or budget < 16 << 20:
            raise ValueError("bounded allowance must be at least16MiB")
    if config.get("lasagna_geometry") and not config.get("lasagna_input"):
        raise ValueError("--lasagna-geometry requires --lasagna-input")
    if config.get("lasagna_input") and (
        config.get("lasagna_model_path") or config.get("lasagna_python")
    ):
        raise ValueError("--lasagna-input conflicts with --lasagna-model-path and --lasagna-python")
    if config.get("fibres_geometry") and not config.get("fibres"):
        raise ValueError("--fibres-geometry requires --fibres")
    if config.get("fibres") and config.get("model_path"):
        raise ValueError("external AFV --fibres conflicts with a fibre --model-path")
    stride = config.get("stride", 1)
    if type(stride) is not int or not (
        1 <= stride <= 8
        or (stride in (16, 32) and config.get("fibres") and config.get("lasagna_input"))
    ):
        raise ValueError("stride must be 1..8; 16 or 32 require both --fibres and --lasagna-input")
    integers = {
        "import_workers": (0, None),
        "chunk": (4, 128),
        # A block edge at least the largest grid dimension solves the whole field as one
        # block (a global solve); its memory, not this check, bounds the size.
        "field_chunk": (4, None),
        "field_halo": (1, None),
        "field_coarse_factor": (0, None),
        "level": (0, None),
        "sweeps": (1, None),
        "maxiter": (1, None),
        "seed_stride": (1, None),
        "mesh_tile": (1, 256),
        "chart_faces": (1, 8192),
        "level_workers": (1, 64),
        "threads": (1, None),
        "cache_bytes": (0, None),
        "reserve_bytes": (0, None),
        "bounded_bytes": (16 << 20, None),
        "prepared_cache_bytes": (0, None),
        "constraint_cache_bytes": (0, None),
        "observation_cache_bytes": (0, None),
        "field_cache_bytes": (0, None),
        "afv_support_snapshot_bytes": (0, None),
    }
    for name, (low, high) in integers.items():
        if name in config:
            value = config[name]
            if type(value) is not int or value < low or (high is not None and value > high):
                raise ValueError(f"{name} must be an integer in {low}..{high or 'unbounded'}")
    for name in (
        "voxel_um",
        "field_spacing_um",
        "rtol",
        "fibre_weight",
        "threshold",
        "invalid_value",
    ):
        value = config.get(name)
        if value is not None and (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (name in ("voxel_um", "field_spacing_um", "rtol") and value <= 0)
            or (name == "fibre_weight" and value < 0)
        ):
            raise ValueError("invalid " + name)
    if config.get("levels") is not None:
        if not config["levels"] or not all(math.isfinite(v) for v in config["levels"]):
            raise ValueError("levels must be nonempty finite numbers")
        if len(set(config["levels"])) != len(config["levels"]):
            raise ValueError("duplicate levels")
    for name, choices in [
        ("prediction", ("classical", "sd2", "qual-v2")),
        ("ct_mode", ("auto", "download", "stream")),
        ("surface_output", ("atlas", "tifxyz", "native")),
        ("flattening_method", RUN_METHODS),
    ]:
        if name in config and config[name] not in choices:
            raise ValueError("invalid " + name)
    if config.get("flattening_method") == "original" and config.get("surface_output") == "tifxyz":
        raise ValueError(
            "original flattening uses --export-tifxyz, not the direct-field surface-output branch"
        )
    if (
        config.get("flattening_method") == "slim"
        and config.get("surface_output", "atlas") != "native"
    ):
        raise ValueError("SLIM flattening requires --surface-output native")
    if config.get("surface_output", "atlas") == "native":
        coverage = config.get("coverage_target", 1.0)
        seconds = config.get("surface_time_limit_s", 1800.0)
        if (
            type(coverage) not in (int, float)
            or not math.isfinite(coverage)
            or not 0 < coverage <= 1
        ):
            raise ValueError("coverage_target must be finite in(0,1]")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("surface time limit must be positive finite seconds")
        if type(config.get("export_tifxyz", False)) is not bool:
            raise ValueError("export_tifxyz must be boolean")
    if "ct_texture" in config:
        if type(config["ct_texture"]) is not bool:
            raise ValueError("ct_texture must be boolean")
        if not config["ct_texture"]:
            if config.get("surface_output", "atlas") != "native":
                raise ValueError("--no-ct-texture requires --surface-output native")
            if config.get("export_tifxyz") is not True:
                raise ValueError("--no-ct-texture requires --export-tifxyz")
    if config.get("surface_output", "atlas") == "tifxyz":
        from .tifxyz_contract import validate_request
        from .util import digest

        request = validate_request(config.get("tifxyz_request"))
        if config.get("tifxyz_request_sha256") != digest(request):
            raise ValueError("tifxyz request digest differs")
        if config.get("levels") is not None and config["levels"] != request["levels"]:
            raise ValueError("--levels must exactly agree with tifxyz JSON levels")
    elif config.get("tifxyz_request") is not None:
        raise ValueError("--tifxyz-config requires --surface-output tifxyz")
