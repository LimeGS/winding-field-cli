"""Prepared-data construction/resume harness for the original field library.

The recipe declares geometry, calibration and SHA-bound NPY/NPZ inputs. Solving,
shared-field updates, storage and recovery live in block_solver/field_store.
"""

import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit
from zipfile import BadZipFile

import numpy as np

from dataset_config import frame, positive, xyz


def _sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _prepared_artifact(entry: dict[str, str], load: Callable[[BinaryIO], Any]) -> Any:
    path = Path(entry["path"])
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
            raise ValueError("prepared source hash differs: " + str(path))
        stream.seek(0)
        value = load(stream)
        if not os.path.samestat(os.fstat(stream.fileno()), path.stat()):
            raise ValueError("prepared source changed during loading: " + str(path))
        return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate configuration member: " + key)
        result[key] = value
    return result


def _invalid(value):
    raise ValueError("nonfinite configuration value: " + value)


def _keys(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError("unknown/invalid " + name + " fields")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def _artifact(value, base):
    _keys(value, {"path", "sha256"}, "input")
    path, digest = value["path"], value["sha256"]
    if (
        not isinstance(path, str)
        or not path
        or urlsplit(path).scheme
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("prepared inputs require local paths and lowercase SHA256")
    return dict(path=str((base / path).resolve()), sha256=digest)


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Validate a prepared run; paths belong to its JSON, not the caller CWD."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    config = json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid)
    if not isinstance(config, dict):
        raise ValueError("run configuration must be an object")
    sources = {}
    if config.get("schema") == "winding-field-cli-request-v1":
        _keys(config, {"schema", "output", "request", "sources"}, "saved request")
        sources = config["sources"]
        config = config["request"]
    config = validate_run_config(config, base=path.parent)
    config.update(config_path=str(path), config_sha256=hashlib.sha256(raw).hexdigest())
    if sources:
        config["request_sources"] = sources
    return config


def validate_run_config(mapping: dict, *, base: Path) -> dict[str, Any]:
    """Normalize an independent request through the existing prepared validator."""
    try:
        config = deepcopy(mapping)
        _keys(
            config,
            {
                "schema",
                "frame",
                "voxel_um",
                "grid",
                "sheet_step",
                "calibration",
                "observations",
                "fixed",
                "constraints",
                "order",
                "solver",
                "execution",
                "gpu",
                "source_identity",
            },
            "run",
        )
        if config.get("schema") != "winding-field-run-v1":
            raise ValueError("unsupported prepared-run schema")
        frame(config["frame"])
        positive(config["voxel_um"], "voxel_um")
        positive(config["sheet_step"], "sheet_step")
        if not isinstance(config["calibration"], str) or not config["calibration"].strip():
            raise ValueError("explicit calibration provenance required")
        grid = config["grid"]
        _keys(grid, {"origin_xyz", "spacing_xyz", "shape_zyx"}, "grid")
        xyz(grid["origin_xyz"], "grid origin")
        xyz(grid["spacing_xyz"], "grid spacing", True)
        shape = grid["shape_zyx"]
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError("three integer ZYX dimensions required")
        for n in shape:
            _integer(n, "grid shape", 2)
        if math.prod(shape) > np.iinfo(np.intp).max:
            raise ValueError("grid exceeds indexing capacity")
        observations = config["observations"]
        _keys(observations, {"target", "weights", "observed"}, "observations")
        if set(observations) != {"target", "weights", "observed"}:
            raise ValueError("target/weights/observed inputs required")
        config["observations"] = {k: _artifact(v, base) for k, v in observations.items()}
        for name in ("fixed", "constraints", "order"):
            if name in config:
                config[name] = _artifact(config[name], base)
        solver = {
            "device": "cpu",
            "dtype": "float64",
            "rtol": 1e-5,
            "maxiter": 4000,
            **config.get("solver", {}),
        }
        _keys(
            solver,
            {
                "device",
                "dtype",
                "rtol",
                "maxiter",
                "preconditioner",
                "kernel",
                "nonconvergence_policy",
            },
            "solver",
        )
        preconditioner = solver.pop("preconditioner", "jacobi")
        if preconditioner not in ("jacobi", "amg"):
            raise ValueError("prepared block workflows require preconditioner='jacobi' or 'amg'")
        if preconditioner == "amg":
            solver["preconditioner"] = "amg"
        kernel = solver.pop("kernel", "eager")
        if kernel not in ("eager", "fused"):
            raise ValueError("prepared block workflows require kernel='eager' or 'fused'")
        if kernel == "fused":
            solver["kernel"] = "fused"
        if solver.get("nonconvergence_policy", "fail") not in ("fail", "continue"):
            raise ValueError("invalid nonconvergence_policy")
        if solver["dtype"] not in ("float32", "float64") or solver["device"] not in (
            "cpu",
            "cuda:0",
        ):
            raise ValueError("solver requires float32/float64 and cpu or allocated cuda:0")
        positive(solver["rtol"], "rtol")
        _integer(solver["maxiter"], "maxiter")
        execution = dict(config["execution"])
        _keys(
            execution,
            {
                "chunk",
                "halo",
                "sweeps",
                "cache_bytes",
                "read_workers",
                "constraint_cache_bytes",
                "prepared_cache_bytes",
                "max_bytes",
                "reserve_bytes",
                "field_advice",
                "status_policy",
            },
            "execution",
        )
        for name, minimum in (("chunk", 1), ("halo", 0), ("sweeps", 1)):
            _integer(execution[name], name, minimum)
        for name, default, minimum in (
            ("cache_bytes", 0, 0),
            ("read_workers", 1, 1),
            ("constraint_cache_bytes", 0, 0),
            ("prepared_cache_bytes", 0, 0),
            ("reserve_bytes", 0, 0),
        ):
            execution[name] = _integer(execution.get(name, default), name, minimum)
        if execution.get("max_bytes") is not None:
            _integer(execution["max_bytes"], "max_bytes")
        if type(execution.get("field_advice", False)) is not bool:
            raise ValueError("field_advice must be boolean")
        if execution.get("status_policy", "all") not in ("all", "checkpoints"):
            raise ValueError("status_policy must be all or checkpoints")
        if not execution["halo"] and any(n > execution["chunk"] for n in shape):
            raise ValueError("multiple cores require positive halo")
        gpu = config.get("gpu")
        if solver["device"] == "cuda:0":
            if gpu is None:
                raise ValueError("CUDA construction requires an explicit GPU lease")
            _keys(gpu, {"index", "uuid", "lock_path", "min_free_bytes", "deadline"}, "GPU")
            gpu = dict(gpu)
            gpu["lock_path"] = str((base / gpu["lock_path"]).resolve())
            from field_runtime import GpuLease

            GpuLease(**gpu)
            config["gpu"] = gpu
        elif gpu is not None:
            raise ValueError("GPU lease supplied for a CPU run")
        extra = config.get("source_identity", {})
        if not isinstance(extra, dict):
            raise ValueError("source_identity must be an object")
        json.dumps(extra, allow_nan=False)
        config.update(solver=solver, execution=execution)
        return config
    except (KeyError, TypeError, OverflowError, ValueError) as error:
        raise ValueError("invalid run configuration: " + str(error)) from error


def _code_sources():
    names = (
        "block_solver",
        "field_store",
        "field_solver",
        "field_constraints",
        "constraint_templates",
        "field_sources",
        "field_prefetch",
        "field_runtime",
        "dataset_config",
        "physical_grid",
        "patch_check",
    )
    paths = [
        Path(__file__).resolve(),
        *(Path(importlib.import_module(name).__file__).resolve() for name in names),
    ]
    return {str(p): _sha(p) for p in paths}


def _prepare_run(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    if "gpu" in config and os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu"]["uuid"]:
        raise ValueError("CUDA_VISIBLE_DEVICES must match the explicitly configured GPU UUID")
    import scipy
    import torch

    from block_solver import _block_order
    from constraint_templates import StencilConstraints
    from field_sources import NpyFixedValues, NpyObservations

    grid, execution, solver = config["grid"], config["execution"], config["solver"]
    shape = tuple(grid["shape_zyx"])
    inputs = config["observations"]
    loader = NpyObservations(
        inputs["target"]["path"],
        inputs["weights"]["path"],
        inputs["observed"]["path"],
        {k: v["sha256"] for k, v in inputs.items()},
    )
    if loader.shape != shape:
        raise ValueError("prepared observations/grid shape differs")
    chunk = execution["chunk"]
    grid_shape = tuple((n + chunk - 1) // chunk for n in shape)
    if "order" in config:
        order = _prepared_artifact(
            config["order"], lambda stream: np.load(stream, allow_pickle=False)
        )
        if not isinstance(order, np.ndarray) or order.dtype.kind not in "iu":
            if isinstance(order, np.lib.npyio.NpzFile):
                order.close()
            raise ValueError("integer order NPY required")
    else:
        order = list(itertools.product(*(range(n) for n in grid_shape)))
    order = _block_order(order, shape, chunk)
    rows = None
    if "constraints" in config:
        rows = _prepared_artifact(
            config["constraints"],
            lambda stream: StencilConstraints.load(
                stream, cache_bytes=execution["constraint_cache_bytes"]
            ),
        )
        if rows.shape != shape:
            raise ValueError("constraint/grid shape differs")
    fixed = None
    if "fixed" in config:
        values = NpyFixedValues(config["fixed"]["path"], config["fixed"]["sha256"], shape)
        # An all-NaN fixed input is an unanchored problem, with the global zero gauge.
        for key in itertools.product(*(range(n) for n in grid_shape)):
            low = np.asarray(key) * chunk
            high = np.minimum(low + chunk, shape)
            if np.isfinite(values(low, high)).any():
                fixed = values
                break
    sources = _code_sources()
    problem: dict[str, Any] = {
        name: deepcopy(config[name])
        for name in (
            "frame",
            "voxel_um",
            "grid",
            "sheet_step",
            "calibration",
            "observations",
            "fixed",
            "constraints",
            "order",
            "source_identity",
            "solver",
        )
        if name in config
    }
    problem["execution"] = {k: execution[k] for k in ("chunk", "halo", "sweeps")}
    identity = dict(
        problem,
        library_sha256={Path(k).name: v for k, v in sources.items()},
        numeric_versions=dict(
            numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__)
        ),
    )
    metadata = dict(
        schema="winding-field-construction-v1",
        identity=identity,
        **{k: config[k] for k in ("frame", "voxel_um", "grid", "sheet_step", "calibration")},
        config_path=config["config_path"],
        config_sha256=config["config_sha256"],
        effective_threads=torch.get_num_threads(),
    )
    if config.get("request_sources"):
        metadata["request_sources"] = config["request_sources"]
    options = dict(
        shape=shape,
        chunk=chunk,
        identity=identity,
        load_observations=loader,
        origin_xyz=grid["origin_xyz"],
        spacing_xyz=grid["spacing_xyz"],
        order=order,
        constraints=rows,
        fixed=fixed,
        halo=execution["halo"],
        sweeps=execution["sweeps"],
        **solver,
        cache_bytes=execution["cache_bytes"],
        prepared_cache_bytes=execution["prepared_cache_bytes"],
        read_workers=execution["read_workers"],
        max_bytes=execution.get("max_bytes"),
        reserve_bytes=execution["reserve_bytes"],
    )
    if "status_policy" in execution:
        # Execution choice, not checkpoint identity: STATUS fsyncs only at checkpoints.
        options["status_policy"] = execution["status_policy"]
    return options, metadata, sources


def run_field(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    prefetch: bool = False,
    stop: Callable[[], bool] | None = None,
    admit: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Construct or resume through the canonical block engine.

    Prepared observations are NPY, constraints a StencilConstraints NPZ. Stencil
    inputs are resident; cache limits bound templates, not total input memory.
    Raw CT/AFV preparation stays in the caller's recipe. CUDA requires the declared
    UUID in CUDA_VISIBLE_DEVICES; this entry point never selects a host GPU itself.
    """
    config = load_run_config(config_path)
    output = Path(output).resolve()
    if type(resume) is not bool or type(prefetch) is not bool:
        raise ValueError("resume/prefetch must be boolean")
    options, metadata, sources = _prepare_run(config)
    from .field_advice import observer_options

    geometry = dict(
        config["grid"],
        frame=config["frame"],
        voxel_um=config["voxel_um"],
        sheet_step=config["sheet_step"],
    )
    options.update(
        observer_options(
            config["execution"].get("field_advice", False),
            output,
            geometry,
            reserve_bytes=options["reserve_bytes"],
        )
    )
    from block_solver import solve_blocks
    from field_prefetch import ObservationPrefetch
    from field_runtime import GpuLease
    from field_store import atomic_json

    first = [True]
    lease = [None]
    with ExitStack() as stack:

        def admission():
            if _sha(config["config_path"]) != config["config_sha256"] or any(
                _sha(path) != sha for path, sha in sources.items()
            ):
                raise ValueError("construction configuration/library source changed")
            if "gpu" in config:
                if lease[0] is None:
                    lease[0] = stack.enter_context(GpuLease(**config["gpu"]))
                else:
                    lease[0].recheck()
            if first[0]:
                path = output / "RUN.json"
                if path.exists():
                    previous = json.loads(path.read_bytes())
                    if previous.get("identity") != metadata["identity"]:
                        raise ValueError("construction run identity differs")
                else:
                    atomic_json(path, metadata)
                first[0] = False
            if admit is not None:
                admit()

        client = None
        if prefetch:
            window = tuple(min(n, options["chunk"] + 2 * options["halo"]) for n in options["shape"])
            client = ObservationPrefetch(
                options["load_observations"],
                options["shape"],
                max_bytes=math.prod(window) * 17 + 1024**2,
            )
        result = solve_blocks(
            output, **options, resume=resume, stop=stop, admit=admission, prefetch=client
        )
        from .field_advice import finish

        finish(options.get("observer"))
        result["effective_threads"] = metadata["effective_threads"]
        return result


def verify_field(config_path, output):
    """Read-only global numerical verification bound to the prepared problem."""
    from .global_residual import verify
    from .surface_sources import FieldSource

    config = load_run_config(config_path) if not isinstance(config_path, dict) else config_path
    options, metadata, _ = _prepare_run(config)
    checkpoint = Path(output).resolve() / "CHECKPOINT.json"
    checkpoint_sha = _sha(checkpoint)
    saved = json.loads(checkpoint.read_bytes())
    if saved["identity"].get("source_identity") != metadata["identity"]:
        raise ValueError("verification problem differs from checkpoint")
    geometry = dict(
        config["grid"],
        frame=config["frame"],
        voxel_um=config["voxel_um"],
        sheet_step=config["sheet_step"],
    )
    descriptor = dict(
        kind="chunks", path=str(checkpoint), sha256=checkpoint_sha, support_policy="stored"
    )
    enabled = config["execution"].get("field_advice", False)
    regions = [] if enabled else None
    started = time.perf_counter() if enabled else None
    with FieldSource(descriptor, geometry, cache_bytes=options["cache_bytes"]) as source:

        def read(lo, hi):
            value, stored = source.read(lo, hi)
            if not stored.all():
                raise ValueError("global verification requires every stored field node")
            return value

        rows = options["constraints"]
        gauge_offset = None
        if options["fixed"] is None:
            gauge_offset = source._load(
                "0-0-0", tuple(min(options["chunk"], n) for n in options["shape"])
            )[1]
        result = verify(
            options["shape"],
            options["chunk"],
            options["spacing_xyz"],
            read,
            options["load_observations"],
            fixed=options["fixed"],
            constraints=None if rows is None else rows.restrict,
            rtol=options["rtol"],
            gauge_offset=gauge_offset,
            **(dict(regions=regions) if enabled else {}),
        )
        source.validate(hashes=True)
    if _sha(checkpoint) != checkpoint_sha:
        raise ValueError("checkpoint changed during verification")
    result.update(checkpoint_sha256=checkpoint_sha, config_sha256=config["config_sha256"])
    if enabled:
        from .field_advice import Advice, record_verification

        observer = Advice(checkpoint.parent, geometry, reserve_bytes=options["reserve_bytes"])
        scientific = {k: v for k, v in saved["identity"].items() if k != "sweeps"}
        try:
            observer.restore(
                dict(
                    manifest=saved,
                    checkpoint_sha256=checkpoint_sha,
                    scientific_sha256=hashlib.sha256(
                        json.dumps(scientific, sort_keys=True, allow_nan=False).encode()
                    ).hexdigest(),
                )
            )
            record_verification(
                observer,
                checkpoint_sha,
                result,
                started,
                sweep=len(saved["progress"]["completed_sweeps"]),
                regions=regions,
            )
        except Exception as error:
            warnings.warn(
                "field advice verification unavailable: " + str(error), RuntimeWarning, stacklevel=2
            )
    return result


def _supervised(config_path: Path, output: Path, *, resume: bool, prefetch: bool) -> int:
    from field_runtime import _read_json, run_supervised

    output = output.resolve()
    if not resume and output.exists():
        raise ValueError("build requires a new output directory")
    config = load_run_config(config_path)
    options, metadata, sources = _prepare_run(config)
    del options  # Resident stencils/memmaps must not outlive preflight alongside the worker.
    if resume:
        checkpoint = _read_json(output / "CHECKPOINT.json", limit=64 * 1024**2)
        if (
            checkpoint is None
            or not isinstance(checkpoint.get("identity"), dict)
            or checkpoint["identity"].get("source_identity") != metadata["identity"]
        ):
            raise ValueError("construction checkpoint identity differs or is missing")
    sources[config["config_path"]] = config["config_sha256"]
    common = ["--config", config["config_path"], "--output", str(output)]
    if prefetch:
        common.append("--prefetch")
    command = [sys.executable, "-m", "winding_field.field_workflow"]
    return run_supervised(
        [*command, "build", *common],
        [*command, "resume", *common],
        output,
        output.with_name(output.name + ".service"),
        source_sha256=sources,
        resume=resume,
    )


_SOLVER_FLAGS = {
    "device": str,
    "dtype": str,
    "rtol": float,
    "maxiter": int,
    "preconditioner": str,
    "kernel": str,
}
_EXECUTION_FLAGS = {
    name: int
    for name in (
        "chunk",
        "halo",
        "sweeps",
        "cache_bytes",
        "read_workers",
        "constraint_cache_bytes",
        "prepared_cache_bytes",
        "max_bytes",
        "reserve_bytes",
    )
}
_GPU_FLAGS = {
    "index": int,
    "uuid": str,
    "lock_path": Path,
    "min_free_bytes": int,
    "deadline": float,
}


def _auto_bytes(value):
    if value == "auto":
        return None
    if isinstance(value, bool) or isinstance(value, float) and not value.is_integer():
        raise ValueError("max_bytes must be an integer or auto")
    return int(value)


def _request_flags(parser):
    group = parser.add_argument_group(
        "prepared request overrides (flags > recipe > defaults)", argument_default=argparse.SUPPRESS
    )
    group.add_argument("--field-frame")
    group.add_argument(
        "--field-voxel-um", type=float, help="field frame voxel calibration in micrometres"
    )
    for name, value_type in (("origin_xyz", float), ("spacing_xyz", float), ("shape_zyx", int)):
        group.add_argument("--grid-" + name.replace("_", "-"), type=value_type, nargs=3)
    group.add_argument("--sheet-step", type=float, help="raw field units per sheet")
    group.add_argument("--calibration", help="explicit calibration provenance")
    group.add_argument(
        "--field-advice",
        action="store_true",
        help="consultative diagnostics only; no implicit verification",
    )
    for name in ("target", "weights", "observed", "fixed", "constraints", "order"):
        group.add_argument("--" + name, type=Path)
        group.add_argument("--" + name + "-sha256")
        if name in ("fixed", "constraints", "order"):
            group.add_argument(
                "--no-" + name,
                dest=name,
                action="store_const",
                const=None,
                help="remove the optional artifact from the effective request",
            )
    for fields, prefix in ((_SOLVER_FLAGS, ""), (_EXECUTION_FLAGS, ""), (_GPU_FLAGS, "gpu-")):
        for name, value_type in fields.items():
            group.add_argument(
                "--" + prefix + name.replace("_", "-"),
                type=_auto_bytes if name == "max_bytes" else value_type,
            )
    group.add_argument("--source-identity", nargs=2, action="append", metavar=("KEY", "JSON_VALUE"))


def _effective_request(args, sources):
    from .cli_settings import merge_explicit, persist_request, request_snapshot, read_settings
    from .tifxyz_contract import parse_json
    from field_runtime import check_resources

    output = args.output.resolve()
    retained = None
    if args.action != "build":
        retained = read_settings(output / "RUN.json") if (output / "RUN.json").exists() else {}
        if retained.get("config_path") and Path(retained["config_path"]).name.startswith(
            output.name + ".request-"
        ):
            if _sha(retained["config_path"]) != retained["config_sha256"]:
                raise ValueError("saved request changed")
    elif output.exists() and (
        args.supervised or any(p.name != ".supervisor.lock" for p in output.iterdir())
    ):
        raise ValueError("build requires a new output directory")
    path = args.config
    if path is None and retained is not None:
        if not retained.get("config_path"):
            raise ValueError(
                "resume/verify without --config requires retained RUN request metadata"
            )
        path = Path(retained["config_path"])
        if _sha(path) != retained["config_sha256"]:
            raise ValueError("saved request changed")
    base = {}
    provenance = dict(sources)
    if path is not None:
        document = read_settings(path, sources=provenance)
        if document.get("schema") == "winding-field-cli-request-v1":
            loaded = load_run_config(path)
            if document["output"] != str(output):
                raise ValueError("saved request belongs to another output")
            provenance.pop(str(Path(path).resolve()))
            provenance.update(loaded.get("request_sources", {}))
            base = {
                k: v
                for k, v in loaded.items()
                if k not in ("config_path", "config_sha256", "request_sources")
            }
        else:
            base = deepcopy(document)
            entries = (
                list(base.get("observations", {}).values())
                if isinstance(base.get("observations"), dict)
                else []
            )
            entries.extend(base.get(name) for name in ("fixed", "constraints", "order"))
            entries.append(base.get("gpu"))
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for key in ("path", "lock_path"):
                    value = entry.get(key)
                    if isinstance(value, str) and value and not urlsplit(value).scheme:
                        entry[key] = str((Path(path).resolve().parent / value).resolve())
    values = vars(args)
    overrides = {}
    for flag, name in (
        ("field_frame", "frame"),
        ("field_voxel_um", "voxel_um"),
        ("sheet_step", "sheet_step"),
        ("calibration", "calibration"),
    ):
        if flag in values:
            overrides[name] = values[flag]
    for prefix, name in (("grid_", "grid"), ("gpu_", "gpu"), ("", "solver"), ("", "execution")):
        fields = (
            ("origin_xyz", "spacing_xyz", "shape_zyx")
            if name == "grid"
            else _GPU_FLAGS
            if name == "gpu"
            else _SOLVER_FLAGS
            if name == "solver"
            else _EXECUTION_FLAGS
        )
        for field in fields:
            if prefix + field in values:
                value = values[prefix + field]
                if isinstance(value, Path):
                    value = str(value.resolve())
                overrides.setdefault(name, {})[field] = value
    if "field_advice" in values:
        overrides.setdefault("execution", {})["field_advice"] = values["field_advice"]
    for name in ("target", "weights", "observed", "fixed", "constraints", "order"):
        if name in values and values[name] is None:
            if name not in ("fixed", "constraints", "order") or name + "_sha256" in values:
                raise ValueError("missing/conflicting artifact: " + name)
            base.pop(name, None)
            continue
        entry = {}
        if name in values:
            entry["path"] = str(values[name].resolve())
        if name + "_sha256" in values:
            entry["sha256"] = values[name + "_sha256"]
        if entry:
            if name in ("target", "weights", "observed"):
                overrides.setdefault("observations", {})[name] = entry
            else:
                overrides[name] = entry
    if "source_identity" in values:
        extra = {}
        for key, value in args.source_identity:
            if key in extra:
                raise ValueError("duplicate source identity key: " + key)
            extra[key] = parse_json(value)
        overrides["source_identity"] = extra
    config = validate_run_config(
        merge_explicit({"schema": "winding-field-run-v1", **base}, overrides), base=Path.cwd()
    )
    saved, raw = request_snapshot(output, config, sources=provenance)
    effective = dict(config, config_path=str(saved), config_sha256=hashlib.sha256(raw).hexdigest())
    if provenance:
        effective["request_sources"] = provenance
    _, metadata, _ = _prepare_run(effective)
    if retained is not None:
        checkpoint = read_settings(output / "CHECKPOINT.json")
        if (retained and retained.get("identity") != metadata["identity"]) or checkpoint.get(
            "identity", {}
        ).get("source_identity") != metadata["identity"]:
            raise ValueError("construction request differs from retained problem")
    if args.action == "verify":
        return effective
    # Only live user-supplied files are admission inputs; retained sources are historical provenance.
    for source, pin in sources.items():
        if _sha(source) != pin:
            raise ValueError("request source changed")
    if args.config is not None and document.get("schema") != "winding-field-cli-request-v1":
        if _sha(args.config) != provenance[str(args.config.resolve())]:
            raise ValueError("request source changed")
    check_resources(output.parent, reserve_bytes=config["execution"]["reserve_bytes"])
    persist_request(output, config, resume=saved.exists(), sources=provenance)
    return saved


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] in {"mesh", "render", "surfaces"}:
        from .surface_workflow import main as surface_main

        return surface_main(tokens)
    parser = argparse.ArgumentParser(prog="winding-field prepared", description=__doc__)
    parser.add_argument(
        "action", choices=("build", "resume", "verify", "mesh", "render", "surfaces")
    )
    parser.add_argument("--config", type=Path, help="optional native prepared-run recipe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefetch", action="store_true", help="one bounded CPU observation prefetch worker"
    )
    parser.add_argument(
        "--supervised",
        action="store_true",
        help="owned worker, source binding and resource-only retries",
    )
    _request_flags(parser)
    from .cli_settings import parse_options

    sources = {}
    args = parse_options(parser, tokens, sources=sources)
    from field_runtime import ResourceUnavailable

    try:
        config = _effective_request(args, sources)
        if args.action == "verify":
            if args.supervised or args.prefetch:
                raise ValueError("verify does not launch solver workers")
            result = verify_field(config, args.output)
            print(json.dumps(result, allow_nan=False))
            return 0 if result["global_converged"] else 3
        if args.supervised:
            return _supervised(
                config, args.output, resume=args.action == "resume", prefetch=args.prefetch
            )
        result = run_field(
            config, args.output, resume=args.action == "resume", prefetch=args.prefetch
        )
    except ResourceUnavailable as error:
        print("resources unavailable: " + str(error), file=sys.stderr)
        return 75
    except (ValueError, OSError, EOFError, BadZipFile) as error:
        print("configuration/input error: " + str(error), file=sys.stderr)
        return 2
    except RuntimeError as error:
        print("construction failed: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
