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
    try:
        config = json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid)
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
        config["observations"] = {k: _artifact(v, path.parent) for k, v in observations.items()}
        for name in ("fixed", "constraints", "order"):
            if name in config:
                config[name] = _artifact(config[name], path.parent)
        solver = {"device": "cpu", "dtype": "float64", "rtol": 1e-5, "maxiter": 4000, **config.get("solver", {})}
        _keys(solver, {"device", "dtype", "rtol", "maxiter"}, "solver")
        if solver["dtype"] not in ("float32", "float64") or solver["device"] not in ("cpu", "cuda:0"):
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
                "max_bytes",
                "reserve_bytes",
            },
            "execution",
        )
        for name, minimum in (("chunk", 1), ("halo", 0), ("sweeps", 1)):
            _integer(execution[name], name, minimum)
        for name, default, minimum in (
            ("cache_bytes", 0, 0),
            ("read_workers", 1, 1),
            ("constraint_cache_bytes", 0, 0),
            ("reserve_bytes", 0, 0),
        ):
            execution[name] = _integer(execution.get(name, default), name, minimum)
        if execution.get("max_bytes") is not None:
            _integer(execution["max_bytes"], "max_bytes")
        if not execution["halo"] and any(n > execution["chunk"] for n in shape):
            raise ValueError("multiple cores require positive halo")
        gpu = config.get("gpu")
        if solver["device"] == "cuda:0":
            if gpu is None:
                raise ValueError("CUDA construction requires an explicit GPU lease")
            _keys(gpu, {"index", "uuid", "lock_path", "min_free_bytes", "deadline"}, "GPU")
            gpu = dict(gpu)
            gpu["lock_path"] = str((path.parent / gpu["lock_path"]).resolve())
            from field_runtime import GpuLease

            GpuLease(**gpu)
            config["gpu"] = gpu
        elif gpu is not None:
            raise ValueError("GPU lease supplied for a CPU run")
        extra = config.get("source_identity", {})
        if not isinstance(extra, dict):
            raise ValueError("source_identity must be an object")
        json.dumps(extra, allow_nan=False)
        config.update(
            solver=solver, execution=execution, config_path=str(path), config_sha256=hashlib.sha256(raw).hexdigest()
        )
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
    paths = [Path(__file__).resolve(), *(Path(importlib.import_module(name).__file__).resolve() for name in names)]
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
        order = _prepared_artifact(config["order"], lambda stream: np.load(stream, allow_pickle=False))
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
            lambda stream: StencilConstraints.load(stream, cache_bytes=execution["constraint_cache_bytes"]),
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
        numeric_versions=dict(numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__)),
    )
    metadata = dict(
        schema="winding-field-construction-v1",
        identity=identity,
        **{k: config[k] for k in ("frame", "voxel_um", "grid", "sheet_step", "calibration")},
        config_path=config["config_path"],
        config_sha256=config["config_sha256"],
    )
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
        read_workers=execution["read_workers"],
        max_bytes=execution.get("max_bytes"),
        reserve_bytes=execution["reserve_bytes"],
    )
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
                options["load_observations"], options["shape"], max_bytes=math.prod(window) * 17 + 1024**2
            )
        return solve_blocks(output, **options, resume=resume, stop=stop, admit=admission, prefetch=client)


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


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] in {"mesh", "render", "surfaces"}:
        from .surface_workflow import main as surface_main

        return surface_main(tokens)
    parser = argparse.ArgumentParser(prog="winding-field prepared", description=__doc__)
    parser.add_argument("action", choices=("build", "resume", "mesh", "render", "surfaces"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefetch", action="store_true", help="one bounded CPU observation prefetch worker")
    parser.add_argument(
        "--supervised", action="store_true", help="owned worker, source binding and resource-only retries"
    )
    args = parser.parse_args(argv)
    from field_runtime import ResourceUnavailable

    try:
        if args.supervised:
            return _supervised(args.config, args.output, resume=args.action == "resume", prefetch=args.prefetch)
        result = run_field(args.config, args.output, resume=args.action == "resume", prefetch=args.prefetch)
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
