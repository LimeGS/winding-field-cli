"""Scroll-independent field/mesh -> durable indexed surfaces -> bounded CPU3D views."""

import argparse
import colorsys
import fcntl
import stat
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import uuid
from copy import deepcopy
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from field_runtime import ResourceUnavailable, check_resources
from field_store import atomic_json
from .field_workflow import _invalid, _object
from .surface_mesh import MeshReader, _integer, _levels, extract_mesh, import_mesh, sha
from .native_mesh import extract_mesh as extract_batched_mesh
from .surface_sources import FieldSource, chunk_cache_floor


@contextmanager
def _checkpoint_lease(path):
    """Hold a shared nonblocking lease for the whole extraction; never create the lock."""
    lock = Path(path).parent / ".writer.lock"
    fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("checkpoint writer lease must be a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("checkpoint has an active writer") from None

        def check():
            current = lock.lstat()
            if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise ValueError("checkpoint writer lease changed")

        check()
        yield check
    finally:
        os.close(fd)


def _keys(value, allowed):
    if not isinstance(value, dict) or value.keys() - set(allowed):
        raise ValueError("unknown/invalid surface configuration fields")


def _json(path, *, sources=None):
    with Path(path).open("rb") as stream:
        raw = stream.read(1024**2 + 1)
    if len(raw) > 1024**2:
        raise ValueError("surface config exceeds1MiB")
    result = json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid)
    if sources is not None:
        sources[str(Path(path).resolve())] = hashlib.sha256(raw).hexdigest()
    return result


def _artifact(value, base):
    _keys(value, {"path", "sha256"})
    if (
        not isinstance(value.get("path"), str)
        or not value["path"]
        or "://" in value["path"]
        or not isinstance(value.get("sha256"), str)
        or not re.fullmatch("[0-9a-f]{64}", value["sha256"])
    ):
        raise ValueError("local artifact path and lowercase SHA256 required")
    return dict(path=str((base / value["path"]).absolute()), sha256=value["sha256"])


def _render_settings(value=None):
    value = {} if value is None else value
    _keys(value, {"width", "height", "views", "triangle_batch", "levels", "reserve_bytes"})
    result = {
        "width": 1024,
        "height": 1024,
        "triangle_batch": 8192,
        "reserve_bytes": 0,
        "views": [
            dict(name="front", azimuth=0.0, elevation=0.0),
            dict(name="oblique", azimuth=35.0, elevation=20.0),
        ],
        **value,
    }
    for key in ("width", "height", "triangle_batch"):
        _integer(result[key], key)
    _integer(result["reserve_bytes"], "reserve_bytes", 0)
    if result["width"] * result["height"] > 16_000_000 or result["triangle_batch"] > 65536:
        raise ValueError("render exceeds bounded pixel/triangle budget")
    if not isinstance(result["views"], list) or not 1 <= len(result["views"]) <= 32:
        raise ValueError("one to32 views required")
    names = set()
    views = []
    for view in result["views"]:
        _keys(view, {"name", "azimuth", "elevation"})
        name = view.get("name")
        if (
            not isinstance(name, str)
            or not re.fullmatch("[a-zA-Z0-9_-]{1,64}", name)
            or name in names
            or name.upper() in {"PLAN", "RENDER"}
        ):
            raise ValueError("unique simple view names required")
        names.add(name)
        view = {"azimuth": 0.0, "elevation": 0.0, **view}
        for key in ("azimuth", "elevation"):
            if type(view[key]) not in (int, float) or not math.isfinite(view[key]):
                raise ValueError("finite view angles required")
        if not -90 <= view["elevation"] <= 90:
            raise ValueError("elevation outside bounds")
        views.append(view)
    result["views"] = views
    if "levels" in result:
        result["levels"] = _levels(result["levels"])
    return result


def load_surface_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = _json(path)
    sources = {}
    if isinstance(config, dict) and config.get("schema") == "winding-field-cli-request-v1":
        _keys(config, {"schema", "output", "request", "sources"})
        sources = config["sources"]
        config = config["request"]
    config = validate_surface_config(config, base=path.parent)
    config.update(config_path=str(path), config_sha256=sha(path))
    if sources:
        config["request_sources"] = sources
    return config


def validate_surface_config(mapping: dict, *, base: Path) -> dict[str, Any]:
    config = deepcopy(mapping)
    _keys(config, {"schema", "field", "mesh", "geometry", "levels", "execution", "render"})
    if config.get("schema") != "winding-surfaces-v1" or ("field" in config) == ("mesh" in config):
        raise ValueError("surface schema and exactly one field/mesh source required")
    geometry = config.get("geometry")
    _keys(
        geometry,
        {
            "frame",
            "voxel_um",
            "origin_xyz",
            "spacing_xyz",
            "shape_zyx",
            "sheet_step",
            "calibration",
        },
    )
    if (
        not isinstance(geometry.get("frame"), str)
        or not geometry["frame"].strip()
        or type(geometry.get("voxel_um")) not in (int, float)
        or not math.isfinite(geometry["voxel_um"])
        or geometry["voxel_um"] <= 0
    ):
        raise ValueError("explicit frame and positive voxel_um required")
    if "field" in config:
        field = config["field"]
        if isinstance(field, dict) and field.get("kind") == "zarr-v2":
            _keys(field, {"kind", "url", "sha256", "complete_sha256", "support_policy"})
            descriptor = dict(field)
        else:
            _keys(field, {"kind", "path", "sha256", "mask", "support_policy"})
            descriptor = _artifact({k: field[k] for k in ("path", "sha256") if k in field}, base)
            descriptor.update({k: v for k, v in field.items() if k not in descriptor})
            if "mask" in field:
                descriptor["mask"] = _artifact(field["mask"], base)
        config["field"] = descriptor
        config["levels"] = _levels(config.get("levels"))
        if len(config["levels"]) > 4096:
            raise ValueError("at most4096 levels per bundle")
    else:
        if "levels" in config:
            raise ValueError("imported meshes have group IDs, not declared isovalues")
        config["mesh"] = _artifact(config["mesh"], base)
    _keys(
        config.get("execution", {}),
        {"tile", "level_batch", "cache_bytes", "reserve_bytes", "mesh_bytes"},
    )
    execution = {
        "tile": 128,
        "level_batch": 4,
        "cache_bytes": 0,
        "reserve_bytes": 0,
        "mesh_bytes": 512 * 1024**2,
        **config.get("execution", {}),
    }
    _keys(execution, {"tile", "level_batch", "cache_bytes", "reserve_bytes", "mesh_bytes"})
    for key in execution:
        _integer(execution[key], key, 0 if key in ("cache_bytes", "reserve_bytes") else 1)
    if execution["tile"] > 256:
        raise ValueError("tile exceeds256 cells")
    config.update(
        execution=execution,
        render=_render_settings(config.get("render")),
    )
    return config


def _publish_png(path, image):
    from PIL import Image

    pending = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".pending")
    Image.fromarray(image).save(pending, format="PNG")
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    if path.exists():
        if path.is_symlink() or sha(path) != sha(pending):
            raise ValueError("existing render differs; preserve it")
        pending.unlink()
    else:
        pending.replace(path)


def _publish_json(path, value):
    if path.exists():
        if path.is_symlink() or json.dumps(
            _json(path), sort_keys=True, allow_nan=False
        ) != json.dumps(value, sort_keys=True, allow_nan=False):
            raise ValueError("existing render receipt differs; preserve it")
    else:
        atomic_json(path, value)


def render_bundle(
    mesh: str | Path, output: str | Path, *, settings: dict | None = None
) -> dict[str, Any]:
    """Render all components from an immutable bundle; geometry is never loaded globally."""
    from .surface_render import render_view

    settings = _render_settings(settings)
    output = Path(output)
    check_resources(
        output.parent,
        reserve_bytes=settings["reserve_bytes"]
        + settings["width"] * settings["height"] * 4
        + 1024**2,
    )
    with MeshReader(mesh) as reader:
        values = [item["level"] for item in reader.manifest["levels"]]
        selected = settings.get("levels", values)
        if any(v not in values for v in selected):
            raise ValueError("requested render level absent from bundle")
        allowed = {i for i, v in enumerate(values) if v in selected}
        palette = np.array(
            [
                colorsys.hsv_to_rgb((i * 0.618033988749895) % 1, 0.55, 0.9)
                for i in range(len(values))
            ]
        )
        binding = dict(
            schema="winding-render-plan-v1",
            mesh_sha256=reader.manifest["database_sha256"],
            settings=settings,
            metadata=reader.metadata,
            code={
                name: sha(Path(__file__).with_name(name))
                for name in ("surface_render.py", "surface_workflow.py")
            },
        )
        plan = output / "PLAN.json"
        if output.exists():
            if output.is_symlink() or not plan.is_file() or _json(plan) != binding:
                raise ValueError("foreign/changed render output")
        else:
            output.mkdir(exist_ok=False)
            atomic_json(plan, binding)
        with (output / ".render.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def triangles():
                for points, ids in reader.triangles(batch_size=settings["triangle_batch"]):
                    use = np.isin(ids, list(allowed))
                    if use.any():
                        yield points[use], palette[ids[use]]

            outputs = []
            for view in settings["views"]:
                check_resources(
                    output,
                    reserve_bytes=settings["reserve_bytes"]
                    + settings["width"] * settings["height"] * 4
                    + 1024**2,
                )
                reader.validate()
                if reader.bounds is None:
                    image = np.zeros((settings["height"], settings["width"], 4), np.uint8)
                    stats = dict(nonempty_pixels=0, triangles_processed=0, empty_geometry=True)
                else:
                    image, stats = render_view(
                        triangles,
                        reader.bounds,
                        width=settings["width"],
                        height=settings["height"],
                        azimuth=view["azimuth"],
                        elevation=view["elevation"],
                    )
                target = output / (view["name"] + ".png")
                _publish_png(target, image)
                entry = dict(
                    path=target.name,
                    sha256=sha(target),
                    bytes=target.stat().st_size,
                    view=view,
                    **stats,
                )
                outputs.append(entry)
                _publish_json(output / (view["name"] + ".json"), entry)
            reader.validate()
            result = dict(
                schema="winding-render-v1",
                status="RENDERED_DIAGNOSTIC",
                metadata=reader.metadata,
                bounds_xyz=reader.bounds,
                outputs=outputs,
                settings=settings,
                physical_acceptance=False,
                binding=binding,
            )
            _publish_json(output / "RENDER.json", result)
            return result


def run_surfaces(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    render: bool = True,
    stop=None,
) -> dict[str, Any]:
    """Build/import an agnostic mesh bundle and optionally render actual3D views."""
    if type(resume) is not bool or type(render) is not bool:
        raise ValueError("resume/render must be boolean")
    config = load_surface_config(config_path)
    output = Path(output)
    execution = config["execution"]

    def stopped():
        if sha(config["config_path"]) != config["config_sha256"]:
            raise ValueError("surface config changed during execution")
        return stop is not None and bool(stop())

    if "field" in config:
        field = config["field"]
        extractor = extract_batched_mesh
        binding = output / "RUN.json"
        if resume and binding.exists():
            if binding.is_symlink():
                raise ValueError("mesh RUN identity cannot be a symlink")
            saved = _json(binding)
            identity = saved.get("identity") if isinstance(saved, dict) else None
            if (
                isinstance(identity, dict)
                and identity.get("algorithm") == "lewiner-shifted-f32-adaptive-global-edges-v2"
            ):
                extractor = extract_mesh
        with ExitStack() as stack:
            lease = None
            if field.get("kind") == "chunks":
                if output.resolve().is_relative_to(Path(field["path"]).parent.resolve()):
                    raise ValueError("mesh output must not modify the source checkpoint directory")
                lease = stack.enter_context(_checkpoint_lease(field["path"]))

            def leased_stop():
                if sha(config["config_path"]) != config["config_sha256"]:
                    raise ValueError("surface config changed during execution")
                if lease is not None:
                    lease()
                return stop is not None and bool(stop())

            # Memory only: never less than one stored chunk, or large chunks decode per tile.
            cache = max(execution["cache_bytes"], chunk_cache_floor(field, config["geometry"]))
            with FieldSource(field, config["geometry"], cache_bytes=cache) as source:
                result = extractor(
                    source,
                    output,
                    config["levels"],
                    tile=execution["tile"],
                    level_batch=execution["level_batch"],
                    resume=resume,
                    stop=leased_stop,
                    reserve_bytes=execution["reserve_bytes"],
                )
    else:
        result = import_mesh(
            config["mesh"],
            output,
            config["geometry"],
            resume=resume,
            max_bytes=execution["mesh_bytes"],
            reserve_bytes=execution["reserve_bytes"],
        )
    if result["status"] != "MESH_READY":
        return result
    if sha(config["config_path"]) != config["config_sha256"]:
        raise ValueError("surface config changed during execution")
    if render:
        rendered = render_bundle(
            output,
            output / "renders",
            settings={**config["render"], "reserve_bytes": execution["reserve_bytes"]},
        )
        return dict(
            status="SURFACES_READY", mesh=result, renders=rendered, physical_acceptance=False
        )
    return result


def _surface_flags(parser):
    from .cli_settings import field_arguments

    field_arguments(parser)
    group = parser.add_argument_group(
        "surface request overrides", argument_default=argparse.SUPPRESS
    )
    group.add_argument("--mesh-sha256")
    group.add_argument(
        "--levels", type=float, nargs="+", help="raw field isovalues, NOT winding W labels"
    )
    for name in ("tile", "level_batch", "cache_bytes", "reserve_bytes", "mesh_bytes"):
        group.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("width", "height", "triangle_batch", "reserve_bytes"):
        group.add_argument("--render-" + name.replace("_", "-"), type=int)
    group.add_argument(
        "--render-levels", type=float, nargs="+", help="raw field isovalues to render"
    )
    view = group.add_argument(
        "--render-view",
        nargs=3,
        action="append",
        metavar=("NAME", "AZIMUTH", "ELEVATION"),
        help="named view; azimuth/elevation in degrees; replaces file views",
    )
    view.element_types = (str, float, float)


def _render_overrides(args):
    result = {
        name: vars(args)["render_" + name]
        for name in ("width", "height", "triangle_batch", "reserve_bytes", "levels")
        if "render_" + name in vars(args)
    }
    if hasattr(args, "render_view"):
        result["views"] = [
            dict(name=name, azimuth=float(az), elevation=float(el))
            for name, az, el in args.render_view
        ]
    return result


def _surface_recipe(args, sources):
    from .cli_settings import field_overrides, merge_explicit, persist_request

    output = args.output.resolve()
    path = args.config
    reference = None
    if args.resume:
        reference = (
            _json(output / "LEVELS.json")["binding"]
            if args.parallel_levels
            else _json(output / "REQUEST.json")
            if (output / "REQUEST.json").exists()
            else None
        )
        if reference is not None and sha(reference["config_path"]) != reference["config_sha256"]:
            raise ValueError("saved surface request changed")
        if path is None:
            if reference is None:
                raise ValueError("resume requires a saved request or --config")
            path = Path(reference["config_path"])
            if sha(path) != reference["config_sha256"]:
                raise ValueError("saved surface request changed")
    elif output.exists():
        raise ValueError("surface build requires a new output")
    overrides = field_overrides(args)
    if args.mesh is not None:
        overrides["mesh"] = dict(path=str(args.mesh.absolute()))
    if hasattr(args, "mesh_sha256"):
        overrides.setdefault("mesh", {})["sha256"] = args.mesh_sha256
    if hasattr(args, "levels"):
        overrides["levels"] = args.levels
    for name in ("tile", "level_batch", "cache_bytes", "reserve_bytes", "mesh_bytes"):
        if hasattr(args, name):
            overrides.setdefault("execution", {})[name] = getattr(args, name)
    render = _render_overrides(args)
    if render:
        overrides["render"] = render
    base = {}
    provenance = dict(sources)
    if path is not None:
        document = _json(path, sources=provenance)
        if document.get("schema") == "winding-field-cli-request-v1":
            loaded = load_surface_config(path)
            if document["output"] != str(output):
                raise ValueError("saved request belongs to another output")
            provenance.pop(str(path.resolve()))
            provenance.update(loaded.get("request_sources", {}))
            base = {
                k: v
                for k, v in loaded.items()
                if k not in ("config_path", "config_sha256", "request_sources")
            }
        else:
            base = deepcopy(document)
            entries = [base.get("field"), base.get("mesh")]
            if isinstance(base.get("field"), dict):
                entries.append(base["field"].get("mask"))
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                value = entry.get("path")
                if isinstance(value, str) and value and "://" not in value:
                    entry["path"] = str((path.resolve().parent / value).absolute())
        if not overrides:
            load_surface_config(path)
            return path
    config = validate_surface_config(
        merge_explicit({"schema": "winding-surfaces-v1", **base}, overrides), base=Path.cwd()
    )
    if reference is not None:
        previous = load_surface_config(reference["config_path"])
        previous = {
            k: v
            for k, v in previous.items()
            if k not in ("config_path", "config_sha256", "request_sources")
        }
        if config == previous:
            return Path(reference["config_path"])
        # Parallel collections bind their complete recipe; serial cache/reader budgets remain operational.
        scientific = deepcopy(config)
        old = deepcopy(previous)
        if not args.parallel_levels:
            for value in (scientific, old):
                value["execution"] = {k: v for k, v in value["execution"].items() if k == "tile"}
        if scientific != old:
            raise ValueError("surface request differs from retained problem")
    if "field" in config:
        with FieldSource(config["field"], config["geometry"], cache_bytes=0) as source:
            source.validate(hashes=args.resume or config["field"].get("kind") != "zarr-v2")
    elif sha(config["mesh"]["path"]) != config["mesh"]["sha256"]:
        raise ValueError("mesh source hash differs")
    check_resources(output.parent, reserve_bytes=config["execution"]["reserve_bytes"])
    for source, pin in sources.items():
        if sha(source) != pin:
            raise ValueError("surface request source changed")
    if path is not None and document.get("schema") != "winding-field-cli-request-v1":
        if sha(path) != provenance[str(path.resolve())]:
            raise ValueError("surface request source changed")
    return persist_request(output, config, resume=False, sources=provenance)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="winding-field prepared", description=__doc__)
    parser.add_argument("action", choices=("mesh", "render", "surfaces"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--parallel-levels",
        action="store_true",
        help="independent per-level bundles; automatically bound CPU/RAM/disk concurrency",
    )
    parser.add_argument(
        "--then-flatten",
        type=Path,
        metavar="DIR",
        help="mesh only: after MESH_READY, run export-sheets on the bundle into DIR as a "
        "separate second step with its own receipts (CHAIN.json records both)",
    )
    parser.add_argument(
        "--flatten-cli-config",
        type=Path,
        metavar="FILE",
        help="export-sheets settings for --then-flatten (source, flattening_solver, scope, ...)",
    )
    _surface_flags(parser)
    from .cli_settings import parse_options, merge_explicit

    sources = {}
    args = parse_options(parser, argv, sources=sources)
    config_path = None
    chain = None
    try:
        if args.flatten_cli_config is not None and args.then_flatten is None:
            raise ValueError("--flatten-cli-config requires --then-flatten")
        if args.then_flatten is not None and (args.action != "mesh" or args.parallel_levels):
            raise ValueError("--then-flatten requires the mesh action without --parallel-levels")
        if args.then_flatten is not None:
            from .mesh_chain import check_chain

            check_chain(args.output, args.then_flatten, args.flatten_cli_config)
        if args.parallel_levels:
            if args.action == "render":
                raise ValueError("--parallel-levels requires mesh/surfaces")
            from .surface_parallel import run_parallel_levels

            config_path = _surface_recipe(args, sources)
            result = run_parallel_levels(
                config_path, args.output, resume=args.resume, render=args.action == "surfaces"
            )
            print(json.dumps(result, allow_nan=False))
            return {"LEVELS_READY": 0, "CHECKPOINTED_STOP": 3, "RESOURCE_UNAVAILABLE": 75}.get(
                result["status"], 1
            )
        if args.action == "render":
            from .cli_settings import field_overrides

            if field_overrides(args) or any(
                hasattr(args, name)
                for name in (
                    "levels",
                    "tile",
                    "level_batch",
                    "cache_bytes",
                    "reserve_bytes",
                    "mesh_bytes",
                    "mesh_sha256",
                )
            ):
                raise ValueError("render accepts render settings, not extraction/source overrides")
            if args.mesh is None:
                raise ValueError("render requires --mesh BUNDLE")
            settings = merge_explicit(
                _json(args.config) if args.config else {}, _render_overrides(args)
            )
            result = render_bundle(args.mesh, args.output, settings=settings)
        else:
            config_path = _surface_recipe(args, sources)
            result = run_surfaces(
                config_path, args.output, resume=args.resume, render=args.action == "surfaces"
            )
            if args.then_flatten is not None:
                from .mesh_chain import chain_flatten

                chain = chain_flatten(
                    args.output, result, args.then_flatten, args.flatten_cli_config
                )
                result = dict(result, chain=chain)
    except ResourceUnavailable as error:
        print("resources unavailable: " + str(error), file=sys.stderr)
        return 75
    except (ValueError, OSError, sqlite3.Error, EOFError, BadZipFile, KeyError, TypeError) as error:
        print("surface input/integrity error: " + str(error), file=sys.stderr)
        return 2
    except RuntimeError as error:
        print("surface processing failed: " + str(error), file=sys.stderr)
        return 1
    finally:
        if config_path is not None and (args.output / "RUN.json").is_file():
            document = _json(config_path)
            if (
                document.get("schema") == "winding-field-cli-request-v1"
                and not (args.output / "REQUEST.json").exists()
            ):
                atomic_json(
                    args.output / "REQUEST.json",
                    dict(config_path=str(config_path), config_sha256=sha(config_path)),
                )
    print(json.dumps(result, allow_nan=False))
    if chain is not None and chain["steps"][1].get("returncode"):
        return chain["steps"][1]["returncode"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
