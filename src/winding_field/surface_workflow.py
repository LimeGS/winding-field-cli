"""Scroll-independent field/mesh -> durable indexed surfaces -> bounded CPU3D views."""

import argparse
import colorsys
import fcntl
import json
import math
import os
import re
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from field_runtime import ResourceUnavailable, check_resources
from field_store import atomic_json
from .field_workflow import _invalid, _object
from surface_mesh import MeshReader, _integer, _levels, extract_mesh, import_mesh, sha
from surface_sources import FieldSource


def _keys(value, allowed):
    if not isinstance(value, dict) or value.keys() - set(allowed):
        raise ValueError("unknown/invalid surface configuration fields")


def _json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(1024**2 + 1)
    if len(raw) > 1024**2:
        raise ValueError("surface config exceeds1MiB")
    return json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid)


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
        "views": [dict(name="front", azimuth=0.0, elevation=0.0), dict(name="oblique", azimuth=35.0, elevation=20.0)],
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
    _keys(config, {"schema", "field", "mesh", "geometry", "levels", "execution", "render"})
    if config.get("schema") != "winding-surfaces-v1" or ("field" in config) == ("mesh" in config):
        raise ValueError("surface schema and exactly one field/mesh source required")
    geometry = config.get("geometry")
    _keys(geometry, {"frame", "voxel_um", "origin_xyz", "spacing_xyz", "shape_zyx", "sheet_step", "calibration"})
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
        _keys(field, {"kind", "path", "sha256", "mask", "support_policy"})
        descriptor = _artifact({k: field[k] for k in ("path", "sha256") if k in field}, path.parent)
        descriptor.update({k: v for k, v in field.items() if k not in descriptor})
        if "mask" in field:
            descriptor["mask"] = _artifact(field["mask"], path.parent)
        config["field"] = descriptor
        config["levels"] = _levels(config.get("levels"))
        if len(config["levels"]) > 4096:
            raise ValueError("at most4096 levels per bundle")
    else:
        if "levels" in config:
            raise ValueError("imported meshes have group IDs, not declared isovalues")
        config["mesh"] = _artifact(config["mesh"], path.parent)
    _keys(config.get("execution", {}), {"tile", "level_batch", "cache_bytes", "reserve_bytes", "mesh_bytes"})
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
        config_path=str(path),
        config_sha256=sha(path),
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
        if path.is_symlink() or json.dumps(_json(path), sort_keys=True, allow_nan=False) != json.dumps(
            value, sort_keys=True, allow_nan=False
        ):
            raise ValueError("existing render receipt differs; preserve it")
    else:
        atomic_json(path, value)


def render_bundle(mesh: str | Path, output: str | Path, *, settings: dict | None = None) -> dict[str, Any]:
    """Render all components from an immutable bundle; geometry is never loaded globally."""
    from .surface_render import render_view

    settings = _render_settings(settings)
    output = Path(output)
    check_resources(
        output.parent, reserve_bytes=settings["reserve_bytes"] + settings["width"] * settings["height"] * 4 + 1024**2
    )
    with MeshReader(mesh) as reader:
        values = [item["level"] for item in reader.manifest["levels"]]
        selected = settings.get("levels", values)
        if any(v not in values for v in selected):
            raise ValueError("requested render level absent from bundle")
        allowed = {i for i, v in enumerate(values) if v in selected}
        palette = np.array([colorsys.hsv_to_rgb((i * 0.618033988749895) % 1, 0.55, 0.9) for i in range(len(values))])
        binding = dict(
            schema="winding-render-plan-v1",
            mesh_sha256=reader.manifest["database_sha256"],
            settings=settings,
            metadata=reader.metadata,
            code={name: sha(Path(__file__).with_name(name)) for name in ("surface_render.py", "surface_workflow.py")},
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
                    reserve_bytes=settings["reserve_bytes"] + settings["width"] * settings["height"] * 4 + 1024**2,
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
                entry = dict(path=target.name, sha256=sha(target), bytes=target.stat().st_size, view=view, **stats)
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
    config_path: str | Path, output: str | Path, *, resume: bool = False, render: bool = True, stop=None
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
        with FieldSource(config["field"], config["geometry"], cache_bytes=execution["cache_bytes"]) as source:
            result = extract_mesh(
                source,
                output,
                config["levels"],
                tile=execution["tile"],
                level_batch=execution["level_batch"],
                resume=resume,
                stop=stopped,
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
            output, output / "renders", settings={**config["render"], "reserve_bytes": execution["reserve_bytes"]}
        )
        return dict(status="SURFACES_READY", mesh=result, renders=rendered, physical_acceptance=False)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="winding-field prepared", description=__doc__)
    parser.add_argument("action", choices=("mesh", "render", "surfaces"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "render":
            if args.mesh is None:
                raise ValueError("render requires --mesh BUNDLE")
            result = render_bundle(args.mesh, args.output, settings=_json(args.config) if args.config else None)
        else:
            if args.config is None:
                raise ValueError("mesh/surfaces requires --config")
            result = run_surfaces(args.config, args.output, resume=args.resume, render=args.action == "surfaces")
    except ResourceUnavailable as error:
        print("resources unavailable: " + str(error), file=sys.stderr)
        return 75
    except (ValueError, OSError, sqlite3.Error, EOFError, BadZipFile, KeyError, TypeError) as error:
        print("surface input/integrity error: " + str(error), file=sys.stderr)
        return 2
    except RuntimeError as error:
        print("surface processing failed: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
