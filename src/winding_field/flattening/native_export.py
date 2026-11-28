"""``export-sheets`` default: every level through the ``scroll render-ct`` sheet route.

The route that flattened PHerc0175A W −66 (handles set aside, holes kept, wrap ends
seamed, LSCM then unrolled seed, one Flatboi SLIM solve per piece, stretch gate, chart
fallback; :func:`winding_field.flattening.native_global.flatten_sheet`) is the default
in every lane. ``run``/``plan`` and ``scroll render-ct`` already use it; this module
makes ``export-sheets`` (and so ``prepared mesh --then-flatten``) use it too, by running
each level of the bundle through ``render-ct`` itself rather than a second copy of it.

The matrix (``--flattening-solver`` × ``--scope``) and the historical recipes
(``--flattening-method original|patches``) stay available as experimental lanes.
"""

import fcntl
import json
import math
from pathlib import Path

import numpy as np

from ..util import atomic_json, file_sha

SCHEMA = "export-sheets-native-v1"
ROUTE = "native-sheet"
# Settings this route reads; anything else belongs to an experimental lane.
ACCEPTED = {"pitch_um", "cache_dir", "cache_gib", "workers", "output_frame"}
GEOMETRY_KEYS = ("frame", "voxel_um", "origin_xyz", "spacing_xyz", "shape_zyx", "sheet_step")


def available():
    """The route needs Flatboi SLIM, which runs on Linux x86_64 only."""
    from .flatboi import supported_platform

    return supported_platform()


def _request(args):
    from ..export_settings import flattening_request

    request = {k: v for k, v in flattening_request(args).items() if v is not None}
    extra = sorted(set(request) - ACCEPTED)
    if extra:
        raise ValueError(
            "settings not used by the default native sheet route: "
            + ", ".join(extra)
            + "; choose an experimental lane with --flattening-solver/--scope or "
            "--flattening-method original|patches"
        )
    return request


def _check_frame(frame, metadata):
    """An explicit output frame must be the mesh's own: this route writes the CT frame."""
    if not frame:
        return
    own = dict(
        id=metadata.get("frame"),
        origin_xyz_um=[float(t) for t in metadata.get("origin_xyz", ())],
        voxel_um=metadata.get("voxel_um"),
    )
    for key, value in frame.items():
        expected = own.get(key)
        same = (
            np.allclose(value, expected)
            if key != "id" and expected is not None
            else value == expected
        )
        if not same:
            raise ValueError(
                f"output frame {key} {value!r} differs from the mesh frame {expected!r}; the "
                "default native sheet route writes the mesh frame (use --flattening-solver/"
                "--scope for another one)"
            )


def _level_bundle(reader, level, output):
    """A single-level bundle of ``level``; reused when it is already complete."""
    from ..surface_mesh import import_mesh

    mesh = output / "mesh"
    if (mesh / "MESH.json").is_file():
        return mesh
    output.mkdir(parents=True, exist_ok=True)
    db = reader.db
    ids, xyz = [], []
    for row in db.execute("SELECT id,x,y,z FROM vertices WHERE level=? ORDER BY id", (level,)):
        ids.append(row[0])
        xyz.append(row[1:])
    faces = np.asarray(
        db.execute("SELECT a,b,c FROM faces WHERE level=? ORDER BY id", (level,)).fetchall(),
        np.int64,
    )
    ids = np.asarray(ids, np.int64)
    reader.validate()
    source = output / "level.npz"
    np.savez(
        source,
        vertices_xyz=np.asarray(xyz, np.float64).reshape(-1, 3),
        faces=np.searchsorted(ids, faces).reshape(-1, 3),
    )
    geometry = {k: reader.metadata[k] for k in GEOMETRY_KEYS}
    result = import_mesh(
        {"path": str(source), "sha256": file_sha(source)},
        mesh,
        geometry,
        resume=(mesh / "RUN.json").exists(),
    )
    if result.get("status") != "MESH_READY":
        raise ValueError(f"level {level} mesh import did not complete")
    return mesh


def _render_argv(args, request, mesh, output, pitch):
    argv = ["render-ct", "--mesh", str(mesh), "--source", str(args.source)]
    argv += ["--level", str(args.level or 0), "--render-pitch-um", repr(float(pitch))]
    if args.voxel_um is not None:
        argv += ["--voxel-um", repr(float(args.voxel_um))]
    if getattr(args, "native_local", None):
        argv += ["--local", str(args.native_local)]
    if request.get("cache_dir"):
        argv += ["--cache", str(request["cache_dir"])]
    if request.get("cache_gib") is not None:
        argv += ["--cache-gib", repr(float(request["cache_gib"]))]
    if request.get("workers") is not None:
        argv += ["--flatten-workers", str(int(request["workers"]))]
    return argv + ["--output", str(output)]


def short(result):
    """Why a native export falls short (a level paused or under its target), else None."""
    from ..scroll import render_short

    reasons = [
        f"{name}: {render_short(level)}"
        for name, level in result.get("levels", {}).items()
        if render_short(level)
    ]
    return "; ".join(reasons) or None


def export_existing(args):
    """Flatten and render every level of a mesh (or field) source with ``render-ct``."""
    from ..scroll import _parser, render_ct
    from ..surface_mesh import MeshReader
    from ..workflow import implementation_sha

    if getattr(args, "stages", "all") != "all":
        raise ValueError(
            "--stages uv|ct belongs to the experimental matrix lane; pass "
            "--flattening-solver and --scope with it"
        )
    if not available():
        raise ValueError(
            "the default native sheet route needs Flatboi SLIM (Linux x86_64); elsewhere "
            "choose --flattening-solver lscm --scope sheet"
        )
    request = _request(args)
    descriptor = getattr(args, "resolved_field", None)
    if descriptor is not None and not args.levels:
        raise ValueError("--levels required with a field source")
    if args.mesh_bundle and args.levels:
        raise ValueError("mesh bundle already binds levels; omit --levels")
    out = Path(args.output).expanduser().absolute()
    if any(q.is_symlink() for q in (out, *out.parents)):
        raise ValueError("export output symlink refused")
    mesh = Path(args.mesh_bundle).expanduser().resolve() if args.mesh_bundle else out / "meshes"
    if args.mesh_bundle and (out == mesh or out in mesh.parents or mesh in out.parents):
        raise ValueError("export output must be separate from the source mesh")
    mesh_sha = None
    if args.mesh_bundle:
        with MeshReader(mesh) as reader:
            mesh_sha = reader.manifest_sha
            _check_frame(request.get("output_frame"), reader.metadata)
    identity = dict(
        schema=SCHEMA,
        route=ROUTE,
        mesh=str(mesh),
        mesh_sha256=mesh_sha,
        field=descriptor,
        levels=args.levels,
        ct_source=args.source,
        ct_level=args.level or 0,
        ct_voxel_um=args.voxel_um,
        settings={k: v for k, v in request.items() if k not in ("cache_dir", "workers")},
        implementation_sha256=implementation_sha(),
    )
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "EXPORT.json").exists():
            saved = json.loads((out / "EXPORT.json").read_text())
            if saved != identity:
                raise ValueError(
                    "export identity changed: this output was made with other inputs, settings "
                    "or flattening route (an output of the matrix lane resumes with its "
                    "--flattening-solver/--scope)"
                )
        else:
            atomic_json(out / "EXPORT.json", identity)
        if descriptor is not None:
            from ..surface_mesh import extract_mesh
            from ..surface_sources import FieldSource

            with FieldSource(descriptor["field"], descriptor["geometry"]) as source:
                made = extract_mesh(source, mesh, args.levels, resume=mesh.exists())
            if made["status"] != "MESH_READY":
                return made
        parser = _parser()
        levels = {}
        with MeshReader(mesh) as reader:
            if mesh_sha is not None and reader.manifest_sha != mesh_sha:
                raise ValueError("source mesh changed")
            _check_frame(request.get("output_frame"), reader.metadata)
            pitch = request.get("pitch_um") or float(min(reader.metadata["spacing_xyz"]))
            if not (math.isfinite(pitch) and pitch > 0):
                raise ValueError("pitch_um must be a positive number")
            catalog = reader.db.execute("SELECT id,value FROM levels ORDER BY id").fetchall()
            for level, value in catalog:
                target = out / f"level-{level:04d}"
                if len(catalog) == 1:
                    source_mesh = mesh
                else:
                    source_mesh = _level_bundle(reader, level, out / "level-meshes" / target.name)
                result = render_ct(
                    parser.parse_args(_render_argv(args, request, source_mesh, target, pitch))
                )
                levels[target.name] = dict(result, level_value=float(value))
            reader.validate()
        complete = all(r.get("status") != "PAUSED" for r in levels.values())
        receipt = dict(
            status="SHEETS_READY" if complete else "PAUSED",
            schema=SCHEMA,
            route=ROUTE,
            identity_sha256=file_sha(out / "EXPORT.json"),
            pitch_um=pitch,
            levels=levels,
            level_receipts={
                name: file_sha(out / name / "LEVEL.json")
                for name, r in levels.items()
                if (out / name / "LEVEL.json").is_file()
            },
        )
        atomic_json(out / "SHEETS.json", receipt)
        return receipt
