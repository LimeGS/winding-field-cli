"""Exact shared domains with explicit LSCM or checkpointed native SLIM."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import patches as p
from .legacy_charts import cross2
from ..overlap_audit import audit_uv
from .native_sheet import cut_component, mean_value_map
from .original import face_quality, scale_uv_to_world
from .sheet_atlas import _arrays, _json, _verified
from ..surface_atlas import face_components, local_chart, split_faces
from ..util import atomic_json, file_sha
from .selection import SCOPES, SOLVERS


def settings(request, solver, scope):
    if solver not in SOLVERS or scope not in SCOPES:
        raise ValueError("invalid flattening selection")
    if not isinstance(request, dict):
        raise ValueError("matrix settings must be an object")
    allowed = {
        "output_frame",
        "pitch_um",
        "tile_size",
        "memory_gib",
        "process_tree_gib",
        "disk_reserve_gib",
        "cache_dir",
        "cache_mib",
        "cache_gib",
        "ct_workers",
        "raster_workers",
    }
    if scope == "patches":
        allowed |= {"max_patch_faces", "max_charts"}
    else:
        allowed |= {"centerline", "reference_z"}
    if solver == "slim":
        allowed |= {"app_run", "native_recipe"}
    if set(request) - allowed:
        raise ValueError(
            "unsupported matrix settings for selection: " + str(sorted(set(request) - allowed))
        )
    extra = {k: request[k] for k in ("centerline", "reference_z") if k in request}
    if "centerline" in extra:
        from .sheet_prepare import centerline_points

        extra["centerline"] = centerline_points(extra["centerline"]).tolist()
    if "reference_z" in extra and (
        "centerline" not in extra
        or type(extra["reference_z"]) not in (int, float)
        or not np.isfinite(extra["reference_z"])
    ):
        raise ValueError("reference_z requires centerline and a finite height")
    base = dict(memory_gib=None, process_tree_gib=None, native_recipe="historical")
    base.update({k: v for k, v in request.items() if k not in extra})
    cfg = p.settings(base, native_required=solver == "slim")
    if solver == "slim" and cfg["native_recipe"] not in p.CHECKPOINT_RECIPES:
        raise ValueError("native_recipe is not checkpoint-qualified")
    cfg.update(extra, flattening_solver=solver, scope=scope)
    return cfg


def solve(cut, output, cfg, guard):
    v, f = cut["vertices"], cut["faces"]
    guard()
    if cfg["flattening_solver"] == "lscm":
        from .lscm import lscm

        b = cut["boundary"]
        first = int(b[0])
        second = int(b[np.argmax(np.linalg.norm(v[b] - v[first], axis=1))])
        # Pin length is measured on the uncentered mesh, the solve on the centered one.
        length = float(np.linalg.norm(v[second] - v[first]))
        uv, pins, fixed = lscm(v - v.mean(0), f, first, second, length=length)
        guard()
        return uv, dict(
            algorithm="LSCM",
            backend="libigl",
            initialization=None,
            pins=pins.tolist(),
            pin_uv=fixed.tolist(),
        )
    from . import flatboi_checkpoint as native

    output = Path(output)
    initial = output / "initial-map.npz"
    obj = output / "inputmesh.obj"
    prepared = output / "INITIAL.json"
    if prepared.exists():
        record = _json(prepared)
        for name, sha in record["files"].items():
            _verified(output, name, sha)
        data = _arrays(initial)
        np.testing.assert_array_equal(data["world"], v)
        np.testing.assert_array_equal(data["faces"], f)
        initialization = record["initialization"]
    else:
        uv, initialization = mean_value_map(cut, guard, diagnostics=output / "initialization")
        np.savez_compressed(initial, world=v, faces=f, uv=uv)
        with obj.open("w") as stream:
            np.savetxt(stream, v, fmt="v %.17g %.17g %.17g")
            np.savetxt(stream, uv, fmt="vt %.17g %.17g")
            np.savetxt(stream, np.repeat(f + 1, 2, axis=1), fmt="f %d/%d %d/%d %d/%d")
        atomic_json(
            prepared,
            dict(
                initialization=initialization, files={q.name: file_sha(q) for q in (initial, obj)}
            ),
        )
    result = native.run(
        SimpleNamespace(
            source_map=initial,
            input=obj,
            output=output / "native",
            app_run=cfg.get("app_run"),
            batch_iterations=10,
            iterations=200,
            hours=12.0,
            native_recipe=cfg["native_recipe"],
        ),
        guard=guard,
    )
    checkpoint = result["checkpoint"]
    guard()
    uv, binding = native.positive_uv(output / "native" / checkpoint["obj"], v, f, guard)
    return uv, dict(
        algorithm="SLIM",
        backend="native-Flatboi",
        initialization=initialization,
        checkpoint=checkpoint,
        binding=binding,
    )


def build_map(vertices, faces, output, cfg, *, guard=lambda: None):
    v, f = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    if (
        v.ndim != 2
        or v.shape[1] != 3
        or not np.isfinite(v).all()
        or f.ndim != 2
        or f.shape[1] != 3
        or f.dtype.kind not in "iu"
        or not len(f)
        or f.min() < 0
        or f.max() >= len(v)
    ):
        raise ValueError("finite indexed triangles required")
    f = f.astype(np.int64)
    out = Path(output).absolute()
    if any(q.is_symlink() for q in (out, *out.parents)) or any(
        q.is_symlink() for q in out.rglob("*")
    ):
        raise ValueError("matrix map output symlink refused")
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(
        vertices_sha256=hashlib.sha256(v.tobytes()).hexdigest(),
        faces_sha256=hashlib.sha256(f.tobytes()).hexdigest(),
        settings=cfg,
    )
    if (out / "IDENTITY.json").exists():
        if _json(out / "IDENTITY.json") != identity:
            raise ValueError("matrix map identity changed")
    else:
        atomic_json(out / "IDENTITY.json", identity)
    if (out / "MAP.json").exists():
        saved = _json(out / "MAP.json")
        for name, sha in saved["files"].items():
            _verified(out, name, sha)
        return dict(_arrays(out / "map.npz"), report=saved["report"])
    components = face_components(f, np.arange(len(f)))
    pending = list(reversed(components))
    domains = []
    while pending:
        guard()
        rows = pending.pop()
        if cfg["scope"] == "patches" and len(rows) > cfg["max_patch_faces"]:
            pending.extend(reversed(split_faces(v, f, rows)))
            continue
        domains.append(rows)
        if cfg["scope"] == "patches" and len(domains) + len(pending) > cfg["max_charts"]:
            raise ValueError("matrix domain budget exceeded; no faces omitted")
    partition = np.empty(len(f), np.int64)
    for i, rows in enumerate(domains):
        partition[rows] = i
    report = dict(
        flattening_solver=cfg["flattening_solver"],
        scope=cfg["scope"],
        source_faces=len(f),
        source_components=len(components),
        partition_sha256=hashlib.sha256(partition.tobytes()).hexdigest(),
        domains=[],
        welded=False,
        disconnected_components_translated=len(components) > 1,
        patch_repacking=cfg["scope"] == "patches" and len(domains) > 1,
        placement="independent domains; isotropic calibration, Z orientation where defined, translation only",
        physical_acceptance=False,
    )
    parts = []
    offset = 0.0
    vertex_offset = 0
    for i, rows in enumerate(domains):
        guard()
        directory = out / f"domain-{i:06d}"
        directory.mkdir(exist_ok=True)
        cv, cf, vi = local_chart(v, f, rows)
        if cfg["scope"] == "sheet" and "centerline" in cfg:
            from .sheet_prepare import reference_cut

            cut, topology = reference_cut(cv, cf, cfg["centerline"], guard)
        else:
            cut, topology = cut_component(cv, cf, guard)
        if (directory / "DOMAIN.json").exists():
            solved = _json(directory / "DOMAIN.json")
            for name, sha in solved["files"].items():
                _verified(directory, name, sha)
            uv = _arrays(directory / "uv.npz")["uv"]
            solver = solved["solver"]
        else:
            uv, solver = solve(cut, directory, cfg, guard)
            if uv.shape != (len(cut["vertices"]), 2) or not np.isfinite(uv).all():
                raise ValueError("solver returned invalid UVs")
            np.savez_compressed(directory / "uv.npz", uv=uv)
            files = {
                str(q.relative_to(directory)): file_sha(q)
                for q in sorted(directory.rglob("*"))
                if q.is_file() and q.name != ".lock"
            }
            atomic_json(directory / "DOMAIN.json", dict(solver=solver, files=files))
        world, tri = cut["vertices"], cut["faces"]
        t = uv[tri]
        signed = cross2(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        if (signed < 0).sum() > (signed > 0).sum():
            uv = uv * np.array([-1.0, 1.0])
        uv, calibration = scale_uv_to_world(world, tri, uv)
        direction = np.linalg.lstsq(uv - uv.mean(0), world[:, 2] - world[:, 2].mean(), rcond=None)[
            0
        ]
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
            rotation = np.column_stack([[direction[1], -direction[0]], direction])
            uv = uv @ rotation
            calibration["rotation_toward_Z"] = rotation.tolist()
        anchor = None
        if "reference_z" in cfg:
            from .sheet_finish import reference_anchor

            if world[:, 2].min() <= cfg["reference_z"] <= world[:, 2].max():
                anchor = reference_anchor(world, tri, uv, cfg["centerline"], cfg["reference_z"])
        shift = np.array([offset, 0.0]) - uv.min(0)
        uv += shift
        if anchor:
            anchor["uv"] = (np.asarray(anchor["uv"]) + shift).tolist()
        offset = uv[:, 0].max() + max(
            cfg["pitch_um"] / cfg["output_frame"]["voxel_um"] * 2,
            float(np.ptp(uv, axis=0).max()) * 0.05,
        )
        parent = vi[cut["parent"]]
        np.testing.assert_array_equal(world, v[parent])
        np.testing.assert_array_equal(parent[tri], f[rows])
        parts.append(
            dict(
                world=world,
                faces=tri + vertex_offset,
                uv=uv,
                parent=parent,
                source_face_rows=rows,
                domain_ids=np.full(len(rows), i, np.int64),
            )
        )
        vertex_offset += len(world)
        report["domains"].append(
            dict(
                id=i,
                faces=len(rows),
                cut=topology,
                solver=solver,
                calibration=calibration,
                translation=shift.tolist(),
                reference=anchor,
                receipt_sha256=file_sha(directory / "DOMAIN.json"),
            )
        )
    result = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    np.testing.assert_array_equal(np.sort(result["source_face_rows"]), np.arange(len(f)))
    display, scientific, quality = face_quality(result["world"], result["faces"], result["uv"])
    audit = audit_uv(result["uv"], result["faces"], out, guard=guard)
    if audit["status"] != "COMPLETE":
        raise ValueError("matrix overlap audit incomplete")
    involved = audit["involved"]
    report.update(
        quality=quality,
        overlap={k: v for k, v in audit.items() if k != "involved"},
        overlapping_faces=audit["overlapping_faces"],
    )
    result.update(
        display=display, scientific=scientific & ~involved, omitted_face_rows=np.empty(0, np.int64)
    )
    np.savez_compressed(out / "map.npz", **result)
    files = {
        str(q.relative_to(out)): file_sha(q)
        for q in sorted(out.rglob("*"))
        if q.is_file() and q.name != ".lock"
    }
    guard()
    atomic_json(out / "MAP.json", dict(identity=identity, report=report, files=files))
    return dict(result, report=report)
