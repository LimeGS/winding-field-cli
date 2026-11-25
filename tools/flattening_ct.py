"""One physically calibrated CT renderer for all benchmark candidates."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import product
from pathlib import Path
import sqlite3
import threading

import numpy as np
from PIL import Image

from winding_field.io import Volume
from winding_field.flatten import cross2
from winding_field.native_ct_sampling import NativeCTSampler
from winding_field.original_recipe import raster_sheet
from winding_field.util import atomic_json, digest, file_sha
from bench_flattening import case_arrays, locked, read_json, safe, seal, verify


def validate_profile(case, profile):
    if profile["field_id"] != case["field_id"] or profile["frame"] != case["frame"]:
        raise ValueError("CT field/frame binding differs")
    if "ct_profile_sha256" in case and digest(profile) != case["ct_profile_sha256"]:
        raise ValueError("CT profile binding changed")


class CheckedCT:
    def __init__(self, config, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.profile = config["profile"]
        p = self.profile
        self.volume = Volume(
            p["source"],
            level=p["level"],
            voxel_um=p["spacing_xyz_um"][0],
            cache=config["cache_dir"],
            cache_bytes=config["cache_bytes"],
            decoded_cache_bytes=config["decoded_cache_bytes"],
            reserve_bytes=config["reserve_bytes"],
            metadata_pins=p.get("metadata_pins"),
        )
        if (
            list(self.volume.spacing_xyz) != p["spacing_xyz_um"]
            or list(self.volume.origin_xyz) != p["origin_xyz_um"]
            or list(self.volume.shape) != p["shape_zyx"]
        ):
            self.volume.close()
            raise ValueError("CT physical metadata binding changed")
        try:
            if self.volume.remote and not p.get("metadata_pins"):
                raise ValueError("remote CT requires explicit metadata pins")
            identity = self.root / "VOLUME.json"
            if identity.exists():
                if digest(read_json(identity)) != digest(self.volume.identity):
                    raise ValueError("resolved CT metadata/runtime identity changed")
            else:
                atomic_json(identity, self.volume.identity)
        except BaseException:
            self.volume.close()
            raise
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "decoded-chunks.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS chunks(key TEXT PRIMARY KEY,sha TEXT)")
        self.volume.bind_ledger(self.root / "encoded-chunks.sqlite")
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.sampler = NativeCTSampler(
            self.volume.shape, self.volume.chunks, self.volume.dtype, self._chunk
        )

    def _key(self, index):
        name = self.volume.metadata.get("dimension_separator", ".").join(map(str, index))
        return self.volume.array_path + "/" + name if self.volume.array_path else name

    def _chunk(self, index):
        key = self._key(index)
        try:
            data = self.volume.decoded_chunk(key)
        except FileNotFoundError:
            data = None
        sha = hashlib.sha256(data.tobytes()).hexdigest() if data is not None else None
        with self.lock:
            old = self.db.execute("SELECT sha FROM chunks WHERE key=?", (key,)).fetchone()
            if old is not None and old != (sha,):
                raise ValueError("CT chunk/availability changed: " + key)
            self.db.execute("INSERT OR IGNORE INTO chunks VALUES (?,?)", (key, sha))
            self.db.commit()
        return data

    def _preload(self, index):
        try:
            self.volume._chunk_bytes(self._key(index))
        except FileNotFoundError:
            pass

    def __enter__(self):
        try:
            self.volume.validate_used()
            for key, sha in self.db.execute("SELECT key,sha FROM chunks").fetchall():
                if sha is None:
                    local = (
                        key.split("/")[-1]
                        if self.volume.metadata.get("dimension_separator", ".") == "."
                        else key.removeprefix(self.volume.array_path + "/")
                    )
                    separator = self.volume.metadata.get("dimension_separator", ".")
                    self._chunk(tuple(map(int, local.split(separator))))
            self.scope = self.volume.verified_chunks()
            self.scope.__enter__()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *exc):
        try:
            return self.scope.__exit__(*exc)
        finally:
            self.close()

    def close(self):
        self.sampler.close()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.db.close()
        self.volume.close()

    def sample(self, xyz_um):
        xyz_um = np.asarray(xyz_um, float)
        if xyz_um.ndim != 2 or xyz_um.shape[1] != 3:
            raise ValueError("XYZ points required")
        v = self.volume
        grid = (xyz_um - v.origin_xyz) / v.spacing_xyz
        values, observed = [], []
        for first in range(0, len(grid), 65536):
            block = grid[first : first + 65536]
            good = (
                np.isfinite(block).all(1)
                & (block >= 0).all(1)
                & (block <= np.array(v.shape[::-1]) - 1).all(1)
            )
            zyx = block[good, ::-1]
            lower, fraction = np.floor(zyx).astype(np.int64), zyx % 1
            indices = []
            for corner in product((0, 1), repeat=3):
                used = np.prod(np.where(corner, fraction, 1 - fraction), axis=1) > 0
                indices.append((lower[used] + corner) // v.chunks)
            keys = np.unique(np.concatenate(indices), axis=0)
            list(self.pool.map(self._preload, map(tuple, keys)))
            result, valid = self.sampler.sample(block)
            values.append(result)
            observed.append(valid)
        return (
            (np.concatenate(values), np.concatenate(observed))
            if values
            else (np.empty(0, np.float32), np.empty(0, bool))
        )


def original_normals(vertices, faces):
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    result = np.zeros_like(vertices, dtype=float)
    for corner in range(3):
        np.add.at(result, faces[:, corner], normal)
    length = np.linalg.norm(result, axis=1)
    return np.divide(
        result, length[:, None], out=np.full_like(result, np.nan), where=length[:, None] > 0
    )


def normals_at(v, f, normals, points, rows):
    tri = v[f[rows]]
    a, b, q = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], points - tri[:, 0]
    aa, ab, bb = (a * a).sum(1), (a * b).sum(1), (b * b).sum(1)
    qa, qb = (q * a).sum(1), (q * b).sum(1)
    den = aa * bb - ab * ab
    w1, w2 = (qa * bb - qb * ab) / den, (qb * aa - qa * ab) / den
    n = normals[f[rows]]
    result = n[:, 0] + w1[:, None] * (n[:, 1] - n[:, 0]) + w2[:, None] * (n[:, 2] - n[:, 0])
    length = np.linalg.norm(result, axis=1)
    return np.divide(
        result, length[:, None], out=np.full_like(result, np.nan), where=length[:, None] > 0
    )


def quadrature_barycentric(subdivisions=1):
    triangles = np.eye(3)[None]
    for _ in range(subdivisions):
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        ab, ac, bc = (a + b) / 2, (a + c) / 2, (b + c) / 2
        triangles = np.concatenate(
            [np.stack(t, axis=1) for t in ((a, ab, ac), (ab, b, bc), (ac, bc, c), (ab, bc, ac))]
        )
    return triangles.mean(1)


def prepare_rois(case, output):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    a = case_arrays(case)
    v, f = a["world_um"], a["faces"]
    tri, normals = v[f], original_normals(v, f)
    centers = tri.mean(1)
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    owners = np.tile(np.arange(len(f)), 3)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    paired = starts[counts == 2]
    left, right = owners[order[paired]], owners[order[paired + 1]]
    distance = np.linalg.norm(centers[left] - centers[right], axis=1)
    graph = coo_matrix(
        (np.r_[distance, distance], (np.r_[left, right], np.r_[right, left])),
        shape=(len(f), len(f)),
    ).tocsr()
    rng = np.random.default_rng(0x175A)
    sampled = np.searchsorted(np.cumsum(area), (np.arange(3) + rng.random(3)) / 3 * area.sum())
    face_normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    face_normal /= np.linalg.norm(face_normal, axis=1)[:, None]
    difficulty = 1 - np.nan_to_num(np.sum(normals[f].mean(1) * face_normal, axis=1), nan=-1.0)
    boundary = owners[counts[inverse] == 1]
    difficult = np.argsort(-difficulty, kind="stable")
    proposed = [*sampled, *difficult[:2], *([int(boundary[0])] if len(boundary) else [])]
    chosen = list(dict.fromkeys(map(int, proposed)))
    supplemental = []
    for center in difficult:
        if len(chosen) >= 6:
            break
        if int(center) not in chosen:
            chosen.append(int(center))
            supplemental.append(int(center))
    regions = []
    for i, center in enumerate(chosen):
        dist = dijkstra(graph, indices=center, directed=False, limit=2000.0)
        regions.append(
            dict(
                id=f"roi-{i:02d}",
                center_face_row=center,
                center_xyz_um=centers[center].tolist(),
                face_rows=np.flatnonzero(dist <= 1000.0).tolist(),
                halo_face_rows=np.flatnonzero(dist <= 2000.0).tolist(),
            )
        )
    identity = dict(
        case=case["source_sha256"],
        seed=0x175A,
        radius_um=1000.0,
        halo_um=1000.0,
        selection_policy="six-distinct-source-face-centers-v2",
    )
    output = Path(output)
    with locked(output):
        if (output / "ROIS.json").exists():
            return verify(output, "ROIS.json", identity)["report"]
        np.save(output / "normals.npy", normals, allow_pickle=False)
        report = dict(
            identity=identity,
            regions=regions,
            normals=str((output / "normals.npy").absolute()),
            normals_sha256=file_sha(output / "normals.npy"),
            insufficient_distinct_regions=len(chosen) < 6,
            original_boundary_absent=not len(boundary),
            supplemental_geometric_centers=supplemental,
            supplemental_policy="unused high-difficulty source faces; not measured solver seams",
        )
        seal(output, "ROIS.json", identity, report, ["normals.npy"])
    return report


def render_references(case, rois, output, ct_config, guard):
    """Diagnostic tangent/orthogonal CT planes, never candidate UVs or coverage."""
    from bench_flattening import tool_identity

    validate_profile(case, ct_config["profile"])
    output = Path(output).absolute()
    source = read_json(Path(case["root"]) / "SOURCE.json")["identity"]["source"]
    for original in (Path(case["root"]).absolute(), Path(source["mesh"]).absolute()):
        if output.is_relative_to(original) or original.is_relative_to(output):
            raise ValueError("reference output must be separate from source geometry")
    a = case_arrays(case)
    v, f = a["world_um"], a["faces"]
    if (
        rois["identity"]["case"] != case["source_sha256"]
        or file_sha(rois["normals"]) != rois["normals_sha256"]
    ):
        raise ValueError("reference source ROI/normals changed")
    normals = np.load(rois["normals"], allow_pickle=False)
    pitch = ct_config["profile"]["pitch_um"]
    identity = dict(
        case=case["source_sha256"],
        source_snapshot=file_sha(Path(case["root"]) / "SOURCE.json"),
        rois=digest(rois),
        profile=digest(ct_config["profile"]),
        code=tool_identity(),
        width=256,
        pitch_um=pitch,
        offset_um=0.0,
        window=[0, 255],
    )
    row, col = np.indices((256, 256))
    dx, dy = (col - 127.5) * pitch, (127.5 - row) * pitch
    with locked(output):
        with CheckedCT(ct_config, output / "ct-ledger") as ct:
            if (output / "REFERENCES.json").exists():
                return verify(output, "REFERENCES.json", identity)["report"]
            regions, paths = [], []
            for roi in rois["regions"]:
                guard()
                face = roi["center_face_row"]
                if type(face) is not int or not 0 <= face < len(f):
                    raise ValueError("invalid reference source face")
                center = v[f[face]].mean(0)
                if not np.array_equal(center, roi["center_xyz_um"]):
                    raise ValueError("reference center differs from original source face")
                name = roi["id"]
                root = safe(output, name)
                binding = dict(reference=identity, roi_id=name, source_face_row=face)
                if (root / "REFERENCE.json").exists():
                    region = verify(root, "REFERENCE.json", binding)["report"]
                else:
                    root.mkdir(parents=True, exist_ok=True)
                    normal = normals_at(v, f, normals, center[None], np.array([face]))[0]
                    tangent = np.full((2, 3), np.nan)
                    if np.isfinite(normal).all():
                        axis = np.eye(3)[np.argmin(abs(normal))]
                        tangent[0] = np.cross(axis, normal)
                        tangent[0] /= np.linalg.norm(tangent[0])
                        tangent[1] = np.cross(normal, tangent[0])
                    basis = np.array(
                        [
                            tangent,
                            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        ]
                    )
                    xyz = (
                        center
                        + dx[None, :, :, None] * basis[:, None, None, 0]
                        + dy[None, :, :, None] * basis[:, None, None, 1]
                    )
                    raw = np.full((4, 256, 256), np.nan, np.float32)
                    support = np.zeros(raw.shape, bool)
                    plane_names = ["tangent", "xy", "xz", "yz"]
                    for i in range(4):
                        guard()
                        if np.isfinite(basis[i]).all():
                            values, valid = ct.sample(xyz[i].reshape(-1, 3))
                            raw[i], support[i] = values.reshape(256, 256), valid.reshape(256, 256)
                    guard()
                    np.savez_compressed(
                        root / "raw.npz", ct=raw, support=support, xyz_um=xyz, basis_xyz=basis
                    )
                    pngs = []
                    for i, plane in enumerate(plane_names):
                        pixels = np.rint(np.clip(np.nan_to_num(raw[i]), 0, 255)).astype("u1")
                        Image.fromarray(
                            np.dstack([pixels, pixels, pixels, support[i].astype("u1") * 255])
                        ).save(root / (plane + ".png"))
                        pngs.append(name + "/" + plane + ".png")
                    region = dict(
                        id=name,
                        source_face_row=face,
                        center_xyz_um=center.tolist(),
                        plane_names=plane_names,
                        pitch_um=pitch,
                        offset_um=0.0,
                        width=256,
                        height=256,
                        raw=name + "/raw.npz",
                        planes=pngs,
                        tangent_missing_reason=None
                        if np.isfinite(tangent).all()
                        else "undefined original normal",
                    )
                    seal(
                        root,
                        "REFERENCE.json",
                        binding,
                        region,
                        ["raw.npz"] + [Path(p).name for p in pngs],
                    )
                regions.append(region)
                paths.extend([name + "/REFERENCE.json", region["raw"], *region["planes"]])
            report = dict(
                case_id=case["id"],
                regions=regions,
                window=[0, 255],
                diagnostic_only=True,
                physical_acceptance=False,
                interpretation="Local planar CT controls, not a flattened surface or an area estimate",
            )
        paths.extend(str(p.relative_to(output)) for p in (output / "ct-ledger").glob("*.sqlite"))
        paths.append("ct-ledger/VOLUME.json")
        seal(output, "REFERENCES.json", identity, report, paths)
    return report


def _occupied(uv, faces, selected, pitch, low, top, tile_size):
    positions = set()
    for row in np.flatnonzero(selected):
        tri = uv[faces[row]]
        if not np.isfinite(tri).all():
            continue
        p = np.column_stack([(tri[:, 0] - low) / pitch, (top - tri[:, 1]) / pitch])
        lower = np.maximum(0, np.floor(p.min(0) / tile_size).astype(int))
        upper = np.maximum(0, np.floor(p.max(0) / tile_size).astype(int))
        if np.prod(upper - lower + 1) > 20000:
            raise MemoryError("single triangle exceeds occupied tile budget")
        for col in range(lower[0], upper[0] + 1):
            for r in range(lower[1], upper[1] + 1):
                positions.add((r * tile_size, col * tile_size))
        if len(positions) > 20000:
            raise MemoryError("occupied tile budget exceeded")
    return sorted(positions)


def render_case(case, mapped, rois, output, ct_config, guard):
    validate_profile(case, ct_config["profile"])
    a = case_arrays(case)
    v, f = a["world_um"], a["faces"]
    if (
        file_sha(rois["normals"]) != rois["normals_sha256"]
        or rois["identity"]["case"] != case["source_sha256"]
    ):
        raise ValueError("source ROI/normals changed")
    normals = np.load(rois["normals"], allow_pickle=False)
    profile = ct_config["profile"]
    identity = dict(
        case=case["source_sha256"],
        profile=case["ct_profile_sha256"],
        rois=digest(rois),
        map={
            k: hashlib.sha256(np.ascontiguousarray(mapped[k]).tobytes()).hexdigest()
            for k in ("world", "faces", "uv", "source_face_rows")
        },
    )
    output = Path(output)
    with locked(output):
        with CheckedCT(ct_config, output / "ct-ledger") as ct:
            if (output / "CT.json").exists():
                return verify(output, "CT.json", identity)["report"]
            uv, faces = mapped["uv"], mapped["faces"]
            tri = uv[faces]
            finite = np.isfinite(tri).all((1, 2))
            signed = cross2(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            eligible = finite & (signed > 0)
            lower, upper = np.nanmin(tri, axis=1), np.nanmax(tri, axis=1)
            low, top = float(np.nanmin(uv[:, 0])), float(np.nanmax(uv[:, 1]))
            offsets = list(map(float, profile["offsets_um"]))
            regions, paths = [], []
            specs = [
                dict(
                    id="overview",
                    face_rows=list(range(len(f))),
                    pitch=profile["overview_pitch_um"],
                    offsets=[0.0],
                )
            ]
            specs += [dict(r, pitch=profile["pitch_um"], offsets=offsets) for r in rois["regions"]]
            for spec in specs:
                guard()
                selected = np.isin(mapped["source_face_rows"], spec["face_rows"])
                pitch, records = spec["pitch"], []
                for row, col in _occupied(uv, faces, selected, pitch, low, top, 256):
                    guard()
                    name = f"{spec['id']}/{row:09d}-{col:09d}"
                    root = output / name
                    canvas = dict(
                        width=256,
                        height=256,
                        pitch=pitch,
                        u0=low + col * pitch,
                        v_top=top - row * pitch,
                    )
                    nearby = np.flatnonzero(
                        (upper[:, 0] >= canvas["u0"])
                        & (lower[:, 0] <= canvas["u0"] + 255 * pitch)
                        & (lower[:, 1] <= canvas["v_top"])
                        & (upper[:, 1] >= canvas["v_top"] - 255 * pitch)
                    )
                    tile_identity = dict(
                        render=identity, name=name, canvas=canvas, offsets=spec["offsets"]
                    )
                    if (root / "TILE.json").exists():
                        record = verify(root, "TILE.json", tile_identity)["report"]
                    else:
                        root.mkdir(parents=True, exist_ok=True)
                        captured = {}

                        def capture(xyz, mask, owner):
                            captured.update(xyz=xyz.copy(), mask=mask.copy(), owner=owner.copy())

                        raster_sheet(
                            mapped["world"],
                            faces[nearby],
                            uv,
                            eligible[nearby],
                            lambda xyz: (np.zeros(len(xyz)), np.ones(len(xyz), bool)),
                            canvas,
                            window=(0, 255),
                            guard=guard,
                            capture=capture,
                            xyz_tolerance=0.024,
                        )
                        owner = captured["owner"]
                        ok = owner >= 0
                        source_rows = np.full(owner.shape, -1, np.int64)
                        source_rows[ok] = mapped["source_face_rows"][nearby[owner[ok]]]
                        ok &= np.isin(source_rows, spec["face_rows"])
                        xyz = captured["xyz"]
                        nn = np.full_like(xyz, np.nan)
                        nn[ok] = normals_at(v, f, normals, xyz[ok], source_rows[ok])
                        raw = np.full((len(spec["offsets"]), 256, 256), np.nan, np.float32)
                        support = np.zeros_like(raw, bool)
                        for i, depth in enumerate(spec["offsets"]):
                            guard()
                            raw[i][ok], support[i][ok] = ct.sample(xyz[ok] + nn[ok] * depth)
                        np.savez_compressed(
                            root / "raw.npz",
                            ct=raw,
                            support=support,
                            xyz_um=xyz,
                            source_face_rows=source_rows,
                            geometric_mask=captured["mask"],
                        )
                        pngs = []
                        for i in range(len(raw)):
                            pixels = np.rint(np.clip(np.nan_to_num(raw[i]), 0, 255)).astype("u1")
                            rgba = np.dstack(
                                [pixels, pixels, pixels, support[i].astype("u1") * 255]
                            )
                            path = f"plane-{i:02d}.png"
                            Image.fromarray(rgba).save(root / path)
                            pngs.append(name + "/" + path)
                        record = dict(
                            raw=name + "/raw.npz",
                            planes=pngs,
                            canvas=canvas,
                            offsets_um=spec["offsets"],
                            source_rows=int(ok.sum()),
                        )
                        seal(
                            root,
                            "TILE.json",
                            tile_identity,
                            record,
                            ["raw.npz"] + [Path(p).name for p in pngs],
                        )
                    records.append(record)
                    paths.extend([name + "/TILE.json", record["raw"], *record["planes"]])
                regions.append(
                    dict(
                        id=spec["id"],
                        pitch_um=pitch,
                        offsets_um=spec["offsets"],
                        tiles=records,
                        missing=not records,
                    )
                )
            # Area quadrature is physical-source based and independent of raster packing.
            bary = quadrature_barycentric(1)
            observed = 0.0
            from flattening_metrics import measure

            measure(case, mapped, output / "geometry-coverage", guard)
            with np.load(
                output / "geometry-coverage/face-metrics.npz", allow_pickle=False
            ) as metrics:
                present = metrics["valid"]
            for first in range(0, len(f), 4096):
                guard()
                rows = np.arange(first, min(first + 4096, len(f)))
                tri3 = v[f[rows]]
                area = (
                    np.linalg.norm(
                        np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0]), axis=1
                    )
                    / 2
                )
                _, support = ct.sample(np.einsum("bc,fcx->fbx", bary, tri3).reshape(-1, 3))
                observed += float(np.sum(area * support.reshape(-1, 4).mean(1) * present[rows]))
            report = dict(
                regions=regions,
                offsets_um=offsets,
                window=[0, 255],
                sampling="verified trilinear",
                CT_observed_area_fraction_estimate=observed / (case["area_mm2"] * 1e6),
                area_quadrature_points_per_face=4,
                physical_acceptance=False,
            )
        paths.extend(str(p.relative_to(output)) for p in (output / "ct-ledger").glob("*.sqlite"))
        paths.append("ct-ledger/VOLUME.json")
        paths += ["geometry-coverage/METRICS.json", "geometry-coverage/face-metrics.npz"]
        seal(output, "CT.json", identity, report, paths)
    return report


def render_reference_run(run, output):
    """Versioned evaluation sidecar; never edits the source run's scientific files."""
    import time
    from bench_flattening import (
        guard_output,
        protect_inference,
        require_benchmark_envelope,
        tool_identity,
        validate_freeze,
    )

    run, output = Path(run).absolute(), Path(output).absolute()
    require_benchmark_envelope()
    queue = read_json(run / "QUEUE.json")
    if any(row["status"] == "RUNNING" for row in queue["rows"]):
        raise InterruptedError("pause the active benchmark sampler before reference downloads")
    if run.is_relative_to(output):
        raise ValueError("reference output must be separate from the source run")
    saved = read_json(run / "RUN.json")
    inventory = read_json(run / "INVENTORY.json")
    stage = queue.get("active_stage", "pilot")
    if stage == "holdout":
        validate_freeze(run)
    entries = [
        entry
        for entry in inventory["entries"]
        if entry["stage"] == stage or (stage == "main" and entry["stage"] == "pilot")
    ]
    for entry in entries:
        originals = [Path(entry["source"]["mesh"]).absolute()]
        if entry["status"] == "READY":
            originals.append(Path(entry["case"]["root"]).absolute())
        if any(output.is_relative_to(p) or p.is_relative_to(output) for p in originals):
            raise ValueError("reference output must be separate from source geometry")
    identity = dict(
        source_run_sha256=file_sha(run / "RUN.json"),
        inventory_sha256=file_sha(run / "INVENTORY.json"),
        stage=stage,
        code=tool_identity(),
    )
    resources = saved["config"]["resources"]
    with locked(run / "queue-lock"), locked(output):
        retained = (
            verify(output, "CONTROL-RUN.json", identity)["report"]
            if (output / "CONTROL-RUN.json").exists()
            else None
        )
        rows = []
        for entry in entries:
            if entry["status"] != "READY":
                rows.append(dict(source=entry["source"], status=entry["status"]))
                continue
            require_benchmark_envelope()
            case = entry["case"]
            case_output = safe(output, case["id"])
            started = time.monotonic()

            def guard():
                if (output / "STOP").exists():
                    raise InterruptedError("reference STOP requested")
                if time.monotonic() - started > resources["seconds"]:
                    raise TimeoutError("reference active time budget exhausted")
                guard_output(
                    output, reserve_bytes=resources["reserve_bytes"], pending_bytes=64 << 20
                )
                if saved["config"].get("protected_inference"):
                    protect_inference(saved["config"]["protected_inference"])
                if sum(
                    p.stat().st_size for p in case_output.rglob("*") if p.is_file()
                ) > resources.get("scratch_bytes", 20 << 30):
                    raise MemoryError("reference scratch budget exceeded; outputs retained")

            guard()
            rois = prepare_rois(case, case_output / "rois")
            config = dict(
                profile=case["ct_profile"],
                cache_dir=str(output / "ct-cache" / case["field_id"]),
                cache_bytes=resources["cache_bytes"],
                decoded_cache_bytes=resources["decoded_cache_bytes"],
                reserve_bytes=resources["reserve_bytes"],
            )
            render_references(case, rois, case_output / "references", config, guard)
            rows.append(
                dict(
                    case_id=case["id"],
                    field_id=case["field_id"],
                    status="COMPLETE",
                    references=case["id"] + "/references/REFERENCES.json",
                )
            )
        fields = sorted({row["field_id"] for row in rows if row["status"] == "COMPLETE"})
        report = dict(
            rows=rows,
            fields=fields,
            diagnostic_only=True,
            physical_acceptance=False,
            status="DIAGNOSTIC_CONTROLS_READY"
            if set(fields) == {"175a", "paris4"}
            and all(row["status"] == "COMPLETE" for row in rows)
            else "FINISHED_WITH_GAPS",
        )
        if retained is not None:
            if report != retained:
                raise ValueError("reference report identity changed")
            return retained
        files = [
            str(p.relative_to(output))
            for p in output.rglob("*")
            if p.is_file()
            and p.name not in (".lock", "STOP", "CONTROL-RUN.json")
            and "ct-cache" not in p.relative_to(output).parts
        ]
        seal(output, "CONTROL-RUN.json", identity, report, files)
    return report


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["references"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(render_reference_run(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
