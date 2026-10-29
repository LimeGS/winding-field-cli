#!/usr/bin/env python3
"""Bounded level-set mesh -> gated native flatten/PCA -> XYZ raster -> CT stack.

Usage (all paths local; output must not exist):
  winding-field surface-export --field phi.npy --metadata phi.npy.json \
    --level 1.25 --seed-xyz 10 20 30 --pitch .25 --output new_surface \
    --ct ct.npy --ct-metadata ct.json --depth-offsets -2 -1 0 1 2

Metadata: frame, origin_xyz, spacing_xyz, voxel_um, optional shape_zyx.
origin_xyz is the CENTER of array [0,0,0]; arrays are ZYX, coordinates XYZ.
voxel_um is micrometres per WORLD UNIT, not the physical solver-grid pitch.
Pitch and depth offsets are in these same world units.
CT metadata must name the SAME frame and voxel_um. Express the CT source's
origin and XYZ sampling pitches in that declared world frame;
the exporter never guesses a frame conversion from a name or array shape.

extract_surface(field, metadata, level, seed_xyz=None, *, step_size=1)
    -> vertices, faces, receipt
project_pca(vertices, faces) -> uv, receipt (FAIL is a normal gate outcome)
flatten_native(vertices, faces, app_run, output_dir, *, iters=200, timeout=120,
               expected_sha256=None) -> uv_or_None, receipt
rasterize_surface(vertices, faces, uv, pitch) -> points, normals, valid, receipt
sample_ct_stack(ct, metadata, points, normals, offsets, *, frame, voxel_um)
    -> float32 values, valid, receipt

--app-run PATH selects native Flatboi instead of PCA, with --flatten-iters 200
and --flatten-timeout 120 defaults. The exact command is:
  PATH flatboi inputmesh.obj 200 symmetric_dirichlet --tol=1e-6
Native UVs receive one global scale: sum(unique XYZ edge lengths) divided by
sum(unique UV edge lengths). Every existing UV/raster gate still applies.
The distortion gate is p95 <=2%; the maximum is reported, not thresholded.
--app-run-sha256 pins the LAUNCHER bytes, not the enclosing AppImage hash.
Only unchanged indexed XYZ vertices/oriented triangle topology are accepted;
per-corner seams and vertex reordering require an explicit external adapter.
--mesh-step 2 coarsens marching cubes in the original array coordinates and
reduces geometric resolution; default 1 preserves the original sampling.

Optional metadata['support'] is a bool array matching the field/CT. Unsupported
field stencils are excluded conservatively; holes are never filled. PCA is a
projection, not a general flattener. Full 3-D self-intersection, physical sheet
identity, writing face and ink remain UNMEASURED even when this export succeeds.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

MAX_FIELD_VOXELS = 64_000_000
MAX_PIXELS = 4_000_000
MAX_STACK_SAMPLES = 64_000_000


def _vector(value, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all() or (positive and (a <= 0).any()):
        raise ValueError(f"{name}: expected three finite" + (" positive" if positive else "") + " values")
    return a


def _metadata(meta, shape):
    if not isinstance(meta.get("frame"), str) or not meta["frame"]:
        raise ValueError("metadata requires an explicit frame")
    origin = _vector(meta.get("origin_xyz"), "origin_xyz")
    spacing = _vector(meta.get("spacing_xyz"), "spacing_xyz", True)
    um = float(meta.get("voxel_um", float("nan")))
    if not np.isfinite(um) or um <= 0:
        raise ValueError("voxel_um must specify positive micrometres per world unit")
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("expected a nonempty ZYX volume with at least two samples per axis")
    if "shape_zyx" in meta and list(meta["shape_zyx"]) != list(shape):
        raise ValueError("shape_zyx disagrees with array")
    return origin, spacing, um


def _support(array, meta, require_finite):
    support = np.asarray(meta.get("support", np.ones(array.shape, dtype=bool)))
    if support.shape != array.shape or support.dtype != np.bool_:
        raise ValueError("support must be a boolean array with the same ZYX shape")
    finite = np.isfinite(array)
    if require_finite and not finite[support].all():
        raise ValueError("field must be finite everywhere in its declared support")
    support = support & finite
    if not support.any():
        raise ValueError("no finite supported samples")
    return support


def mesh_geometry(vertices_xyz, faces, bounds_xyz=None):
    """Basic mesh checks only: boundary edges are valid, intersections unmeasured."""
    v, f = np.asarray(vertices_xyz), np.asarray(faces)
    r = dict(status="UNMEASURED", basic_status="FAIL", self_intersection="UNMEASURED")
    if (
        v.ndim != 2
        or v.shape[1:] != (3,)
        or not len(v)
        or not np.isfinite(v).all()
        or f.ndim != 2
        or f.shape[1:] != (3,)
        or not len(f)
        or not np.issubdtype(f.dtype, np.integer)
        or f.min() < 0
        or f.max() >= len(v)
    ):
        return {**r, "status": "FAIL", "reason": "invalid vertices or face indices"}
    tri = v[f]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    degenerate = int((area2 <= 1e-12).sum())
    duplicate = len(f) - len(np.unique(np.sort(f, axis=1), axis=0))
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    direction = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    nonmanifold = int((counts > 2).sum())
    inconsistent = int(((counts == 2) & (direction != 0)).sum())
    outside = False
    if bounds_xyz is not None:
        lo, hi = np.asarray(bounds_xyz, float)
        outside = bool(((v < lo - 1e-8) | (v > hi + 1e-8)).any())
    failed = degenerate or duplicate or nonmanifold or inconsistent or outside
    r.update(
        basic_status="FAIL" if failed else "PASS",
        status="FAIL" if failed else "UNMEASURED",
        vertices=len(v),
        faces=len(f),
        degenerate_faces=degenerate,
        duplicate_faces=duplicate,
        nonmanifold_edges=nonmanifold,
        inconsistent_edges=inconsistent,
        boundary_edges=int((counts == 1).sum()),
        outside_bounds=outside,
        bounds_xyz=[v.min(0).tolist(), v.max(0).tolist()],
    )
    return r


def _seed_distances(seed, triangles):
    """Exact point/triangle distance: projected interior or nearest edge."""
    a, b, c = triangles.transpose(1, 0, 2)
    ab, ac, ap = b - a, c - a, seed - a
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    aa, bb, cc = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    den = aa * cc - bb * bb
    u = (cc * dot(ap, ab) - bb * dot(ap, ac)) / den
    w = (aa * dot(ap, ac) - bb * dot(ap, ab)) / den
    interior = (u >= 0) & (w >= 0) & (u + w <= 1)
    distance = np.where(interior, np.linalg.norm(ap - u[:, None] * ab - w[:, None] * ac, axis=1), np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = np.clip(dot(seed - start, edge) / dot(edge, edge), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(seed - start - t[:, None] * edge, axis=1))
    return distance


def extract_surface(field, metadata, level, seed_xyz=None, *, step_size=1):
    """Extract the requested RAW field value, selecting one reindexed component."""
    field = np.asarray(field)
    origin, spacing, um = _metadata(metadata, field.shape)
    if type(step_size) is not int or not 1 <= step_size < min(field.shape):
        raise ValueError("step_size must be an integer >=1 and smaller than every field dimension")
    if field.size > MAX_FIELD_VOXELS:
        raise ValueError("bounded exporter: crop field to at most 64M voxels")
    if not np.isfinite(level):
        raise ValueError("level must be finite")
    support = _support(field, metadata, True)
    values = field[support]
    if not values.min() < level < values.max():
        raise ValueError("level must be strictly inside supported field range")
    work = np.where(support, field, level).astype(np.float32)
    if not np.isfinite(work).all():
        raise ValueError("field must remain finite as marching-cubes float32")
    vz, faces, _, _ = marching_cubes(work, level=float(level), step_size=step_size, allow_degenerate=False)
    before = len(faces)
    # A coarse cube must retain ALL intervening support, not only sampled corners.
    cube_ok = minimum_filter(support, size=step_size + 1, origin=-((step_size + 1) // 2), mode="constant", cval=False)
    last_sample = (np.array(field.shape) - 1) // step_size * step_size
    cell = np.minimum(np.floor(vz / step_size).astype(int) * step_size, last_sample - step_size)
    good_vertex = cube_ok[tuple(cell.T)]
    faces = faces[good_vertex[faces].all(1)]
    if not len(faces):
        raise ValueError("no triangles with fully supported interpolation stencils")
    ids, remap = np.unique(faces, return_inverse=True)
    vertices = origin + vz[ids, ::-1].astype(float) * spacing
    # Swapping ZYX -> XYZ reverses handedness; retain the original oriented surface.
    faces = remap.reshape(-1, 3)[:, ::-1]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2)
    nc, labels = connected_components(graph, directed=False)
    face_labels = labels[faces[:, 0]]
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    sizes = np.bincount(face_labels, weights=area, minlength=nc)
    seed_distance = None
    if seed_xyz is None:
        chosen = int(sizes.argmax())
    else:
        seed = _vector(seed_xyz, "seed_xyz")
        distances = _seed_distances(seed, tri)
        nearest = int(distances.argmin())
        chosen, seed_distance = int(face_labels[nearest]), float(distances[nearest])
    faces = faces[face_labels == chosen]
    ids, remap = np.unique(faces, return_inverse=True)
    vertices, faces = vertices[ids], remap.reshape(-1, 3)
    geometry = mesh_geometry(vertices, faces, [origin, origin + (np.array(field.shape[::-1]) - 1) * spacing])
    receipt = dict(
        level=float(level),
        level_units="raw_field_value",
        frame=metadata["frame"],
        origin_xyz=origin.tolist(),
        spacing_xyz=spacing.tolist(),
        shape_zyx=list(field.shape),
        step_size=step_size,
        geometry_resolution_reduced=step_size > 1,
        marching_step_xyz_world=(spacing * step_size).tolist(),
        marching_step_xyz_um=(spacing * step_size * um).tolist(),
        last_sample_index_zyx=last_sample.tolist(),
        voxel_um=um,
        sample_location="origin_is_array_000_center",
        selection="seed" if seed_xyz is not None else "largest",
        selection_rule="nearest_triangle" if seed_xyz is not None else "largest_surface_area",
        seed_xyz=None if seed_xyz is None else seed.tolist(),
        seed_distance=seed_distance,
        components=int(nc),
        selected_component=chosen,
        component_areas=sizes.tolist(),
        unsupported_voxels=int((~support).sum()),
        removed_unsupported_faces=before - len(tri),
        geometry=geometry,
        sheet_identity="UNMEASURED",
        writing_face="UNMEASURED",
        ink="NOT_MEASURED",
        normal_convention="exported triangle winding; not writing-face orientation",
    )
    return vertices, faces, receipt


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _uv_overlaps(tri, faces, tolerance, max_pairs):
    """Continuous triangle SAT, independent of raster pitch (touching edges allowed)."""
    centers = tri.mean(1)
    radii = np.linalg.norm(tri - centers[:, None], axis=2).max(1)
    max_radius = radii.max()
    tree = cKDTree(centers)
    overlaps = contacts = examined = 0
    for i, center in enumerate(centers):
        js = np.asarray(tree.query_ball_point(center, radii[i] + max_radius + tolerance), dtype=int)
        js = js[js > i]
        js = js[np.linalg.norm(centers[js] - center, axis=1) <= radii[i] + radii[js] + tolerance]
        examined += len(js)
        if examined > max_pairs:
            return None, None, examined
        if not len(js):
            continue
        a = np.broadcast_to(tri[i], (len(js), 3, 2))
        b = tri[js]
        edges = np.concatenate([np.roll(a, -1, axis=1) - a, np.roll(b, -1, axis=1) - b], axis=1)
        axes = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)
        axes /= np.linalg.norm(axes, axis=2)[..., None]
        pa = np.einsum("nvd,nad->nva", a - center, axes)
        pb = np.einsum("nvd,nad->nva", b - center, axes)
        interval = np.minimum(pa.max(1), pb.max(1)) - np.maximum(pa.min(1), pb.min(1))
        inside = (interval > tolerance).all(1)
        overlap_or_touch = (interval >= -tolerance).all(1)
        shared = (faces[i][None, :, None] == faces[js, None, :]).any(axis=(1, 2))
        overlaps += int(inside.sum())
        contacts += int((overlap_or_touch & ~inside & ~shared).sum())
    return overlaps, contacts, examined


def validate_uv(vertices_xyz, faces, uv, *, max_distortion=0.02, max_pairs=2_000_000):
    """Gate triangle orientation, metric distortion and continuous UV overlap."""
    v, f, uv = np.asarray(vertices_xyz, float), np.asarray(faces), np.asarray(uv, float)
    r = dict(status="FAIL", overlap_pairs=None, nonlocal_contacts=None, reasons=[])
    if not np.isfinite(max_distortion) or not 0 <= max_distortion < 1 or max_pairs < 1:
        raise ValueError("invalid UV gate limits")
    geometry = mesh_geometry(v, f)
    if geometry["basic_status"] != "PASS":
        return {**r, "reasons": ["basic_mesh_failed"]}
    if uv.shape != (len(v), 2) or not np.isfinite(uv).all():
        return {**r, "reasons": ["invalid_uv"]}
    tri = uv[f]
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    signed = _cross2(a, b)
    eps = max(float(np.ptp(uv, axis=0).max()), 1.0) * 1e-10
    degenerate = int((np.abs(signed) <= eps**2).sum())
    inverted = min(int((signed > eps**2).sum()), int((signed < -(eps**2)).sum()))
    r.update(
        degenerate_triangles=degenerate,
        inverted_triangles=inverted,
        p95_distortion_limit=float(max_distortion),
        tolerance_world=eps,
        distortion_gate_statistic="p95",
        policy_version="uv-p95-v2",
    )
    if degenerate:
        return {**r, "reasons": ["degenerate_uv_triangles"]}
    if inverted:
        r["reasons"].append("inverted_uv_triangles")
    matrix = np.stack([a, b], axis=-1)
    xyz_edges = np.stack([v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]], axis=-1)
    singular = np.linalg.svd(xyz_edges @ np.linalg.inv(matrix), compute_uv=False)
    distortion = np.max(np.abs(singular - 1), axis=1)
    r.update(max_distortion=float(distortion.max()), p95_distortion=float(np.percentile(distortion, 95)))
    # The accepted policy is p95, not the maximum; retain the tail for disclosure.
    if r["p95_distortion"] > max_distortion + 1e-10:
        r["reasons"].append("metric_distortion")
    pairs, contacts, examined = _uv_overlaps(tri, f, eps, max_pairs)
    r.update(
        overlap_pairs=pairs,
        nonlocal_contacts=contacts,
        candidate_pairs=examined,
        overlap_status="UNMEASURED" if pairs is None else "MEASURED",
    )
    if pairs is None:
        r["reasons"].append("overlap_budget_exceeded")
    elif pairs or contacts:
        r["reasons"].append("uv_overlap_or_nonlocal_contact")
    r["status"] = "FAIL" if r["reasons"] else "PASS"
    return r


def project_pca(vertices_xyz, faces, *, max_distortion=0.02, max_pairs=2_000_000):
    """Orthogonal PCA chart in world units, NOT an optimized flattening."""
    v = np.asarray(vertices_xyz, float)
    if mesh_geometry(v, faces)["basic_status"] != "PASS":
        raise ValueError("basic mesh failed before projection")
    center = v.mean(0)
    _, _, axes = np.linalg.svd(v - center, full_matrices=False)
    basis = axes[:2].copy()
    for axis in basis:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    tri = v[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(0)
    if np.dot(np.cross(*basis), normal) < 0:
        basis[1] *= -1
    uv = (v - center) @ basis.T
    shift = uv.min(0)
    uv -= shift
    gate = validate_uv(v, faces, uv, max_distortion=max_distortion, max_pairs=max_pairs)
    gate.update(
        method="pca_projection",
        center_xyz=center.tolist(),
        basis_uv_xyz=basis.tolist(),
        uv_shift=shift.tolist(),
        optimized_flattening=False,
        failure_action="crop a nearly planar chart or use a validated external Flatboi result",
    )
    return uv, gate


def read_flatboi_uv(path, vertices_xyz, faces, *, vertex_atol=1e-6):
    """Bind OBJ UVs only to unchanged indexed XYZ and oriented triangle topology.

    Supports f v/vt[/vn], or one vt per vertex with bare f v[//vn]. Face row
    order/cyclic rotations may change. Exact duplicate vt values are aliases;
    vertex reordering and differing per-corner UV seams are rejected.
    """
    v, f = np.asarray(vertices_xyz, float), np.asarray(faces)
    if not np.isfinite(vertex_atol) or vertex_atol < 0 or mesh_geometry(v, f)["basic_status"] != "PASS":
        raise ValueError("invalid original mesh or vertex_atol")
    positions, texcoords, obj_faces, texture_faces = [], [], [], []

    def index(token, count):
        value = int(token)
        if value == 0:
            raise ValueError("OBJ indices cannot be zero")
        return value - 1 if value > 0 else count + value

    with open(path) as stream:
        for line_number, line in enumerate(stream, 1):
            parts = line.split("#", 1)[0].split()
            if not parts:
                continue
            kind, values = parts[0], parts[1:]
            try:
                if kind == "v":
                    if len(values) != 3:
                        raise ValueError("only XYZ vertex records are supported")
                    positions.append([float(x) for x in values])
                elif kind == "vt":
                    if len(values) not in (2, 3) or (len(values) == 3 and float(values[2]) != 0):
                        raise ValueError("expected two-dimensional vt (optional zero third coordinate)")
                    texcoords.append([float(x) for x in values[:2]])
                elif kind == "f":
                    if len(values) != 3:
                        raise ValueError("only triangular OBJ faces are supported")
                    corners = [x.split("/") for x in values]
                    if any(len(x) > 3 or not x[0] for x in corners):
                        raise ValueError("unsupported OBJ face syntax")
                    obj_faces.append([index(x[0], len(positions)) for x in corners])
                    texture_faces.append(
                        [index(x[1], len(texcoords)) if len(x) > 1 and x[1] else None for x in corners]
                    )
            except (ValueError, IndexError) as error:
                raise ValueError(f"OBJ line {line_number}: {error}") from error
    positions, textures = np.asarray(positions, float), np.asarray(texcoords, float)
    of = np.asarray(obj_faces, dtype=np.int64)
    if (
        positions.shape != v.shape
        or not np.isfinite(positions).all()
        or not np.allclose(positions, v, atol=vertex_atol, rtol=0)
    ):
        raise ValueError("OBJ XYZ vertices changed/reordered; cannot bind UVs to original mesh")
    if of.shape != f.shape or not len(of) or of.min() < 0 or of.max() >= len(v):
        raise ValueError("OBJ face indices/count disagree with original mesh")

    def topology(a):
        rotations = np.take_along_axis(a, (np.argmin(a, axis=1)[:, None] + np.arange(3)) % 3, axis=1)
        return sorted(map(tuple, rotations.tolist()))

    if topology(of) != topology(f):
        raise ValueError("OBJ oriented topology differs from original mesh")
    if textures.ndim != 2 or textures.shape[1] != 2 or not len(textures) or not np.isfinite(textures).all():
        raise ValueError("OBJ is missing finite two-dimensional UVs")
    missing = [i is None for row in texture_faces for i in row]
    exact_aliases = 0
    if all(missing):
        if len(textures) != len(v):
            raise ValueError("unindexed UVs require exactly one vt per vertex")
        uv, mode = textures.copy(), "vt_per_vertex"
    else:
        if any(missing):
            raise ValueError("mixed indexed/unindexed UVs are unsupported")
        tf = np.asarray(texture_faces, dtype=np.int64)
        if tf.min() < 0 or tf.max() >= len(textures):
            raise ValueError("OBJ texture index out of bounds")
        bindings = np.full(len(v), -1, dtype=np.int64)
        for face, tface in zip(of, tf):
            for vi, ti in zip(face, tface):
                if bindings[vi] not in (-1, ti):
                    if not np.array_equal(textures[bindings[vi]], textures[ti]):
                        raise ValueError("unsupported UV seam: one XYZ vertex uses different per-corner UVs")
                    exact_aliases += 1
                bindings[vi] = ti
        if (bindings < 0).any():
            raise ValueError("unreferenced original vertices have no UV binding")
        uv, mode = textures[bindings], "indexed_v_vt"
    return uv, dict(
        status="PASS",
        mode=mode,
        vertex_mapping="identity",
        exact_uv_alias_occurrences=exact_aliases,
        oriented_topology="equal_up_to_face_order_and_cyclic_rotation",
        vertex_atol_world=float(vertex_atol),
        maximum_xyz_difference=float(np.abs(positions - v).max()),
    )


def scale_uv_to_world(vertices_xyz, faces, uv):
    """One global scale from unique edge lengths; no per-axis/nonlinear fitting."""
    v, f, uv = np.asarray(vertices_xyz, float), np.asarray(faces), np.asarray(uv, float)
    if mesh_geometry(v, f)["basic_status"] != "PASS" or uv.shape != (len(v), 2) or not np.isfinite(uv).all():
        raise ValueError("invalid mesh or UVs for global scaling")
    edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    xyz_lengths = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1)
    uv_lengths = np.linalg.norm(uv[edges[:, 1]] - uv[edges[:, 0]], axis=1)
    if (uv_lengths <= 0).any() or not np.isfinite(uv_lengths).all():
        raise ValueError("zero/nonfinite UV edge prevents global scaling")
    scale = float(xyz_lengths.sum() / uv_lengths.sum())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid global UV scale")
    shift = uv.min(0)
    scaled = (uv - shift) * scale
    if not np.isfinite(scaled).all():
        raise ValueError("nonfinite globally scaled UV")
    return scaled, dict(
        world_units_per_uv_unit=scale,
        uv_origin_removed=shift.tolist(),
        definition="sum_unique_xyz_edge_lengths / sum_unique_uv_edge_lengths",
        unique_edges=len(edges),
        xyz_edge_length_sum=float(xyz_lengths.sum()),
        uv_edge_length_sum=float(uv_lengths.sum()),
        anisotropic_or_nonlinear_fit=False,
    )


def flatten_native(vertices_xyz, faces, app_run, output_dir, *, iters=200, timeout=120, expected_sha256=None):
    """Run local Flatboi in a new directory; return UV/receipt, including failures."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(exist_ok=False)
    app_run = Path(app_run).resolve()
    mesh = output_dir / "inputmesh.obj"
    flat = output_dir / "inputmesh_flatboi.obj"
    command = [str(app_run), "flatboi", str(mesh), str(iters), "symmetric_dirichlet", "--tol=1e-6"]
    receipt = dict(
        method="native_flatboi",
        status="FAIL",
        command=command,
        returncode=None,
        timed_out=False,
        timeout_seconds=float(timeout),
        iterations=iters,
        energy="symmetric_dirichlet",
        relative_energy_tolerance=1e-6,
        app_run_sha256=None,
        expected_app_run_sha256=expected_sha256,
        self_intersection="UNMEASURED",
        reasons=[],
        artifacts={},
    )
    uv = None
    start = time.monotonic()
    try:
        if type(iters) is not int or iters < 1 or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("flatten iters/timeout must be positive")
        if mesh_geometry(vertices_xyz, faces)["basic_status"] != "PASS":
            raise ValueError("basic mesh failed before native flattening")
        if not app_run.is_file() or not os.access(app_run, os.X_OK):
            receipt["status"] = "UNAVAILABLE"
            raise ValueError("AppRun unavailable or not executable")
        receipt["app_run_sha256"] = _sha(app_run)
        if expected_sha256 is not None and receipt["app_run_sha256"] != expected_sha256:
            raise ValueError("AppRun SHA256 mismatch (this pins the launcher, not the AppImage)")
        with mesh.open("x") as stream:
            np.savetxt(stream, vertices_xyz, fmt="v %.17g %.17g %.17g")
            np.savetxt(stream, np.asarray(faces) + 1, fmt="f %d %d %d")
        receipt["input_mesh_sha256"] = _sha(mesh)
        with (output_dir / "stdout.log").open("xb") as stdout, (output_dir / "stderr.log").open("xb") as stderr:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=output_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except OSError:
                receipt["status"] = "UNAVAILABLE"
                raise
            try:
                receipt["returncode"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                receipt["timed_out"], receipt["status"] = True, "TIMEOUT"
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                receipt["returncode"] = process.wait()
                raise ValueError("native flattening timed out; owned process group terminated")
        if receipt["returncode"] != 0:
            raise ValueError(f"Flatboi exited {receipt['returncode']}")
        if _sha(mesh) != receipt["input_mesh_sha256"] or _sha(app_run) != receipt["app_run_sha256"]:
            raise ValueError("native input mesh or AppRun changed during execution")
        if not flat.is_file() or flat.is_symlink():
            raise ValueError("missing regular inputmesh_flatboi.obj output")
        receipt["output_obj_sha256"] = _sha(flat)
        raw_uv, receipt["binding"] = read_flatboi_uv(flat, vertices_xyz, faces)
        uv, receipt["scaling"] = scale_uv_to_world(vertices_xyz, faces, raw_uv)
        receipt["uv_gate"] = validate_uv(vertices_xyz, faces, uv)
        receipt["status"] = receipt["uv_gate"]["status"]
        receipt["reasons"] = receipt["uv_gate"]["reasons"]
    except (ValueError, OSError, RuntimeError) as error:
        uv = None
        receipt["reasons"].append(f"{type(error).__name__}: {error}")
    finally:
        receipt["seconds"] = time.monotonic() - start
        receipt["artifacts"] = {
            p.name: {"sha256": _sha(p), "bytes": p.stat().st_size} for p in sorted(output_dir.iterdir()) if p.is_file()
        }
        (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return uv, receipt


def rasterize_surface(
    vertices_xyz, faces, uv, pitch, *, max_distortion=0.02, max_pixels=MAX_PIXELS, vertex_normals_xyz=None
):
    """Barycentric XYZ samples at uv_min + (column,row)*pitch; holes stay NaN."""
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("pitch must be positive in world-coordinate units")
    v, f, uv = np.asarray(vertices_xyz, float), np.asarray(faces), np.asarray(uv, float)
    gate = validate_uv(v, f, uv, max_distortion=max_distortion)
    if gate["status"] != "PASS":
        raise ValueError("UV gate failed: " + ", ".join(gate["reasons"]))
    low = uv.min(0)
    with np.errstate(over="ignore"):
        dimensions = np.ceil(np.ptp(uv, axis=0) / pitch) + 1
    if not np.isfinite(dimensions).all() or (dimensions > max_pixels).any() or np.prod(dimensions) > max_pixels:
        raise ValueError("bounded raster exceeded max_pixels; crop or increase pitch")
    wh = dimensions.astype(np.int64)
    width, height = map(int, wh)
    points = np.full((height, width, 3), np.nan)
    normals = np.full_like(points, np.nan)
    owner = np.full((height, width), -1, dtype=np.int64)
    if vertex_normals_xyz is None:
        vertex_normals = np.zeros_like(v)
        raw = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        for k in range(3):
            np.add.at(vertex_normals, f[:, k], raw)
    else:
        vertex_normals = np.array(vertex_normals_xyz, dtype=float, copy=True)
        if vertex_normals.shape != v.shape or not np.isfinite(vertex_normals).all():
            raise ValueError("invalid supplied vertex normals")
    lengths = np.linalg.norm(vertex_normals, axis=1)
    if (lengths <= 1e-12).any():
        raise ValueError("ambiguous vertex normal")
    vertex_normals /= lengths[:, None]
    for index, face in enumerate(f):
        t = uv[face]
        start = np.maximum(0, np.ceil((t.min(0) - low) / pitch - 1e-9).astype(int))
        stop = np.minimum(wh - 1, np.floor((t.max(0) - low) / pitch + 1e-9).astype(int))
        if (stop < start).any():
            continue
        cols, rows = np.meshgrid(np.arange(start[0], stop[0] + 1), np.arange(start[1], stop[1] + 1))
        q = low + np.stack([cols, rows], axis=-1) * pitch
        e1, e2 = t[1] - t[0], t[2] - t[0]
        den = _cross2(e1, e2)
        b1, b2 = _cross2(q - t[0], e2) / den, _cross2(e1, q - t[0]) / den
        weights = np.stack([1 - b1 - b2, b1, b2], axis=-1)
        inside = (weights >= -1e-9).all(-1)
        rr, cc, ww = rows[inside], cols[inside], weights[inside]
        xyz, n = ww @ v[face], ww @ vertex_normals[face]
        n /= np.linalg.norm(n, axis=1)[:, None]
        used = owner[rr, cc] >= 0
        if used.any() and (
            not np.allclose(points[rr[used], cc[used]], xyz[used], atol=1e-8, rtol=0)
            or not np.allclose(normals[rr[used], cc[used]], n[used], atol=1e-8, rtol=0)
        ):
            raise ValueError("UV raster multimapping: shared pixel has different XYZ/normal")
        rr, cc = rr[~used], cc[~used]
        points[rr, cc], normals[rr, cc], owner[rr, cc] = xyz[~used], n[~used], index
    valid = owner >= 0
    if not valid.any():
        raise ValueError("no covered raster samples at this pitch")
    receipt = dict(
        status="PASS",
        shape_hw=[height, width],
        pitch_world=float(pitch),
        uv_origin=low.tolist(),
        uv_axes="columns=u, rows=v",
        sample_location="grid_nodes",
        covered_pixels=int(valid.sum()),
        uncovered_pixels=int((~valid).sum()),
        coverage_fraction=float(valid.mean()),
        multimapped_pixels=0,
        normal_source="global_mesh" if vertex_normals_xyz is not None else "chart_mesh",
        interpolation="triangle_barycentric",
        hole_filling=False,
        uv_gate=gate,
    )
    return points, normals, valid, receipt


def sample_ct_stack(ct, metadata, points_xyz, normals_xyz, depth_offsets, *, frame, voxel_um):
    """Trilinear CT at points + offset*unit_normal; invalid stencils stay NaN."""
    ct = np.asarray(ct)
    origin, spacing, um = _metadata(metadata, ct.shape)
    if metadata["frame"] != frame:
        raise ValueError("CT and surface frame differ; convert XYZ explicitly")
    if not np.isfinite(voxel_um) or not np.isclose(um, voxel_um, rtol=1e-12, atol=0):
        raise ValueError("CT and surface voxel_um differ; convert world units explicitly")
    points, normals = np.asarray(points_xyz, float), np.asarray(normals_xyz, float)
    offsets = np.asarray(depth_offsets, float)
    if (
        points.ndim != 3
        or points.shape[-1] != 3
        or normals.shape != points.shape
        or offsets.ndim != 1
        or not len(offsets)
        or not np.isfinite(offsets).all()
    ):
        raise ValueError("expected H,W,3 point/normal maps and finite depth offsets")
    if len(offsets) * points.shape[0] * points.shape[1] > MAX_STACK_SAMPLES:
        raise ValueError("bounded stack exceeded 64M samples; crop, coarsen pitch or reduce depths")
    valid_surface = np.isfinite(points).all(-1)
    norm = np.linalg.norm(normals, axis=-1)
    if not np.isfinite(norm[valid_surface]).all() or not np.allclose(norm[valid_surface], 1, atol=1e-7, rtol=0):
        raise ValueError("covered surface normals must be finite unit vectors")
    support = metadata.get("support")
    if support is not None:
        support = np.asarray(support)
        if support.shape != ct.shape or support.dtype != np.bool_:
            raise ValueError("CT support must be a boolean array with the same ZYX shape")
    stack = np.full((len(offsets),) + points.shape[:2], np.nan, dtype=np.float32)
    mask = np.zeros(stack.shape, bool)
    for layer, offset in enumerate(offsets):
        grid = (points + offset * normals - origin) / spacing
        inside = valid_surface & (grid >= 0).all(-1) & (grid <= np.array(ct.shape[::-1]) - 1).all(-1)
        rows, cols = np.where(inside)
        g = grid[inside]
        lower, upper = np.floor(g).astype(int), np.ceil(g).astype(int)
        fraction = g - lower
        good = np.ones(len(g), bool)
        sampled = np.zeros(len(g), float)
        # Eight gathers avoid a full float64 copy of a potentially mmap-backed CT.
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    bits = np.array([dx, dy, dz], bool)
                    xyz = np.where(bits, upper, lower)
                    weight = np.prod(np.where(bits, fraction, 1 - fraction), axis=1)
                    corner = ct[tuple(xyz[:, ::-1].T)].astype(float)
                    observed = np.isfinite(corner)
                    if support is not None:
                        observed &= support[tuple(xyz[:, ::-1].T)]
                    good &= observed | (weight == 0)
                    sampled += np.where(observed, corner, 0) * weight
        good &= np.isfinite(sampled) & (np.abs(sampled) <= np.finfo(np.float32).max)
        rr, cc = rows[good], cols[good]
        stack[layer, rr, cc] = sampled[good]
        mask[layer, rr, cc] = True
    receipt = dict(
        status="SAMPLED" if mask.any() else "NO_COVERAGE",
        frame=frame,
        voxel_um=um,
        ct_origin_xyz=origin.tolist(),
        ct_spacing_xyz=spacing.tolist(),
        depth_offsets_world=offsets.tolist(),
        depth_offsets_um=(offsets * um).tolist(),
        shape_lhw=list(stack.shape),
        valid_samples_per_layer=mask.sum(axis=(1, 2)).tolist(),
        interpolation="trilinear",
        out_of_bounds="NaN",
        hole_filling=False,
        normal_direction="exported triangle winding; writing face unknown",
        ink="NOT_MEASURED",
    )
    return stack, mask, receipt


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _preview(path, values, valid, window=None):
    from PIL import Image, PngImagePlugin

    lo, hi = np.percentile(values[valid], [1, 99]) if window is None else window
    if not np.isfinite([lo, hi]).all() or (window is not None and hi <= lo):
        raise ValueError("invalid declared CT display window")
    gray = np.zeros(values.shape, np.uint8)
    gray[valid] = np.rint(255 * np.clip((values[valid] - lo) / max(hi - lo, 1e-12), 0, 1)).astype(np.uint8)
    rgba = np.dstack([gray, gray, gray, valid.astype(np.uint8) * 255])
    text = PngImagePlugin.PngInfo()
    text.add_text("Description", "CT texture preview, not ink; transparent pixels are unobserved")
    Image.fromarray(rgba).save(path, pnginfo=text)
    return dict(
        window=[float(lo), float(hi)],
        normalization="p01_p99_preview_only" if window is None else "declared_window",
        invalid="transparent",
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--level", type=float, required=True, help="actual RAW field isovalue, no automatic selection")
    p.add_argument("--seed-xyz", "--seedXYZ", type=float, nargs=3)
    p.add_argument("--output", type=Path, required=True, help="new directory, parent must exist")
    p.add_argument("--voxel-um", type=float, help="explicit metadata voxel_um override (um per world unit)")
    p.add_argument("--support", type=Path, help="optional bool ZYX field-support .npy")
    p.add_argument("--mesh-step", type=int, default=1, help="marching-cubes stride; >1 reduces geometry resolution")
    p.add_argument("--app-run", type=Path, help="local VC3D AppRun; omitted uses gated PCA projection")
    p.add_argument("--app-run-sha256", help="optional expected SHA256 of AppRun itself, not the AppImage")
    p.add_argument("--flatten-iters", type=int, default=200)
    p.add_argument("--flatten-timeout", type=float, default=120, help="native subprocess timeout in seconds")
    p.add_argument("--ct", type=Path, help="optional local CT ZYX .npy")
    p.add_argument("--ct-metadata", type=Path, help="CT frame/origin/spacing/voxel_um JSON")
    p.add_argument("--ct-support", type=Path, help="optional bool ZYX CT-support .npy")
    p.add_argument("--pitch", type=float, required=True, help="UV grid pitch in WORLD units")
    p.add_argument("--depth-offsets", nargs="+", type=float, default=[0.0], help="normal offsets in WORLD units")
    args = p.parse_args(argv)
    if bool(args.ct) != bool(args.ct_metadata) or (args.ct_support and not args.ct):
        p.error("--ct and --ct-metadata are required together; --ct-support needs CT")
    if args.app_run_sha256 and not args.app_run:
        p.error("--app-run-sha256 requires --app-run")
    args.output.mkdir(exist_ok=False)
    receipt = dict(
        schema="winding.surface-export.v1",
        status="FAILED",
        argv=list(sys.argv[1:] if argv is None else argv),
        source_sha256=_sha(__file__),
        self_intersection="UNMEASURED",
        sheet_identity="UNMEASURED",
        writing_face="UNMEASURED",
        ink="NOT_MEASURED",
        artifacts={},
    )
    code = 2
    try:
        paths = [args.field, args.metadata, args.support, args.ct, args.ct_metadata, args.ct_support]
        receipt["inputs_sha256"] = {str(path): _sha(path) for path in paths if path is not None}
        meta = json.loads(args.metadata.read_text())
        if args.voxel_um is not None:
            meta["voxel_um"] = args.voxel_um
        if args.support:
            meta["support"] = np.load(args.support, allow_pickle=False)
        field = np.load(args.field, mmap_mode="r", allow_pickle=False)
        vertices, faces, extraction = extract_surface(field, meta, args.level, args.seed_xyz, step_size=args.mesh_step)
        receipt["extraction"] = extraction
        np.savez(args.output / "surface.npz", vertices_xyz=vertices, faces=faces)
        with (args.output / "surface.obj").open("x") as f:
            f.write(f"# XYZ world frame {meta['frame']}; {meta['voxel_um']} um/world-unit\n")
            np.savetxt(f, vertices, fmt="v %.17g %.17g %.17g")
            np.savetxt(f, faces + 1, fmt="f %d %d %d")
        if args.app_run:
            uv, projection = flatten_native(
                vertices,
                faces,
                args.app_run,
                args.output / "native_flatten",
                iters=args.flatten_iters,
                timeout=args.flatten_timeout,
                expected_sha256=args.app_run_sha256,
            )
        else:
            uv, projection = project_pca(vertices, faces)
        receipt["projection"] = projection
        if uv is not None:
            np.savez(args.output / "uv.npz", uv=uv, faces=faces)
        if projection["status"] != "PASS":
            raise ValueError("UV generation failed: " + ", ".join(projection["reasons"]))
        points, normals, valid, raster = rasterize_surface(vertices, faces, uv, args.pitch)
        raster["pitch_um"] = [args.pitch * meta["voxel_um"]] * 2
        receipt["raster"] = raster
        np.savez(
            args.output / "grid.npz",
            points_xyz=points,
            normals_xyz=normals,
            valid=valid,
            uv_origin=raster["uv_origin"],
            pitch=args.pitch,
            pitch_um=raster["pitch_um"],
        )
        if args.ct:
            cm = json.loads(args.ct_metadata.read_text())
            if args.ct_support:
                cm["support"] = np.load(args.ct_support, allow_pickle=False)
            ct = np.load(args.ct, mmap_mode="r", allow_pickle=False)
            stack, coverage, sampled = sample_ct_stack(
                ct, cm, points, normals, args.depth_offsets, frame=meta["frame"], voxel_um=meta["voxel_um"]
            )
            receipt["ct"] = sampled
            np.savez(
                args.output / "ct_stack.npz",
                values=stack,
                valid=coverage,
                depth_offsets=np.asarray(args.depth_offsets),
                depth_offsets_um=sampled["depth_offsets_um"],
            )
            if not coverage.any():
                raise ValueError("CT stack has no covered samples")
            layer = int(np.argmin(np.abs(args.depth_offsets)))
            if coverage[layer].any():
                receipt["preview"] = {
                    **_preview(args.output / "ct_preview.png", stack[layer], coverage[layer], cm.get("display_window")),
                    "layer": layer,
                    "depth_offset_world": args.depth_offsets[layer],
                }
        if receipt["inputs_sha256"] != {str(path): _sha(path) for path in paths if path is not None}:
            raise ValueError("input bytes changed during export")
        receipt["status"], code = "EXPORTED_UNCERTIFIED", 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, MemoryError) as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        receipt["artifacts"] = {
            str(f.relative_to(args.output)): {"sha256": _sha(f), "bytes": f.stat().st_size}
            for f in sorted(args.output.rglob("*"))
            if f.is_file()
        }
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"status": receipt["status"], "receipt": str(args.output / "receipt.json"), "error": receipt.get("error")}
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
