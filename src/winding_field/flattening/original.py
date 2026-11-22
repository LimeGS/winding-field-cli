"""Original whole-code-05 global LSCM and raster, independent of field resolution.

Port of the project's frozen 2026-10-24 recipe (see provenance/original-recipe.json).
Coordinates, UV pitch and XYZ tolerance use the caller's declared world unit.
"""

import math
from itertools import islice, product

import numpy as np

from .lscm import lscm
from .topology import duplicate_faces, edge_groups


def sample_points(xyz, shape, chunks, chunk, *, prefetch=None):
    """Frozen positive-weight trilinear accumulation; chunk returns array or None."""
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) > 100000:
        raise ValueError("expected <=100000 XYZ points")
    shape, chunks = np.asarray(shape), np.asarray(chunks)
    grid = (shape + chunks - 1) // chunks
    points = xyz[:, ::-1].astype(np.float64)
    valid = np.isfinite(points).all(1) & (points >= 0).all(1) & (points <= shape - 1).all(1)
    values = np.zeros(len(points), np.float64)
    ids = np.flatnonzero(valid)
    if len(ids):
        lower = np.floor(points[ids]).astype(np.int64)
        fraction = points[ids] - lower
        jobs = []
        for corner in product((0, 1), repeat=3):
            corner = np.array(corner)
            weight = np.prod(np.where(corner, fraction, 1 - fraction), axis=1)
            used = weight > 0
            node = lower[used] + corner
            ci, local = node // chunks, node % chunks
            keys = (ci[:, 0] * grid[1] + ci[:, 1]) * grid[2] + ci[:, 2]
            offsets = (local[:, 0] * chunks[1] + local[:, 1]) * chunks[2] + local[:, 2]
            jobs.append((keys, ids[used], offsets, weight[used]))
        keys, targets, offsets, weights = [np.concatenate([j[i] for j in jobs]) for i in range(4)]
        order = np.argsort(keys, kind="stable")
        keys, targets, offsets, weights = [a[order] for a in (keys, targets, offsets, weights)]
        cuts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(keys)]
        for position, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            if prefetch is not None and position % 4 == 0:
                batch = [
                    int(keys[start]) for start in cuts[position : min(position + 4, len(cuts) - 1)]
                ]
                prefetch(
                    [
                        (
                            k // int(grid[1] * grid[2]),
                            k // int(grid[2]) % int(grid[1]),
                            k % int(grid[2]),
                        )
                        for k in batch
                    ]
                )
            key = int(keys[a])
            index = (
                key // int(grid[1] * grid[2]),
                key // int(grid[2]) % int(grid[1]),
                key % int(grid[2]),
            )
            data = chunk(index)
            if data is None:
                valid[targets[a:b]] = False
            else:
                if data.shape != tuple(chunks):
                    raise ValueError("malformed CT chunk")
                np.add.at(values, targets[a:b], data.ravel()[offsets[a:b]] * weights[a:b])
    values[~valid] = np.nan
    return values.astype(np.float32), valid


def face_quality(vertices, faces, uv):
    v, f, uv = np.asarray(vertices, float), np.asarray(faces, np.int64), np.asarray(uv, float)
    area = np.empty(len(f))
    error = np.full(len(f), np.inf)
    positive = np.empty(len(f), bool)
    display = np.empty(len(f), bool)
    scientific = np.empty(len(f), bool)
    # Keep full ordered scalar arrays for the original global sums and percentile.
    for start in range(0, len(f), 4096):
        sl = slice(start, start + 4096)
        block = f[sl]
        edges = np.stack(
            [v[block[:, 1]] - v[block[:, 0]], v[block[:, 2]] - v[block[:, 0]]], axis=-1
        )
        t = uv[block]
        matrix = np.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]], axis=-1)
        det = np.linalg.det(matrix)
        valid = np.isfinite(matrix).all((1, 2)) & (det > 0)
        singular = np.full((len(block), 2), np.nan)
        singular[valid] = np.linalg.svd(
            edges[valid] @ np.linalg.inv(matrix[valid]), compute_uv=False
        )
        ratio = np.full(len(block), np.inf)
        good = valid & (singular[:, 1] > 0)
        ratio[good] = singular[good, 0] / singular[good, 1]
        positive[sl] = good
        display[sl] = good & (ratio <= 1.25) & (singular[:, 0] <= 2.0) & (singular[:, 1] >= 0.5)
        block_error = error[sl]
        block_error[good] = np.max(abs(singular[good] - 1), axis=1)
        scientific[sl] = good & (block_error <= 0.02)
        area[sl] = np.linalg.norm(np.cross(edges[:, :, 0], edges[:, :, 1]), axis=1) / 2
    finite = error[np.isfinite(error)]
    report = dict(
        scientific_limit=0.02,
        scientific_acceptance=False,
        maximum_visual_scale_factor=2.0,
        maximum_visual_anisotropy=1.25,
        display_area_fraction=float(area[display].sum() / area.sum()),
        scientific_area_fraction=float(area[scientific].sum() / area.sum()),
        inverted_or_invalid_area_fraction=float(area[~positive].sum() / area.sum()),
        p95_valid_face_metric_distortion=float(np.percentile(finite, 95)) if len(finite) else None,
    )
    return display, scientific, report


def scale_uv_to_world(v, f, uv):
    return _scale_uv_to_world(v, f, uv, degenerate=None)


def _scale_uv_to_world(v, f, uv, degenerate):
    if uv.shape != (len(v), 2) or not np.isfinite(uv).all():
        raise ValueError("invalid mesh or UVs for global scaling")
    if degenerate is None:
        t = v[f]
        area2 = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        degenerate = (area2 <= 1e-12).any()
        del t, area2
    if type(f) in (np.ndarray, np.memmap) and f.ndim == 2 and f.shape[1] == 3 and len(f):
        edges, counts, inconsistent, _ = edge_groups(f)
        invalid_faces = degenerate or duplicate_faces(f)
    else:
        # Preserve legacy handling of empty/nontriangle rows, including NumPy errors.
        oriented = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges, inverse, counts = np.unique(
            np.sort(oriented, axis=1), axis=0, return_inverse=True, return_counts=True
        )
        signs = np.bincount(inverse, weights=np.where(oriented[:, 0] < oriented[:, 1], 1, -1))
        inconsistent = ((counts == 2) & (signs != 0)).any()
        invalid_faces = degenerate or len(np.unique(np.sort(f, axis=1), axis=0)) != len(f)
    if invalid_faces or (counts > 2).any() or inconsistent:
        raise ValueError("invalid mesh or UVs for global scaling")
    xyz_lengths = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1)
    uv_lengths = np.linalg.norm(uv[edges[:, 1]] - uv[edges[:, 0]], axis=1)
    if (uv_lengths <= 0).any() or not np.isfinite(uv_lengths).all():
        raise ValueError("zero/nonfinite UV edge prevents global scaling")
    xyz_sum, uv_sum = xyz_lengths.sum(), uv_lengths.sum()
    scale = float(xyz_sum / uv_sum)
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
        xyz_edge_length_sum=float(xyz_sum),
        uv_edge_length_sum=float(uv_sum),
        anisotropic_or_nonlinear_fit=False,
    )


def make_map(vertices, faces):
    import igl

    v, f = np.asarray(vertices, float), np.asarray(faces)
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
    area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    degenerate = area <= 1e-12
    area /= 2
    curvature = igl.gaussian_curvature(v, f)
    selected = None
    for threshold in [0.1, 0.05, 0.02]:
        keep = (curvature[f] >= -threshold).all(1)
        candidates = np.flatnonzero(keep)
        if not len(candidates):
            continue
        count, labels = igl.facet_components(f[keep])
        best = np.argmax(np.bincount(labels, weights=area[keep]))
        chosen = candidates[labels == best]
        used, inverse = np.unique(f[chosen], return_inverse=True)
        world, tri, parent = igl.split_nonmanifold(v[used], inverse.reshape(-1, 3))
        parent = used[parent]
        boundary = igl.boundary_facets(tri)
        boundary = boundary[0] if isinstance(boundary, tuple) else boundary
        successor = {int(a): int(b) for a, b in boundary}
        remaining = set(successor)
        loops = 0
        if len(successor) != len(boundary):
            raise ValueError("nonmanifold mapping boundary")
        while remaining:
            first = min(remaining)
            current = first
            while current in remaining:
                remaining.remove(current)
                current = successor[current]
            if current != first:
                raise ValueError("open mapping boundary")
            loops += 1
        chi = int(igl.euler_characteristic(tri))
        genus = (2 - loops - chi) / 2
        selected = (world, tri, parent, chosen, threshold, genus, loops, int(count), keep)
        if genus == 0:
            break
    if selected is None:
        raise ValueError("no connected global mapping domain")
    world, tri, parent, chosen, threshold, genus, loops, count, keep = selected
    if genus != 0:
        raise ValueError("global domain still has nonplanar handles; no reliable sheet map")
    if not np.array_equal(world, v[parent]) or not np.array_equal(parent[tri], f[chosen]):
        raise ValueError("mapping-domain selection changed source geometry")
    boundary = igl.boundary_loop(tri).astype(np.int64)
    if not len(boundary):
        raise ValueError("mapping domain has no boundary")
    centered = world - world.mean(0)
    first = int(boundary[0])
    second = int(boundary[np.argmax(np.linalg.norm(centered[boundary] - centered[first], axis=1))])
    length = float(np.linalg.norm(centered[first] - centered[second]))
    uv, _, _ = lscm(centered, tri, first, second, length=length)
    # Source XYZ and every chosen face were checked above; their area predicate is identical.
    uv, calibration = _scale_uv_to_world(world, tri, uv, degenerate[chosen].any())
    t = uv[tri]
    a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    signed = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    weights = area[chosen]
    if weights[signed < 0].sum() > weights[signed > 0].sum():
        uv[:, 0] *= -1
    direction = np.linalg.lstsq(uv - uv.mean(0), world[:, 2] - world[:, 2].mean(), rcond=None)[0]
    if np.linalg.norm(direction) > 1e-12:
        direction /= np.linalg.norm(direction)
        uv = uv @ np.column_stack([[direction[1], -direction[0]], direction])
    display, scientific, quality = face_quality(world, tri, uv)
    if not display.any():
        raise ValueError("no reliable global-map region under the declared visual mask")
    points = uv[np.unique(tri[display])]
    report = dict(
        method="global LSCM on one connected genus-zero domain; holes retained, no patch packing",
        source_faces=len(f),
        mapping_domain_faces=len(tri),
        unmapped_source_faces=len(f) - len(tri),
        source_area_world2=float(area.sum()),
        mapping_domain_area_world2=float(weights.sum()),
        high_saddle_cut_radians=-threshold,
        topology_masked_area_world2=float(area[~keep].sum()),
        unplaced_connected_piece_area_world2=float(area[keep].sum() - weights.sum()),
        mapping_domain_genus=genus,
        mapping_boundary_loops=loops,
        connected_pieces_after_mask=count,
        calibration=calibration,
        quality=quality,
        uv_bounds=[points.min(0).tolist(), points.max(0).tolist()],
        original_XYZ_preserved=True,
        original_faces_retained_in_source=True,
        physical_acceptance=False,
    )
    return dict(
        world=world,
        faces=tri,
        uv=uv,
        parent=parent,
        source_face_rows=chosen,
        display=display,
        scientific=scientific,
        report=report,
    )


def inside_triangles(faces, uv, indices, query):
    t = uv[faces[indices]].astype(np.longdouble)
    q = np.asarray(query, np.longdouble) - t[:, 0]
    a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    cross = lambda x, y: x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
    determinant = cross(a, b)
    n1, n2 = cross(q, b), cross(a, q)
    return (determinant > 0) & (n1 >= 0) & (n2 >= 0) & (n1 <= determinant - n2)


def _points(vertices, faces, uv, indices, query, tolerance):
    try:
        tri = uv[faces[indices]].astype(np.longdouble)
        q = np.asarray(query, np.longdouble)
        a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        relative = q - tri[:, 0]
        cross = lambda x, y: x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
        determinant = cross(a, b)
        if not (determinant > 0).all():
            raise ValueError("invalid local UV determinant")
        n1, n2 = cross(relative, b), cross(a, relative)
        w1, w2 = n1 / determinant, n2 / determinant
        eps = np.finfo(np.longdouble).eps * 16
        den_error = eps * (abs(a[:, 0] * b[:, 1]) + abs(a[:, 1] * b[:, 0]))
        if not (den_error < determinant).all():
            raise ValueError("UV interpolation precision unavailable")
        error1 = (
            eps * (abs(relative[:, 0] * b[:, 1]) + abs(relative[:, 1] * b[:, 0]))
            + abs(w1) * den_error
        ) / (determinant - den_error)
        error2 = (
            eps * (abs(a[:, 0] * relative[:, 1]) + abs(a[:, 1] * relative[:, 0]))
            + abs(w2) * den_error
        ) / (determinant - den_error)
        for corner, weights in enumerate(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))):
            exact = (q == tri[:, corner]).all(1)
            w1[exact], w2[exact] = weights
            error1[exact], error2[exact] = 0, 0
        source = vertices[faces[indices]].astype(np.longdouble)
        x, y = source[:, 1] - source[:, 0], source[:, 2] - source[:, 0]
        error = error1 * np.linalg.norm(x, axis=1) + error2 * np.linalg.norm(y, axis=1)
        if not (error < tolerance / 16).all():
            raise ValueError("XYZ interpolation accuracy cannot be established")
        tol = error1 + error2 + eps
        if not ((w1 >= -tol).all() and (w2 >= -tol).all() and (w1 + w2 <= 1 + tol).all()):
            raise ValueError("query outside triangle")
        return np.asarray(source[:, 0] + w1[:, None] * x + w2[:, None] * y, np.float64), np.ones(
            len(indices), bool
        )
    except ValueError:
        if len(indices) <= 1:
            return np.full((len(indices), 3), np.nan), np.zeros(len(indices), bool)
        mid = len(indices) // 2
        a, ok_a = _points(vertices, faces, uv, indices[:mid], query[:mid], tolerance)
        b, ok_b = _points(vertices, faces, uv, indices[mid:], query[mid:], tolerance)
        return np.concatenate([a, b]), np.r_[ok_a, ok_b]


def raster_candidates(faces, uv, canvas, guard=lambda: None):
    """Conservative pixel bounds in bounded batches; keep original IDs and order."""
    pitch = canvas["pitch"]
    u0 = canvas.get("global_u0", canvas["u0"])
    top = canvas.get("global_v_top", canvas["v_top"])
    col = canvas.get("column_offset", 0)
    row = canvas.get("row_offset", 0)
    for start in range(0, len(faces), 65536):
        guard()
        tri = uv[faces[start : start + 65536]]
        finite = np.isfinite(tri).all((1, 2))
        ids = np.flatnonzero(finite)
        tri = tri[finite]
        if not len(ids):
            continue
        pixel = np.stack(
            ((tri[:, :, 0] - u0) / pitch - col, (top - tri[:, :, 1]) / pitch - row), axis=-1
        )
        lower = np.maximum(0, np.ceil(pixel.min(1) - 1e-10))
        upper = np.minimum(
            [canvas["width"] - 1, canvas["height"] - 1], np.floor(pixel.max(1) + 1e-10)
        )
        for index in ids[(lower <= upper).all(1)]:
            yield start + int(index)


def raster_sheet(
    vertices,
    faces,
    uv,
    eligible,
    sample,
    canvas,
    *,
    window=(0.0, 180.0),
    guard=lambda: None,
    capture=None,
    xyz_tolerance=0.005,
    candidate_ids=None,
):
    """Rasterize an ordered stream of original face IDs, or scan candidates for this tile."""

    v, f, uv = np.asarray(vertices, float), np.asarray(faces), np.asarray(uv, float)
    # Published sheet maps commonly have int32 connectivity; indexing needs no tile-local copy.
    if f.dtype not in (np.int32, np.int64):
        f = f.astype(np.int64)
    width, height = int(canvas["width"]), int(canvas["height"])
    pitch = float(canvas["pitch"])
    u0, top = float(canvas["u0"]), float(canvas["v_top"])
    lo, hi = map(float, window)
    global_u0, global_top = canvas.get("global_u0", u0), canvas.get("global_v_top", top)
    column_offset, row_offset = canvas.get("column_offset", 0), canvas.get("row_offset", 0)
    if not (
        1 <= width <= 4096
        and 1 <= height <= 4096
        and all(map(math.isfinite, (pitch, u0, top, lo, hi, xyz_tolerance)))
        and pitch > 0
        and hi > lo
        and xyz_tolerance > 0
    ):
        raise ValueError("bounded finite sheet canvas/window required")
    guard()
    xyz = np.full((height, width, 3), np.nan)
    # A pixel's first XYZ is already its min and max. Touch the bounds buffer
    # only on a second known hit; ambiguous pixels never become eligible again.
    bounds = np.empty((height * width, 2, 3))
    multiple = np.zeros(height * width, bool)
    owner = np.full((height, width), -1, np.int32)
    mask = np.zeros((height, width), np.uint8)
    coords, owners, states = xyz.reshape(-1, 3), owner.ravel(), mask.ravel()
    display = np.asarray(eligible, bool)
    if display.ndim > 1:
        display = display.reshape(len(display))
    candidates = raster_candidates(f, uv, canvas, guard) if candidate_ids is None else candidate_ids

    def checked_ids():
        for i in candidates:
            if i % 2048 == 0:
                guard()
            yield i

    def merge_hits(pixel, indices, blocked, points, known):
        # A pixel may be visited many times. Replay its visits in face order,
        # vectorizing only independent pixels so an unknown retains its prior XYZ.
        order = np.argsort(pixel, kind="stable")
        cuts = np.r_[0, np.flatnonzero(np.diff(pixel[order])) + 1, len(pixel)]
        pos, end = cuts[:-1], cuts[1:]
        while len(pos):
            hit = order[pos]
            ok = known[hit]
            if not ok.all():
                bad = hit[~ok]
                states[pixel[bad]], owners[pixel[bad]] = 3, -2
                coords[pixel[bad[blocked[bad]]]] = np.nan
                hit = hit[ok]
            key = pixel[hit]
            old = owners[key]
            fresh = old == -1
            if fresh.all():
                coords[key], owners[key], states[key] = points[hit], indices[hit], 1
            else:
                new, k = hit[fresh], key[fresh]
                coords[k], owners[k], states[k] = points[new], indices[new], 1
                coords[key[old == -2]] = np.nan
                overlap = old >= 0
                k, h = key[overlap], hit[overlap]
                low = coords[k]
                high = low.copy()
                seen = multiple[k]
                low[seen], high[seen] = bounds[k[seen], 0], bounds[k[seen], 1]
                p = points[h]
                low, high = np.minimum(low, p), np.maximum(high, p)
                bounds[k, 0], bounds[k, 1], multiple[k] = low, high, True
                conflict = np.linalg.norm(high - low, axis=1) > xyz_tolerance
                replace = (indices[h] < old[overlap]) & ~conflict
                coords[k[replace]], owners[k[replace]] = p[replace], indices[h[replace]]
                k = k[conflict]
                states[k], owners[k], coords[k] = 3, -2, np.nan
            more = pos + 1 < end
            pos, end = pos[more] + 1, end[more]

    stream = checked_ids()
    while len(ids := np.fromiter(islice(stream, 2048), np.int64)):
        tri = f[ids]
        t = uv[tri]
        finite = np.isfinite(t).all((1, 2))
        if not finite.all():
            ids, tri, t = ids[finite], tri[finite], t[finite]
        pixel = np.stack(
            (
                (t[:, :, 0] - global_u0) / pitch - column_offset,
                (global_top - t[:, :, 1]) / pitch - row_offset,
            ),
            axis=-1,
        )
        lower, upper = np.ceil(pixel.min(1) - 1e-10), np.floor(pixel.max(1) + 1e-10)
        if candidate_ids is not None:
            near = (np.maximum(0, lower) <= np.minimum([width - 1, height - 1], upper)).all(1)
            if not near.all():
                ids, tri, t, lower, upper = ids[near], tri[near], t[near], lower[near], upper[near]
        a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        signed = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        degenerate = signed == 0
        lower[degenerate], upper[degenerate] = 0, -1
        lower = np.maximum(0, lower.astype(int))
        upper = np.minimum([width - 1, height - 1], upper.astype(int))
        use = (lower <= upper).all(1)
        if not use.all():
            ids, tri, signed, lower, upper = ids[use], tri[use], signed[use], lower[use], upper[use]
        if not len(ids):
            continue
        reverse = signed < 0
        tri[reverse] = tri[reverse][:, [0, 2, 1]]
        blocked = ~display[ids] | reverse
        nx, ny = (upper - lower + 1).T
        ends = np.cumsum(nx * ny)
        starts = ends - nx * ny
        # Prefix sums index the concatenated bounding boxes without expanding a
        # large triangle (or a stack of overlapping triangles) beyond 128² pixels.
        for start in range(0, int(ends[-1]), 128**2):
            if start and start % (16 * 128**2) == 0:
                guard()
            offset = np.arange(start, min(start + 128**2, ends[-1]))
            first, last = np.searchsorted(ends, (offset[0], offset[-1]), side="right")
            active = slice(first, last + 1)
            counts = np.minimum(ends[active], offset[-1] + 1) - np.maximum(starts[active], start)
            local = np.repeat(np.arange(first, last + 1), counts)
            offset -= starts[local]
            cc = lower[local, 0] + offset % nx[local]
            rr = lower[local, 1] + offset // nx[local]
            query = np.column_stack(
                (global_u0 + (cc + column_offset) * pitch, global_top - (rr + row_offset) * pitch)
            )
            inside = inside_triangles(tri, uv, local, query)
            if not inside.any():
                continue
            local = local[inside]
            indices, bad = ids[local], blocked[local]
            known = ~bad
            if known.all():
                points, known = _points(v, f, uv, indices, query[inside], xyz_tolerance)
            else:
                points = np.empty((len(local), 3))
                if known.any():
                    points[known], known[known] = _points(
                        v, f, uv, indices[known], query[inside][known], xyz_tolerance
                    )
            merge_hits(rr[inside] * width + cc[inside], indices, bad, points, known)
    image = np.zeros((height, width), np.uint8)
    for start in range(0, height, max(1, 100000 // width)):
        guard()
        stop = min(height, start + max(1, 100000 // width))
        rr, cc = np.where(mask[start:stop] == 1)
        rr = rr + start
        if not len(rr):
            continue
        values, valid = sample(xyz[rr, cc])
        valid = np.asarray(valid, bool) & np.isfinite(values)
        image[rr[valid], cc[valid]] = np.rint(
            np.clip((values[valid] - lo) * 255 / (hi - lo), 0, 255)
        ).astype(np.uint8)
        mask[rr[valid], cc[valid]] = 2
    report = dict(
        CT_observed_pixels=int((mask == 2).sum()),
        CT_missing_pixels=int((mask == 1).sum()),
        ambiguous_pixels=int((mask == 3).sum()),
        black_unmapped_pixels=int((mask == 0).sum()),
        canvas=canvas,
        black_is_unmapped_or_unsupported=True,
        original_UV_positions_preserved=True,
        patch_repacking=False,
        physical_acceptance=False,
    )
    if capture is not None:
        capture(xyz, mask, owner)
    return image, mask, report
