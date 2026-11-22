"""Conservative mesh-derived XYZ cells from the original raster's shared buffers."""

import numpy as np
from PIL import Image
from ..util import atomic_json, file_sha, digest


def domain_edges(faces, uv, eligible, guard):
    """Sorted undirected edges with incidence != 2 in the full eligible domain."""
    eligible = np.asarray(eligible, bool)
    packed = len(uv) <= 2**32 and (not faces.size or faces.min() >= 0)
    keys = np.empty(3 * len(faces), np.uint64) if packed else []
    used = 0
    for first in range(0, len(faces), 65536):
        guard()
        f = faces[first : first + 65536]
        t = uv[f]
        a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        f = f[
            eligible[first : first + len(f)]
            & (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] > 0)
            & np.isfinite(t).all(axis=(1, 2))
        ]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            lo, hi = np.minimum(f[:, i], f[:, j]), np.maximum(f[:, i], f[:, j])
            if packed:
                keys[used : used + len(f)] = (lo.astype(np.uint64) << 32) | hi.astype(np.uint64)
                used += len(f)
            else:
                keys.append(np.column_stack((lo, hi)))
    if not packed:
        edges, counts = np.unique(np.concatenate(keys), axis=0, return_counts=True)
        return edges[counts != 2]
    guard()
    unique, counts = np.unique(keys[:used], return_counts=True)
    boundary = unique[counts != 2]
    return np.column_stack((boundary >> 32, boundary & 0xFFFFFFFF)).astype(faces.dtype)


def export_xyz(
    root,
    xyz,
    mask,
    faces,
    uv,
    eligible,
    canvas,
    frame,
    guard=lambda: None,
    *,
    binding=None,
    publish=None,
    boundary_batches=None,
):
    root.mkdir(exist_ok=False)
    guard()
    valid = (mask == 1) | (mask == 2)
    original = int(valid.sum())
    height, width = valid.shape
    unsafe = np.zeros((height - 1, width - 1), bool)
    pitch = float(canvas["pitch"])
    if boundary_batches is None:
        edges = domain_edges(faces, uv, eligible, guard)
        boundary_batches = (edges[i : i + 2048] for i in range(0, len(edges), 2048))
    for batch in boundary_batches:
        guard()
        t = uv[batch]
        x = (t[:, :, 0] - canvas.get("global_u0", canvas["u0"])) / pitch - canvas.get(
            "column_offset", 0
        )
        y = (canvas.get("global_v_top", canvas["v_top"]) - t[:, :, 1]) / pitch - canvas.get(
            "row_offset", 0
        )
        xmin, xmax = (
            np.maximum(0, np.floor(x.min(1)).astype(int)),
            np.minimum(width - 2, np.floor(x.max(1)).astype(int)),
        )
        ymin, ymax = (
            np.maximum(0, np.floor(y.min(1)).astype(int)),
            np.minimum(height - 2, np.floor(y.max(1)).astype(int)),
        )
        for i in np.flatnonzero((xmax >= xmin) & (ymax >= ymin)):
            unsafe[ymin[i] : ymax[i] + 1, xmin[i] : xmax[i] + 1] = True
    for y in range(0, height - 1, 64):
        guard()
        stop = min(y + 64, height - 1)
        p = xyz[y : stop + 1].astype(np.float32).astype(float)
        v = valid[y : stop + 1]
        active = v[:-1, :-1] & v[:-1, 1:] & v[1:, :-1] & v[1:, 1:]
        a, b, c, d = (q[active] for q in (p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]))
        bad = np.zeros(len(a), bool)
        for s, t in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            bad |= np.linalg.norm(s - t, axis=1) > 8 * pitch
        n1 = np.cross(b - a, c - a)
        n2 = np.cross(d - b, c - b)
        bad |= (
            (np.linalg.norm(n1, axis=1) <= 1e-12)
            | (np.linalg.norm(n2, axis=1) <= 1e-12)
            | (np.sum(n1 * n2, axis=1) <= 0)
        )
        unsafe[y:stop][active] |= bad
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        valid[dy : dy + height - 1, dx : dx + width - 1] &= ~unsafe
    cells = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    used = np.zeros_like(valid)
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        used[dy : dy + height - 1, dx : dx + width - 1] |= cells
    valid &= used
    if publish is not None:
        crop, canvas = publish
        xyz, mask, valid = xyz[crop], mask[crop], valid[crop]
        height, width = valid.shape
        original = int(((mask == 1) | (mask == 2)).sum())
        cells = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    if not valid.any() or (publish is None and not cells.any()):
        raise ValueError("TIFXYZ contains no retained surface cells")
    error = 0.0
    for axis, name in enumerate(("x.tif", "y.tif", "z.tif")):
        guard()
        channel = np.full(valid.shape, -1, np.float32)
        for y in range(0, height, 64):
            s = slice(y, y + 64)
            v = valid[s]
            values = xyz[s, :, axis][v]
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("invalid retained XYZ in export frame")
            channel[s][v] = values
            if len(values):
                error = max(error, float(np.max(np.abs(channel[s][v].astype(float) - values))))
        Image.fromarray(channel).save(root / name, compression="tiff_adobe_deflate")
        with Image.open(root / name) as im:
            decoded = np.asarray(im)
            if decoded.dtype != np.float32 or not np.array_equal(decoded, channel):
                raise ValueError("TIFXYZ readback mismatch")
        del channel, decoded
    Image.fromarray(valid.astype(np.uint8) * 255).save(root / "valid-mask.png")
    meta = dict(
        format="tifxyz",
        type="seg",
        uuid=digest(dict(frame=frame, canvas=canvas, binding=binding)),
        source_binding=binding,
        shape_yx=[height, width],
        dtype="float32",
        scale=[1 / pitch, 1 / pitch],
        pitch_um=pitch * frame["voxel_um"],
        coordinate_units="output-frame voxels",
        axis_order="XYZ",
        invalid_value=-1,
        output_frame=frame,
        canvas=canvas,
        row_direction="decreasing UV v",
        physical_acceptance=False,
    )
    atomic_json(root / "meta.json", meta)
    files = {p.name: dict(sha256=file_sha(p), bytes=p.stat().st_size) for p in root.iterdir()}
    report = dict(
        status="TIFXYZ_READBACK_VERIFIED"
        if cells.any()
        else "TIFXYZ_SHARED_NODES_READBACK_VERIFIED",
        files=files,
        original_mapped_nodes=original,
        retained_nodes=int(valid.sum()),
        removed_nodes=original - int(valid.sum()),
        retained_cells=int(cells.sum()),
        max_float32_error_world=error,
        geometry_mask_includes_CT_missing=True,
        physical_acceptance=False,
        vc3d_readback=False,
        exclusions="domain-boundary rectangles, jumps >8*pitch, collapsed/folded cells, orphan nodes",
    )
    atomic_json(root / "VERIFIED.json", report)
    return report
