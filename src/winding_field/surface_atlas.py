#!/usr/bin/env python3
"""Declared one-human-seed development atlas; every original face is retained.

CLI: --field --metadata --inputs <old probe inputs.json> --patch --app-run
     --output NEW [--max-depth 3] [--max-charts 8] [--budget-seconds 540]
Run under run_watch.py --timeout 600, CPU thread limits 4. Native charts use
200 iterations, symmetric_dirichlet, --tol=1e-6, at most 120 seconds each.
Pitch/offsets are in the declared world units. This is NOT blind autonomous recovery.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .surface_export import _metadata, _sha, extract_surface, flatten_native, rasterize_surface, sample_ct_stack
from .surface_probe import (
    context_image,
    evaluate_patch,
    load_field_support,
    patch_coordinates,
    seam_registration,
    write_json,
)

from dataset_config import frame as validate_frame
from dataset_config import positive


def choose_human_seed(xyz, metadata):
    """Geometry only; callers must freeze this receipt BEFORE sampling the field."""
    xyz = np.asarray(xyz, float)
    origin, spacing, _ = _metadata(metadata, metadata["shape_zyx"])
    q = (xyz - origin) / spacing
    valid = (
        np.isfinite(xyz).all(-1)
        & (xyz >= 0).all(-1)
        & (q >= 16).all(-1)
        & (q <= np.array(metadata["shape_zyx"][::-1]) - 1 - 16).all(-1)
    )
    if not valid.any():
        raise ValueError("no interior human point for the frozen geometric seed rule")
    center = origin + (np.array(metadata["shape_zyx"][::-1]) - 1) * spacing / 2
    distances = np.where(valid, np.linalg.norm(xyz - center, axis=-1), np.inf)
    row, col = np.unravel_index(distances.argmin(), distances.shape)
    rr, cc = np.indices(valid.shape)
    excluded = (np.abs(rr - row) <= 2) & (np.abs(cc - col) <= 2)
    return dict(
        human_seed=True,
        seed_uv_rc=[int(row), int(col)],
        seed_xyz=xyz[row, col].tolist(),
        roi_center_xyz=center.tolist(),
        interior_grid_margin=16,
        excluded_uv_radius=2,
        rule="nearest valid interior supplied-patch XYZ point to physical field-sample ROI center; row-major tie",
        selection_inputs="human geometry and ROI metadata only; no field/CT/score",
        scope="one-seed development demonstrator; not blind autonomous success or model/holdout tuning",
        frozen_utc=datetime.now(UTC).isoformat(),
    ), excluded


def local_chart(vertices, faces, face_ids):
    ids, inverse = np.unique(faces[face_ids], return_inverse=True)
    return vertices[ids], inverse.reshape(-1, 3), ids


def face_components(faces, ids):
    f = faces[ids]
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    owners = np.tile(np.arange(len(f)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    equal = (edges[order][1:] == edges[order][:-1]).all(1)
    a, b = owners[order][:-1][equal], owners[order][1:][equal]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(f), len(f)))
    count, labels = connected_components(graph, directed=False)
    return sorted([ids[labels == k] for k in range(count)], key=lambda x: int(x.min()))


def split_faces(vertices, faces, face_ids):
    """Balanced longest-PCA-axis centroid cut, then edge-connected components."""
    face_ids = np.asarray(face_ids, dtype=np.int64)
    if len(face_ids) < 2:
        return [face_ids]
    centers = vertices[faces[face_ids]].mean(1)
    _, _, axes = np.linalg.svd(centers - centers.mean(0), full_matrices=False)
    axis = axes[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    order = np.lexsort((face_ids, centers @ axis))
    halves = [face_ids[order[: len(order) // 2]], face_ids[order[len(order) // 2 :]]]
    return [piece for half in halves for piece in face_components(faces, half)]


def conservation(vertices, faces, pieces):
    ids = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    if (ids < 0).any() or (ids >= len(faces)).any():
        return dict(status="FAIL", reason="invalid global face IDs")
    counts = np.bincount(ids, minlength=len(faces))
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    return dict(
        status="PASS" if (counts == 1).all() else "FAIL",
        source_faces=len(faces),
        retained_faces=len(ids),
        removed_faces=int((counts == 0).sum()),
        duplicated_faces=int(np.maximum(counts - 1, 0).sum()),
        source_area_world2=float(areas.sum()),
        retained_area_world2=float(areas[ids].sum()),
        removed_area_world2=float(areas[counts == 0].sum()),
    )


def save_cuts(path, faces, leaves):
    owner = np.full(len(faces), -1, dtype=np.int64)
    for i, leaf in enumerate(leaves):
        owner[leaf["face_ids"]] = i
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    fids = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    shared = (edges[order][1:] == edges[order][:-1]).all(1)
    pair = np.column_stack([owner[fids[order][:-1]], owner[fids[order][1:]]])
    cut = shared & (pair[:, 0] != pair[:, 1])
    np.savez(
        path,
        global_edges=edges[order][:-1][cut],
        chart_pairs=pair[cut],
        chart_directories=np.array([c["directory"] for c in leaves]),
    )
    return int(cut.sum())


def chart_preview(path, stack, valid, layer, *, pitch_um, window=(0, 255)):
    from PIL import Image, PngImagePlugin

    lo, hi = window
    if not np.isfinite(window).all() or hi <= lo:
        raise ValueError("invalid CT display window")
    gray = np.rint(255 * np.clip((np.nan_to_num(stack[layer]) - lo) / (hi - lo), 0, 1)).astype(np.uint8)
    rgba = np.dstack([gray, gray, gray, valid[layer].astype(np.uint8) * 255])
    info = PngImagePlugin.PngInfo()
    info.add_text(
        "Description", f"CT texture, not ink; {pitch_um:g}um/pixel; fixed [{lo:g},{hi:g}] window; invalid transparent"
    )
    Image.fromarray(rgba).save(path, pnginfo=info)


def build_atlas(
    vertices,
    faces,
    metadata,
    ct,
    ct_metadata,
    app_run,
    output,
    *,
    max_depth=3,
    max_charts=8,
    budget_seconds=540,
    pitch_world=1,
    depth_offsets_world=(-8, -4, 0, 4, 8),
):
    if not 0 <= max_depth <= 3 or not 1 <= max_charts <= 8 or budget_seconds <= 0:
        raise ValueError("bounded atlas requires depth <=3, <=8 charts and positive time budget")
    positive(pitch_world, "pitch_world")
    offsets = np.asarray(depth_offsets_world, float)
    if offsets.ndim != 1 or not len(offsets) or not np.isfinite(offsets).all():
        raise ValueError("finite depth offsets are required")
    output = Path(output)
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    plan = dict(
        max_depth=max_depth,
        max_charts=max_charts,
        budget_seconds=budget_seconds,
        split_rule="triangle-centroid longest PCA direction; balanced count; edge-connected pieces",
        face_removal_allowed=False,
        distortion_statistic="p95",
        distortion_limit=0.02,
        policy_version="uv-p95-v2",
        pitch_world=pitch_world,
        pitch_um=pitch_world * metadata["voxel_um"],
        depth_offsets_world=offsets.tolist(),
        native_iters=200,
        native_timeout_max=120,
        geometry_status="UNMEASURED",
        writing_face="UNMEASURED",
        ink="NOT_MEASURED",
    )
    write_json(output / "plan.json", plan)
    nodes, active, pending = [], {}, []

    def add(ids, depth, parent=None):
        name = f"chart-{len(nodes):03d}"
        directory = output / name
        directory.mkdir()
        v, f, vi = local_chart(vertices, faces, ids)
        np.savez(directory / "chart.npz", vertices_xyz=v, faces=f, global_vertex_ids=vi, global_face_ids=ids)
        node = dict(
            directory=name,
            depth=depth,
            parent=parent,
            faces=len(ids),
            face_ids=ids,
            uv_status="UNMEASURED",
            render_status="UNMEASURED",
            status="PENDING",
        )
        nodes.append(node)
        active[name] = node
        pending.append(node)

    def public(node):
        return {k: v for k, v in node.items() if k != "face_ids"}

    def progress():
        state = dict(
            elapsed_s=time.monotonic() - start,
            nodes=[public(n) for n in nodes],
            leaves=list(active),
            pending=[n["directory"] for n in pending],
        )
        (output / "progress.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")

    raw = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    global_normals = np.zeros_like(vertices)
    for col in range(3):
        np.add.at(global_normals, faces[:, col], raw)
    global_normals /= np.maximum(np.linalg.norm(global_normals, axis=1)[:, None], 1e-12)
    for ids in face_components(faces, np.arange(len(faces))):
        add(ids, 0)
    while pending:
        node = pending.pop(0)
        remaining = budget_seconds - (time.monotonic() - start)
        if remaining <= 5 or len(active) > max_charts:
            node.update(status="FAIL", reason="atlas_time_or_chart_budget_exceeded")
            progress()
            continue
        directory = output / node["directory"]
        v, f, vi = local_chart(vertices, faces, node["face_ids"])
        print(
            "ATLAS",
            node["directory"],
            "depth",
            node["depth"],
            "faces",
            len(f),
            "remaining",
            round(remaining),
            flush=True,
        )
        uv, native = flatten_native(
            v, f, app_run, directory / "native", iters=200, timeout=min(120, max(1, remaining - 5))
        )
        node["native"] = native
        node["uv_status"] = native["status"]
        if uv is not None:
            np.savez(directory / "uv.npz", uv=uv)
        if native["status"] == "PASS":
            try:
                points, normals, valid, raster = rasterize_surface(
                    v, f, uv, pitch_world, vertex_normals_xyz=global_normals[vi]
                )
                raster["pitch_um"] = [pitch_world * metadata["voxel_um"]] * 2
                np.savez(
                    directory / "grid.npz",
                    points_xyz=points,
                    normals_xyz=normals,
                    valid=valid,
                    pitch_um=raster["pitch_um"],
                    uv_origin=raster["uv_origin"],
                )
                stack, mask, sampled = sample_ct_stack(
                    ct, ct_metadata, points, normals, offsets, frame=metadata["frame"], voxel_um=metadata["voxel_um"]
                )
                np.savez(
                    directory / "ct_stack.npz",
                    values=stack,
                    valid=mask,
                    depth_offsets=offsets,
                    depth_offsets_um=offsets * metadata["voxel_um"],
                )
                if not mask.any():
                    raise ValueError("chart has no CT coverage")
                chart_preview(
                    directory / "ct_preview.png",
                    stack,
                    mask,
                    int(np.argmin(np.abs(offsets))),
                    pitch_um=pitch_world * metadata["voxel_um"],
                    window=ct_metadata.get("display_window", [0, 255]),
                )
                node.update(render_status="PASS", status="PASS", raster=raster, sampling=sampled)
                del points, normals, valid, stack, mask
            except (ValueError, OSError, RuntimeError) as error:
                node.update(render_status="FAIL", status="FAIL", reason=str(error))
        else:
            parts = (
                split_faces(vertices, faces, node["face_ids"])
                if node["depth"] < max_depth and native.get("uv_gate")
                else []
            )
            if len(parts) > 1 and len(active) - 1 + len(parts) <= max_charts:
                del active[node["directory"]]
                node["status"] = "SPLIT"
                for ids in parts:
                    add(ids, node["depth"] + 1, node["directory"])
            else:
                node.update(status="FAIL", reason="required_chart_failed_at_frozen_budget_or_native_unavailable")
        write_json(directory / "receipt.json", public(node))
        progress()
    leaves = list(active.values())
    conserved = conservation(vertices, faces, [c["face_ids"] for c in leaves])
    cut_count = save_cuts(output / "cuts.npz", faces, leaves)
    success = conserved["status"] == "PASS" and all(c["status"] == "PASS" for c in leaves)
    result = dict(
        atlas_status="PASS" if success else "FAIL",
        frame=metadata["frame"],
        voxel_um=metadata["voxel_um"],
        geometry_status="UNMEASURED",
        scientific_scope="bounded mesh atlas; seed provenance supplied by caller; physical lamina/writing-face/ink unverified",
        plan=plan,
        conservation=conserved,
        cut_edges=cut_count,
        nodes=[public(c) for c in nodes],
        leaves=[public(c) for c in leaves],
        successful_charts=sum(c["status"] == "PASS" for c in leaves),
        seconds=time.monotonic() - start,
        cut_correspondence="cuts.npz + chart.npz global IDs",
    )
    result["artifacts"] = {
        str(p.relative_to(output)): dict(sha256=_sha(p), bytes=p.stat().st_size)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    write_json(output / "receipt.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("field", "metadata", "inputs", "app-run", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--patch", type=Path, help="optional human-seed patch; otherwise use the frozen prior seed in inputs"
    )
    p.add_argument("--patch-frame")
    p.add_argument("--patch-voxel-um", type=float)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--max-charts", type=int, default=8)
    p.add_argument("--budget-seconds", type=float, default=540)
    args = p.parse_args(argv)
    args.output.mkdir(exist_ok=False)
    result = dict(
        status="FAILED",
        human_seed=args.patch is not None,
        source_sha256=_sha(__file__),
        source_hashes={n: _sha(Path(__file__).with_name(n)) for n in ("surface_export.py", "surface_probe.py")},
    )
    try:
        import tifffile

        from patch_check import sample_field

        meta = json.loads(args.metadata.read_text())
        inputs = json.loads(args.inputs.read_text())
        validate_frame(meta.get("frame"))
        positive(meta.get("voxel_um"), "voxel_um")
        render = inputs.get("render", {})
        plan = dict(
            max_depth=args.max_depth,
            max_charts=args.max_charts,
            budget_seconds=args.budget_seconds,
            human_seed=args.patch is not None,
            exclude_uv_radius=2,
            mesh_step=render.get("mesh_step", 2),
            pitch_world=render.get("pitch_world", 1),
            policy_correction="approved gate is p95<=2%, not maximum<=2%; maximum still reported",
            automatic_result_unchanged=True,
            model_or_holdout_tuning=False,
        )
        write_json(args.output / "plan.json", plan)
        field = np.load(args.field, mmap_mode="r", allow_pickle=False)
        if _sha(args.field) != meta["field_sha256"] or _sha(args.metadata) != inputs["metadata_sha256"]:
            raise ValueError("frozen field/metadata hash mismatch")
        if inputs["field_sha256"] != meta["field_sha256"]:
            raise ValueError("frozen input field identity mismatch")
        if args.patch is not None:
            coordinates = patch_coordinates(
                args.patch, meta, patch_frame=args.patch_frame, patch_voxel_um=args.patch_voxel_um
            )
            xyz = np.stack([tifffile.imread(args.patch / f"{a}.tif") for a in "xyz"], axis=-1)
            selection, excluded = choose_human_seed(xyz, meta)
            selection["patch_coordinates"] = coordinates
            selection["patch_sha256"] = {n: _sha(args.patch / n) for n in ("meta.json", "x.tif", "y.tif", "z.tif")}
            np.save(args.output / "excluded_human_uv.npy", excluded)
        else:
            if _sha(inputs["seed"]) != inputs["seed_sha256"]:
                raise ValueError("frozen seed hash mismatch")
            selection = json.loads(Path(inputs["seed"]).read_text())
            if (
                selection["field_sha256"] != meta["field_sha256"]
                or selection.get("frame", meta["frame"]) != meta["frame"]
            ):
                raise ValueError("frozen seed and field identities differ")
            selection = dict(selection, human_seed=False)
        write_json(args.output / "seed_geometry.json", selection)
        value = sample_field(field, np.array([selection["seed_xyz"]]), meta["origin_xyz"], meta["spacing_xyz"])[0]
        if not np.isfinite(value):
            raise ValueError("seed has no field support")
        if args.patch is None:
            tolerance = 2e-5 + 0.5 * float(np.spacing(np.asarray(abs(value), dtype=field.dtype)))
            if not np.isclose(value, selection["level"], rtol=0, atol=tolerance):
                raise ValueError("frozen prior level differs from field at the frozen seed")
            value = float(selection["level"])
        selection = {
            **selection,
            "level": float(value),
            "level_method": "raw field at frozen human seed" if args.patch else "unchanged frozen prior-seed level",
            "field_sha256": meta["field_sha256"],
        }
        write_json(args.output / "seed.json", selection)
        print("FROZEN SEED", json.dumps(selection), flush=True)
        support, _ = load_field_support(field, args.field, meta)
        extraction_meta = {k: v for k, v in meta.items() if k != "support"}
        if support is not None:
            extraction_meta["support"] = support
        vertices, faces, extraction = extract_surface(
            field, extraction_meta, value, selection["seed_xyz"], step_size=render.get("mesh_step", 2)
        )
        np.savez(args.output / "surface.npz", vertices_xyz=vertices, faces=faces)
        write_json(args.output / "extraction.json", extraction)
        for key in ("ct", "ct_metadata"):
            if _sha(inputs[key]) != inputs[key + "_sha256"]:
                raise ValueError("cached CT identity mismatch: " + key)
        ct = np.load(inputs["ct"], mmap_mode="r", allow_pickle=False)
        ct_meta = json.loads(Path(inputs["ct_metadata"]).read_text())
        atlas = build_atlas(
            vertices,
            faces,
            meta,
            ct,
            ct_meta,
            args.app_run,
            args.output / "atlas",
            max_depth=args.max_depth,
            max_charts=args.max_charts,
            budget_seconds=args.budget_seconds,
            pitch_world=render.get("pitch_world", 1),
            depth_offsets_world=render.get("depth_offsets_world", [-8, -4, 0, 4, 8]),
        )
        ownership = (
            np.load(args.field.parent / meta["core_ownership"], mmap_mode="r", allow_pickle=False)
            if meta.get("core_ownership") is not None
            else None
        )
        result.update(
            seed=selection,
            inputs=inputs,
            extraction=extraction,
            atlas_status=atlas["atlas_status"],
            atlas_receipt_sha256=_sha(args.output / "atlas/receipt.json"),
            conservation=atlas["conservation"],
            successful_charts=atlas["successful_charts"],
            required_charts=len(atlas["leaves"]),
            seams=seam_registration(vertices, faces, meta, ownership)
            if ownership is not None
            else dict(status="NOT_APPLICABLE", reason="no tiled ownership declared"),
        )
        if args.patch is not None:
            result["validation"] = evaluate_patch(
                args.patch,
                vertices,
                faces,
                meta,
                ownership,
                exclude_uv_rc=selection["seed_uv_rc"],
                exact=True,
                patch_frame=args.patch_frame,
                patch_voxel_um=args.patch_voxel_um,
            )
        else:
            result["validation"] = dict(status="NOT_EVALUATED", reason="no human reference supplied")
        context_image(args.output / "ct_context.png", ct, ct_meta, selection, vertices)
        result["status"] = "EXPORTED_UNCERTIFIED" if atlas["atlas_status"] == "PASS" else "ATLAS_FAILED"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(args.output / "result.json", result)
    print(
        "SEEDED RESULT",
        json.dumps(dict(status=result["status"], error=result.get("error"), path=str(args.output))),
        flush=True,
    )
    return 0 if result["status"] == "EXPORTED_UNCERTIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
