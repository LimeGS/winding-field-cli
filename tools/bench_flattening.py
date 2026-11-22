"""Opt-in, source-bound flattening experiments. Never changes production defaults."""

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

import numpy as np

from winding_field.util import atomic_json, digest, file_sha


def read_json(path):
    return json.loads(Path(path).read_text())


def safe(root, name):
    root = Path(root).absolute()
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or any(p in ("", ".", "..") for p in name.split("/"))
    ):
        raise ValueError("unsafe payload path")
    path = root / name
    if Path(name).is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink or absolute payload path refused")
    return path


@contextmanager
def locked(root):
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("output symlink refused")
    root.mkdir(parents=True, exist_ok=True)
    with safe(root, ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield root


def verify(root, name, identity=None):
    root = Path(root)
    record = read_json(safe(root, name))
    if identity is not None and record["identity"] != identity:
        raise ValueError("receipt identity changed: " + name)
    for path, sha in record["files"].items():
        if file_sha(safe(root, path)) != sha:
            raise ValueError("payload hash changed: " + path)
    return record


def seal(root, name, identity, report, paths):
    root = Path(root)
    record = dict(
        identity=identity,
        report=report,
        files={p: file_sha(safe(root, p)) for p in paths},
        physical_acceptance=False,
    )
    atomic_json(safe(root, name), record)
    return record


def validate_freeze(run):
    run = Path(run)
    path = safe(run, "FREEZE.json")
    if not path.is_file():
        raise ValueError("holdout requires a bound review FREEZE.json")
    freeze = read_json(path)
    if not isinstance(freeze, dict) or freeze.get("run_sha256") != file_sha(run / "RUN.json"):
        raise ValueError("holdout FREEZE run identity differs")
    decision = freeze.get("decision")
    if (
        not isinstance(decision, str)
        or not Path(decision).is_file()
        or file_sha(decision) != freeze.get("decision_sha256")
    ):
        raise ValueError("holdout FREEZE requires hash-bound human review decision")
    candidates = freeze.get("candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(
            not isinstance(c, list)
            or len(c) != 2
            or c[0] not in METHODS
            or c[1] not in ("sheet", "patches")
            for c in candidates
        )
    ):
        raise ValueError("holdout FREEZE requires explicit candidate recipes")
    return freeze


def raw_level(w, step):
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (w, step)) or step <= 0:
        raise ValueError("finite W and positive sheet_step required")
    return float(w * step)


def field_descriptor(config):
    result = json.loads(json.dumps(config))
    if (
        result["geometry"]["frame"].startswith("PHercParis4/")
        and result["field"].get("support_policy") != "stored"
    ):
        raise ValueError("Paris4 requires stored support, not the observed sentinel")
    return result


def resolve_mesh(job, receipts):
    sha = job["source_control"]["database_sha256"]
    found = []
    for p in receipts:
        p = Path(p)
        if read_json(p).get("database_sha256") == sha:
            found.append(p.parent)
    if not found:
        raise ValueError("unresolved source database hash: " + sha)
    # Byte-identical archival mirrors are allowed; verification occurs on open.
    return min(found, key=str)


def case_arrays(case):
    root = Path(case["root"])
    record = verify(root, "SOURCE.json")
    if record["report"] != {k: v for k, v in case.items() if k != "root"}:
        raise ValueError("case binding changed")
    return {
        Path(p).stem: np.load(safe(root, p), allow_pickle=False, mmap_mode="r")
        for p in record["files"]
        if p.endswith(".npy")
    }


def load_case(source, output):
    from winding_field.patch_pipeline import open_mesh, read_level

    output = Path(output).absolute()
    mesh = Path(source["mesh"]).absolute()
    if output.is_relative_to(mesh) or mesh.is_relative_to(output):
        raise ValueError("source and output must be separate; containment overlap")
    for key in ("field_id", "resolution_id"):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", source[key]):
            raise ValueError("invalid case identifier")
    ct = source["ct_profile"]
    if ct["field_id"] != source["field_id"] or ct["frame"] != source["frame"]:
        raise ValueError("CT field/frame binding differs")
    if file_sha(safe(mesh, "MESH.json")) != source["mesh_manifest_sha256"]:
        raise ValueError("mesh receipt hash changed")
    identity = dict(schema="flattening-case-v1", source=source)
    if (output / "SOURCE.json").exists():
        record = verify(output, "SOURCE.json", identity)
        return dict(record["report"], root=str(output))
    with open_mesh(mesh) as (reader, _):
        levels = reader.manifest["levels"]
        if (
            len(levels) != 1
            or levels[0]["level"] != source["raw_phi"]
            or reader.metadata["frame"] != source["frame"]
        ):
            raise ValueError("mesh level/frame binding differs")
        if raw_level(source["W"], reader.metadata["sheet_step"]) != source["raw_phi"]:
            raise ValueError("nominal W/raw phi binding differs")
        v, f, vi, fi, _, _ = read_level(reader, levels[0]["id"], memory_gib=6.0)
        world = v * reader.metadata["voxel_um"]
        reader.validate()
    area = (
        np.linalg.norm(
            np.cross(world[f[:, 1]] - world[f[:, 0]], world[f[:, 2]] - world[f[:, 0]]), axis=1
        )
        / 2
    )
    w = source["W"]
    label = ("p" if w >= 0 else "m") + (
        f"{abs(w):03d}" if type(w) is int else str(abs(w)).replace(".", "_")
    )
    report = dict(
        id=f"{source['field_id']}-{source['resolution_id']}-W{label}",
        field_id=source["field_id"],
        resolution_id=source["resolution_id"],
        W=w,
        raw_phi=source["raw_phi"],
        frame=source["frame"],
        area_mm2=float(area.sum() / 1e6),
        source_sha256=source["mesh_manifest_sha256"],
        ct_profile=ct,
        ct_profile_sha256=digest(ct),
        faces=len(f),
        vertices=len(v),
    )
    with locked(output):
        if (output / "SOURCE.json").exists():
            return dict(verify(output, "SOURCE.json", identity)["report"], root=str(output))
        arrays = dict(world_um=world, faces=f, source_vertex_ids=vi, source_face_ids=fi)
        for name, values in arrays.items():
            np.save(output / (name + ".npy"), values, allow_pickle=False)
        seal(output, "SOURCE.json", identity, report, [k + ".npy" for k in arrays])
    return dict(report, root=str(output))


def prepare_domains(case, output, scope, guard, *, max_patch_faces=4096):
    import igl
    from winding_field.native_global import cut_component
    from winding_field.surface_atlas import face_components, local_chart, split_faces

    if scope not in ("sheet", "patches") or type(max_patch_faces) is not int or max_patch_faces < 1:
        raise ValueError("invalid domain selection")
    a = case_arrays(case)
    v, f = a["world_um"], a["faces"]
    identity = dict(
        source=file_sha(Path(case["root"]) / "SOURCE.json"),
        scope=scope,
        max_patch_faces=max_patch_faces,
    )
    if scope == "patches":
        identity["partition_policy"] = "pca-bisection-until-vertex-manifold-v2"
    output = Path(output)
    with locked(output):
        if (output / "DOMAINS.json").exists():
            saved = verify(output, "DOMAINS.json", identity)
            domains = []
            for row in saved["report"]["domains"]:
                with np.load(safe(output, row["file"]), allow_pickle=False) as data:
                    domains.append(
                        dict({k: data[k] for k in data.files}, component_id=row["component_id"])
                    )
            return dict(identity=identity, domains=domains, root=str(output))
        domains, records, files = [], [], []
        topology_refinements = 0
        for component, rows in enumerate(face_components(f, np.arange(len(f)))):
            pending = [rows]
            while pending:
                guard()
                rows = pending.pop()
                if scope == "patches" and len(rows) > max_patch_faces:
                    pending.extend(reversed(split_faces(v, f, rows)))
                    continue
                cv, cf, vi = local_chart(v, f, rows)
                if scope == "patches" and len(rows) > 1 and not igl.is_vertex_manifold(cf).all():
                    topology_refinements += 1
                    pending.extend(reversed(split_faces(v, f, rows)))
                    continue
                cut, topology = cut_component(cv, cf, guard)
                cut["parent"] = vi[cut["parent"]]
                cut["source_face_rows"] = rows
                cut["domain_id"] = np.array(len(domains), np.int64)
                name = f"domain-{len(domains):06d}.npz"
                np.savez_compressed(safe(output, name), **cut)
                files.append(name)
                domains.append(dict(cut, component_id=component))
                records.append(dict(file=name, component_id=component, topology=topology))
        owned = np.concatenate([d["source_face_rows"] for d in domains])
        np.testing.assert_array_equal(np.sort(owned), np.arange(len(f)))
        seal(
            output,
            "DOMAINS.json",
            identity,
            dict(domains=records, topology_refinements=topology_refinements),
            files,
        )
    return dict(identity=identity, domains=domains, root=str(output))


def assemble_map(case, domains, solved, output):
    from winding_field.flatten import cross2

    source = case_arrays(case)
    v, f = source["world_um"], source["faces"]
    if len(solved) != len(domains["domains"]):
        raise ValueError("every domain needs an explicit solver outcome")
    parts, omitted = [], []
    for cut, result in zip(domains["domains"], solved):
        rows = cut["source_face_rows"]
        if result["status"] != "COMPLETE":
            omitted.extend(rows.tolist())
            continue
        uv = np.asarray(result["uv"], float).copy()
        if uv.shape != (len(cut["vertices"]), 2) or not np.isfinite(uv).all():
            raise ValueError("solver returned invalid UV")
        np.testing.assert_array_equal(cut["vertices"], v[cut["parent"]])
        np.testing.assert_array_equal(cut["parent"][cut["faces"]], f[rows])
        signed = cross2(
            uv[cut["faces"][:, 1]] - uv[cut["faces"][:, 0]],
            uv[cut["faces"][:, 2]] - uv[cut["faces"][:, 0]],
        )
        if signed.sum() < 0:
            uv[:, 0] *= -1
        parts.append(dict(cut, uv=uv, receipt=result["receipt"]))
    calibration = []
    for component in sorted({p["component_id"] for p in parts}):
        selected = [p for p in parts if p["component_id"] == component]
        # Shared original edges count once; average their mapped seam occurrences.
        keys, lengths = [], []
        for p in selected:
            edges = np.unique(
                np.sort(
                    np.concatenate(
                        [p["faces"][:, [0, 1]], p["faces"][:, [1, 2]], p["faces"][:, [2, 0]]]
                    ),
                    axis=1,
                ),
                axis=0,
            )
            keys.append(np.sort(p["parent"][edges], axis=1))
            lengths.append(np.linalg.norm(p["uv"][edges[:, 1]] - p["uv"][edges[:, 0]], axis=1))
        edges, inverse = np.unique(np.concatenate(keys), axis=0, return_inverse=True)
        lengths = np.concatenate(lengths)
        if not np.isfinite(lengths).all() or (lengths <= 0).any():
            raise ValueError("collapsed/nonfinite edge cannot be calibrated")
        means = np.bincount(inverse, weights=lengths) / np.bincount(inverse)
        scale = float(np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1).sum() / means.sum())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid component calibration")
        calibration.append(
            dict(component_id=component, scale=scale, independent_patch_scaling=False)
        )
        for p in selected:
            p["uv"] *= scale
    packed = {k: [] for k in ("world", "faces", "uv", "parent", "source_face_rows", "domain_ids")}
    offset, nv = 0.0, 0
    for p in parts:
        uv = p["uv"]
        uv -= uv.min(0)
        uv += [offset, 0.0]
        offset = float(uv[:, 0].max()) + 2 * case["ct_profile"]["pitch_um"]
        packed["world"].append(p["vertices"])
        packed["faces"].append(p["faces"] + nv)
        packed["uv"].append(uv)
        packed["parent"].append(p["parent"])
        packed["source_face_rows"].append(p["source_face_rows"])
        packed["domain_ids"].append(np.full(len(p["faces"]), int(p["domain_id"]), np.int64))
        nv += len(uv)
    empty = dict(
        world=np.empty((0, 3)),
        faces=np.empty((0, 3), np.int64),
        uv=np.empty((0, 2)),
        parent=np.empty(0, np.int64),
        source_face_rows=np.empty(0, np.int64),
        domain_ids=np.empty(0, np.int64),
    )
    mapped = {k: np.concatenate(values) if values else empty[k] for k, values in packed.items()}
    mapped["omitted_face_rows"] = np.asarray(omitted, np.int64)
    np.testing.assert_array_equal(
        np.sort(np.r_[mapped["source_face_rows"], mapped["omitted_face_rows"]]), np.arange(len(f))
    )
    report = dict(
        calibration=calibration,
        original_XYZ_preserved=True,
        physical_acceptance=False,
        outcomes=[dict(status=r["status"], receipt=r["receipt"]) for r in solved],
    )
    output = Path(output)
    with locked(output):
        np.savez_compressed(output / "map.npz", **mapped)
        seal(
            output,
            "MAP.json",
            dict(case=case["source_sha256"], domains=domains["identity"]),
            report,
            ["map.npz"],
        )
    return dict(mapped, report=report)


METHODS = ("legacy-lscm", "lscm", "mvc", "arap", "slim", "arap-slim", "abf", "bff", "particles")


def external_uv(cut, data):
    """Accept explicit index correspondence only; geometry edits are not flattening."""
    v, f = cut["vertices"], cut["faces"]
    world, tri = np.asarray(data["world"]), np.asarray(data["faces"])
    parent, rows = np.asarray(data["parent"]), np.asarray(data["source_face_rows"])
    uv = np.asarray(data["uv"], float)
    valid = (
        parent.dtype.kind in "iu"
        and rows.dtype.kind in "iu"
        and tri.dtype.kind in "iu"
        and parent.shape == (len(v),)
        and rows.shape == (len(f),)
        and world.shape == v.shape
        and tri.shape == f.shape
        and uv.shape == (len(v), 2)
        and np.array_equal(np.sort(parent), np.arange(len(v)))
        and np.array_equal(np.sort(rows), np.arange(len(f)))
        and tri.min() >= 0
        and tri.max() < len(v)
    )
    if (
        not valid
        or not np.array_equal(world, v[parent])
        or not np.array_equal(parent[tri], f[rows])
    ):
        raise ValueError("external geometry/correspondence changed")
    if not np.isfinite(uv).all():
        raise ValueError("external UV nonfinite")
    result = np.empty_like(uv)
    result[parent] = uv
    return result


def solve_domain(method, cut, output, backend, guard):
    from winding_field import native_global as n
    from winding_field.flattening_matrix import solve
    from winding_field.util import file_sha

    if method not in METHODS or method == "legacy-lscm":
        raise ValueError("unknown domain solver; legacy is a separate saved-map baseline")
    guard()
    output = Path(output)
    if backend.get("status") in ("UNAVAILABLE", "INCOMPATIBLE_GEOMETRY"):
        return dict(status=backend["status"], receipt=dict(method=method, reason=backend["reason"]))
    if method in ("abf", "bff", "particles"):
        if backend.get("status") != "QUALIFIED":
            return dict(
                status="UNAVAILABLE",
                receipt=dict(method=method, reason="external build not qualified"),
            )
        import subprocess
        from winding_field.flatten import write_obj

        command = backend["argv"]
        if (
            not command
            or not backend.get("revision")
            or file_sha(command[0]) != backend["executable_sha256"]
        ):
            raise ValueError("external backend identity changed")
        with locked(output):
            source = output / "input.npz"
            np.savez_compressed(source, world=cut["vertices"], faces=cut["faces"])
            dest = output / "external.npz"
            argv = [
                a.replace("{input}", str(source.absolute())).replace(
                    "{output}", str(dest.absolute())
                )
                for a in command
            ]
            with (
                (output / "stdout.log").open("wb") as stdout,
                (output / "stderr.log").open("wb") as stderr,
            ):
                child = subprocess.Popen(argv, stdout=stdout, stderr=stderr)
                try:
                    while child.poll() is None:
                        guard()
                        try:
                            child.wait(timeout=0.2)
                        except subprocess.TimeoutExpired:
                            pass
                    if child.returncode:
                        raise RuntimeError("external solver exit " + str(child.returncode))
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait()
            with np.load(dest, allow_pickle=False) as data:
                uv = external_uv(cut, {k: data[k] for k in data.files})
        receipt = dict(method=method, backend=backend, argv=argv)
    elif method == "lscm":
        uv, details = solve(cut, output, dict(flattening_solver="lscm"), guard)
        receipt = dict(method=method, details=details)
    else:
        uv, initialization = n.mean_value_map(cut, guard)
        receipt = dict(method=method, initialization=initialization)
        if method in ("arap", "arap-slim"):
            uv, receipt["arap"] = n.arap_map(cut, uv, guard)
        if method in ("slim", "arap-slim"):
            from types import SimpleNamespace
            from winding_field import flatboi_checkpoint as native

            try:
                n.validate_map(dict(world=cut["vertices"], faces=cut["faces"], uv=uv))
            except ValueError as error:
                raise ValueError("invalid SLIM seed: " + str(error)) from error
            if not backend.get("app_run"):
                return dict(
                    status="UNAVAILABLE",
                    receipt=dict(method=method, reason="pinned Flatboi unavailable"),
                )
            with locked(output):
                source, obj = output / "initial-map.npz", output / "inputmesh.obj"
                # A resumed seed must not replace the exact initial numerical payload.
                if source.exists():
                    with np.load(source, allow_pickle=False) as saved:
                        for key, value in (
                            ("world", cut["vertices"]),
                            ("faces", cut["faces"]),
                            ("uv", uv),
                        ):
                            np.testing.assert_array_equal(saved[key], value)
                else:
                    np.savez_compressed(source, world=cut["vertices"], faces=cut["faces"], uv=uv)
                if not obj.exists():
                    # The checkpoint binder requires exact float64 XYZ, not write_obj's 12 digits.
                    pending = output / "inputmesh.obj.partial"
                    with pending.open("w") as stream:
                        for values, fmt in (
                            (cut["vertices"], "v %.17g %.17g %.17g"),
                            (uv, "vt %.17g %.17g"),
                        ):
                            for first in range(0, len(values), 100000):
                                guard()
                                np.savetxt(stream, values[first : first + 100000], fmt=fmt)
                        for first in range(0, len(cut["faces"]), 100000):
                            guard()
                            np.savetxt(
                                stream,
                                np.repeat(cut["faces"][first : first + 100000] + 1, 2, axis=1),
                                fmt="f %d/%d %d/%d %d/%d",
                            )
                    pending.replace(obj)
                result = native.run(
                    SimpleNamespace(
                        source_map=source,
                        input=obj,
                        output=output / "native",
                        app_run=backend["app_run"],
                        native_recipe=backend["native_recipe"],
                        batch_iterations=10,
                        iterations=200,
                        hours=1.0,
                    ),
                    guard=guard,
                )
                checkpoint = result["checkpoint"]
                uv, binding = native.positive_uv(
                    output / "native" / checkpoint["obj"], cut["vertices"], cut["faces"], guard
                )
            receipt.update(native=result, binding=binding)
    guard()
    if uv.shape != (len(cut["vertices"]), 2) or not np.isfinite(uv).all():
        raise ValueError("solver returned invalid UV")
    return dict(status="COMPLETE", uv=uv, receipt=receipt)


def guard_output(root, *, reserve_bytes=20 << 30, pending_bytes=0):
    root = Path(root)
    while not root.exists():
        root = root.parent
    if shutil.disk_usage(root).free < reserve_bytes + pending_bytes:
        raise InterruptedError("disk reserve would be violated")


def supervise(command, guard, *, unit=None, slice_name=None, timeout=None):
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise TimeoutError("parent cell wall time budget exhausted")
    deadline = None if timeout is None else time.monotonic() + timeout
    if unit is not None:
        if not re.fullmatch(r"field-flattening-cell-[a-zA-Z0-9_-]+", unit):
            raise ValueError("refusing non-benchmark unit ownership")
        command = [
            "systemd-run",
            "--unit=" + unit,
            "--wait",
            "--pipe",
            "--collect",
            "--property=KillMode=control-group",
            "--property=TimeoutStopSec=5",
            "--property=CPUQuota=200%",
            "--property=CPUWeight=10",
            "--property=Nice=10",
            "--property=MemoryMax=20G",
            "--property=MemorySwapMax=0",
            *(["--property=RuntimeMaxSec=" + str(timeout + 1)] if timeout is not None else []),
            *(["--slice=" + slice_name] if slice_name else []),
            *[
                "--setenv=" + key + "=" + os.environ[key]
                for key in (
                    "PYTHONPATH",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                )
                if key in os.environ
            ],
            *command,
        ]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while True:
            guard()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("parent cell wall time budget exhausted")
            try:
                result = process.wait(timeout=0.2)
                if result:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("parent cell wall time budget exhausted")
                    raise subprocess.CalledProcessError(result, command)
                return
            except subprocess.TimeoutExpired:
                pass
    finally:
        if unit is not None:
            # A native child may have its own session; the owned cgroup still contains it.
            for action in (["kill", "--kill-whom=all", "--signal=SIGKILL"], ["stop"]):
                subprocess.run(
                    ["systemctl", *action, unit],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def require_benchmark_envelope(cgroup_root=Path("/sys/fs/cgroup"), proc=Path("/proc")):
    """Require an aggregate benchmark slice, not independent two-CPU child limits."""
    try:
        relative = (proc / "self/cgroup").read_text().strip().removeprefix("0::/")
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("invalid cgroup location")
        group = cgroup_root / relative
        controlled = None
        for path in (group, *group.parents):
            if path == cgroup_root or cgroup_root not in path.parents:
                break
            if not path.name.startswith("field-flattening-") or not path.name.endswith(".slice"):
                continue
            quota, period = map(int, (path / "cpu.max").read_text().split())
            ram = int((path / "memory.max").read_text())
            swap = int((path / "memory.swap.max").read_text())
            if not 0 < quota <= 2 * period or not 0 < ram <= 20 << 30 or swap != 0:
                raise ValueError("unsafe aggregate envelope")
            controlled = dict(slice=path.name, memory_max_bytes=ram)
            break
        if controlled is None:
            raise ValueError("bounded aggregate benchmark slice missing")
        nice = int((proc / "self/stat").read_text().rsplit(")", 1)[1].split()[16])
        if nice < 10:
            raise ValueError("Nice must be at least10")
        available = (
            int(
                next(
                    line.split()[1]
                    for line in (proc / "meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
            )
            * 1024
        )
        if available < controlled["memory_max_bytes"] + (4 << 30):
            raise ValueError("host RAM headroom below requested envelope plus4GiB")
        return controlled
    except (OSError, ValueError, StopIteration, IndexError) as error:
        raise ValueError("unsafe benchmark envelope/headroom/Nice: " + str(error)) from error


def queue_status(rows):
    if any(r["status"] in ("PENDING", "RUNNING", "PAUSED") for r in rows):
        return "RUNNING"
    return "COMPLETE" if all(r["status"] == "COMPLETE" for r in rows) else "FINISHED_WITH_GAPS"


@contextmanager
def stop_signals(output):
    def stop(sig, frame):
        atomic_json(Path(output) / "STOP", dict(signal=sig))
        raise InterruptedError("worker signal received; retained checkpoints preserved")

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def tool_identity():
    from importlib import import_module
    from importlib.metadata import version

    modules = (
        "native_global",
        "flattening_matrix",
        "flatboi_checkpoint",
        "original_recipe",
        "native_ct_sampling",
        "io",
        "ct_reader",
        "surface_export",
        "flattening.obj",
        "flattening.topology",
        "surface_atlas",
        "patch_pipeline",
        "patch_mapping",
        "native_patches",
        "tifxyz_guide",
        "surface_mesh",
        "surface_sources",
        "remote_field",
        "range_surface",
        "flatten",
        "sheet_atlas",
        "native_mesh",
        "native_cache_index",
        "util",
    )
    return dict(
        files={
            name: file_sha(Path(__file__).with_name(name + ".py"))
            for name in (
                "bench_flattening",
                "flattening_metrics",
                "flattening_ct",
                "flattening_review",
            )
        },
        implementations={
            name: file_sha(import_module("winding_field." + name).__file__) for name in modules
        },
        numerical_store=file_sha(import_module("field_store").__file__),
        runtime={
            name: version(name) for name in ("numpy", "scipy", "libigl", "Pillow", "vc-reader")
        },
    )


def run_cell(request, output):
    from flattening_ct import CheckedCT, prepare_rois, render_case
    from flattening_metrics import measure

    output = Path(output)
    case = request["case"]
    identity = dict(request=request, code=tool_identity())
    start = time.monotonic()
    last_scratch_check = -float("inf")
    active = (
        read_json(output / "TIME.json")["active_seconds"]
        if (output / "TIME.json").exists()
        else 0.0
    )

    def guard():
        nonlocal last_scratch_check
        if (output / "STOP").exists() or (
            request.get("run") and (Path(request["run"]) / "STOP").exists()
        ):
            raise InterruptedError("STOP requested")
        if active + time.monotonic() - start > request["seconds"]:
            raise TimeoutError("active cell time budget exhausted")
        guard_output(output, reserve_bytes=request["reserve_bytes"])
        guard_output(request["ct_config"]["cache_dir"], reserve_bytes=request["reserve_bytes"])
        if request.get("protected_inference"):
            protect_inference(request["protected_inference"])
        if time.monotonic() - last_scratch_check >= 5:
            if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > request.get(
                "scratch_bytes", 20 << 30
            ):
                raise MemoryError("cell scratch budget exceeded; products retained")
            last_scratch_check = time.monotonic()

    guard()
    with locked(output):
        if (output / "REQUEST.json").exists():
            if read_json(output / "REQUEST.json") != identity:
                raise ValueError("cell request/code identity changed")
        else:
            atomic_json(output / "REQUEST.json", identity)
        case_arrays(case)
        if (output / "CELL.json").exists():
            record = verify(output, "CELL.json", identity)
            if record["report"]["status"] == "COMPLETE":
                with CheckedCT(request["ct_config"], output / "ct/ct-ledger"):
                    pass
            return record["report"]
        try:
            if request["backend"].get("status") in ("UNAVAILABLE", "INCOMPATIBLE_GEOMETRY"):
                result = dict(
                    status=request["backend"]["status"],
                    outcomes=[dict(method=request["method"], reason=request["backend"]["reason"])],
                )
                seal(output, "CELL.json", identity, result, ["REQUEST.json"])
                return result
            shared = Path(request["shared"]) / case["id"]
            method = request["method"]
            maproot = output / "map"
            if (maproot / "MAP.json").exists():
                saved = verify(maproot, "MAP.json")
                with np.load(maproot / "map.npz", allow_pickle=False) as data:
                    mapped = dict({k: data[k] for k in data.files}, report=saved["report"])
            elif method == "legacy-lscm":
                a = case_arrays(case)
                if case["field_id"] == "175a" and not request.get("legacy_map"):
                    raise ValueError("175a legacy baseline requires its hash-bound saved JOB map")
                if request.get("legacy_map"):
                    path = Path(request["legacy_map"])
                    if file_sha(path) != request["legacy_map_sha256"]:
                        raise ValueError("legacy UV hash changed")
                    with np.load(path, allow_pickle=False) as data:
                        vi, fi = a["source_vertex_ids"], a["source_face_ids"]
                        parent = np.searchsorted(vi, data["source_vertex_ids"])
                        rows = np.searchsorted(fi, data["source_face_ids"])
                        if (
                            (parent >= len(vi)).any()
                            or (rows >= len(fi)).any()
                            or not np.array_equal(vi[parent], data["source_vertex_ids"])
                            or not np.array_equal(fi[rows], data["source_face_ids"])
                        ):
                            raise ValueError("legacy source IDs unknown")
                        unit = request["legacy_unit_um"]
                        mapped = dict(
                            world=data["world"] * unit,
                            uv=data["uv"] * unit,
                            faces=data["faces"],
                            parent=parent,
                            source_face_rows=rows,
                            domain_ids=np.zeros(len(rows), np.int64),
                        )
                else:
                    from winding_field.original_recipe import make_map

                    mapped = make_map(a["world_um"], a["faces"])
                    mapped["domain_ids"] = np.zeros(len(mapped["faces"]), np.int64)
                with locked(maproot):
                    np.savez_compressed(
                        maproot / "map.npz",
                        **{k: v for k, v in mapped.items() if isinstance(v, np.ndarray)},
                    )
                    seal(maproot, "MAP.json", identity, mapped.get("report", {}), ["map.npz"])
            else:
                domains = prepare_domains(case, shared / request["scope"], request["scope"], guard)
                solved = []
                for i, cut in enumerate(domains["domains"]):
                    guard()
                    root = output / "solvers" / f"{i:06d}"
                    binding = dict(
                        method=method,
                        backend=request["backend"],
                        domains=domains["identity"],
                        domain=i,
                    )
                    if (root / "DOMAIN.json").exists():
                        record = verify(root, "DOMAIN.json", binding)
                        result = dict(record["report"])
                        if result["status"] == "COMPLETE":
                            result["uv"] = np.load(root / "uv.npy", allow_pickle=False)
                    else:
                        result = solve_domain(method, cut, root, request["backend"], guard)
                        with locked(root):
                            files = []
                            if result["status"] == "COMPLETE":
                                np.save(root / "uv.npy", result["uv"], allow_pickle=False)
                                files = ["uv.npy"]
                            seal(
                                root,
                                "DOMAIN.json",
                                binding,
                                {k: v for k, v in result.items() if k != "uv"},
                                files,
                            )
                    solved.append(result)
                if not any(r["status"] == "COMPLETE" for r in solved):
                    result = dict(status="UNAVAILABLE", outcomes=[r["receipt"] for r in solved])
                    seal(output, "CELL.json", identity, result, ["REQUEST.json"])
                    return result
                mapped = assemble_map(case, domains, solved, maproot)
            guard()
            metrics = measure(case, mapped, output / "metrics", guard)
            rois = prepare_rois(case, shared / "rois")
            rendered = render_case(case, mapped, rois, output / "ct", request["ct_config"], guard)
            result = dict(
                status="COMPLETE",
                case_id=case["id"],
                metrics=metrics,
                CT_observed_area_fraction_estimate=rendered["CT_observed_area_fraction_estimate"],
                physical_acceptance=False,
                wall_seconds=time.monotonic() - start,
            )
            files = [
                str(p.relative_to(output))
                for p in output.rglob("*")
                if p.is_file() and p.name not in (".lock", "TIME.json", "CELL.json", "STOP")
            ]
            guard()
            seal(output, "CELL.json", identity, result, files)
            return result
        finally:
            atomic_json(
                output / "TIME.json", dict(active_seconds=active + time.monotonic() - start)
            )


def protect_inference(config):
    root = Path(config["root"])
    if (root / "END.json").exists():
        return
    status = root / "STATUS.json"
    if not status.exists() or time.time() - status.stat().st_mtime > config.get(
        "max_stale_seconds", 900
    ):
        raise InterruptedError(
            "protected inference is stale; benchmark paused, inference unchanged"
        )


def _prepare_source(source, output, reserve_bytes):
    if Path(source["mesh"]).exists():
        return source
    if not source.get("field_config"):
        raise FileNotFoundError(source["mesh"])
    from winding_field.surface_sources import FieldSource
    from winding_field.surface_mesh import extract_mesh

    descriptor = field_descriptor(read_json(source["field_config"]))
    if file_sha(source["field_config"]) != source["field_config_sha256"]:
        raise ValueError("field descriptor changed")
    target = output / "meshes" / (source["field_id"] + "-" + str(source["W"]))
    target.parent.mkdir(exist_ok=True)
    guard_output(target, reserve_bytes=reserve_bytes)
    with FieldSource(descriptor["field"], descriptor["geometry"]) as field:
        result = extract_mesh(
            field,
            target,
            [source["raw_phi"]],
            resume=(target / "RUN.json").exists(),
            reserve_bytes=reserve_bytes,
        )
        if result["status"] != "MESH_READY":
            raise InterruptedError("source mesh preparation paused")
    return dict(
        source, mesh=str(target.absolute()), mesh_manifest_sha256=file_sha(target / "MESH.json")
    )


def inventory(config, output):
    output = Path(output).absolute()
    if set(s["field_id"] for s in config["sources"]) != {"175a", "paris4"}:
        raise ValueError("both 175a and paris4 source profiles required")
    if (
        not config["methods"]
        or set(config["methods"]) - set(METHODS)
        or set(config["scopes"]) - {"sheet", "patches"}
    ):
        raise ValueError("unknown benchmark method/scope")
    identity = dict(schema="flattening-benchmark-v1", config=config, code=tool_identity())
    with locked(output):
        if (output / "RUN.json").exists():
            if read_json(output / "RUN.json") != identity:
                raise ValueError("run identity changed")
            if (output / "INVENTORY.json").exists():
                retained = read_json(output / "INVENTORY.json")
                for entry in retained["entries"]:
                    if entry["status"] == "READY":
                        case_arrays(entry["case"])
                return retained
        else:
            atomic_json(output / "RUN.json", identity)
        atomic_json(output / "BACKENDS.json", config.get("backends", {}))
        entries, rows = [], []
        for i, source in enumerate(config["sources"]):
            guard_output(output, reserve_bytes=config["resources"]["reserve_bytes"])
            try:
                prepared = _prepare_source(source, output, config["resources"]["reserve_bytes"])
                case = load_case(prepared, output / "cases" / f"{i:04d}")
                entries.append(
                    dict(case=case, stage=source["stage"], source=prepared, status="READY")
                )
            except FileNotFoundError as error:
                entries.append(
                    dict(
                        source=source,
                        status="MISSING_INPUT",
                        reason=str(error),
                        stage=source["stage"],
                    )
                )
                w = source["W"]
                label = ("p" if w >= 0 else "m") + (
                    f"{abs(w):03d}" if type(w) is int else str(abs(w)).replace(".", "_")
                )
                case = dict(source, id=f"{source['field_id']}-{source['resolution_id']}-W{label}")
            for scope in config["scopes"]:
                for method in config["methods"]:
                    if method == "legacy-lscm" and scope != config["scopes"][0]:
                        continue
                    rows.append(
                        dict(
                            case_id=case["id"],
                            field_id=case["field_id"],
                            resolution_id=case["resolution_id"],
                            W=case["W"],
                            method=method,
                            scope=scope,
                            track="A",
                            stage=source["stage"],
                            status="PENDING"
                            if entries[-1]["status"] == "READY"
                            else "MISSING_INPUT",
                            output=f"cells/{case['id']}/A/{scope}/{method}",
                            case_index=len(entries) - 1,
                        )
                    )
        report = dict(fields=["175a", "paris4"], entries=entries)
        atomic_json(output / "INVENTORY.json", report)
        atomic_json(
            output / "SELECTION.json",
            dict(
                cases=[
                    dict(field=s["field_id"], W=s["W"], stage=s["stage"]) for s in config["sources"]
                ]
            ),
        )
        atomic_json(output / "QUEUE.json", dict(rows=rows, active_stage="pilot", status="RUNNING"))
    return report


def run_queue(run, stage, *, workers=False):
    run = Path(run).absolute()
    freeze = validate_freeze(run) if stage == "holdout" else None
    saved = read_json(run / "RUN.json")
    if saved["code"] != tool_identity():
        raise ValueError("frozen benchmark runtime/code changed; use a new run")
    config = saved["config"]
    envelope = require_benchmark_envelope() if workers else None
    with locked(run / "queue-lock"):
        queue = read_json(run / "QUEUE.json")
        entries = read_json(run / "INVENTORY.json")["entries"]
        queue["active_stage"] = stage
        chosen = [
            r
            for r in queue["rows"]
            if r["stage"] == stage or (stage == "main" and r["stage"] == "pilot")
        ]
        for row in chosen:
            if row["status"] == "MISSING_INPUT":
                continue
            if stage == "holdout" and [row["method"], row["scope"]] not in freeze["candidates"]:
                row["status"] = "NOT_SELECTED"
                continue
            if workers:
                try:
                    envelope = require_benchmark_envelope()
                except ValueError as error:
                    row.update(status="PAUSED", reason=str(error))
                    atomic_json(run / "QUEUE.json", queue)
                    break
            case = entries[row["case_index"]]["case"]
            source = entries[row["case_index"]]["source"]
            resources = config["resources"]
            request = dict(
                case=case,
                method=row["method"],
                scope=row["scope"],
                track=row["track"],
                run=str(run),
                backend=config.get("backends", {}).get(row["method"], {}),
                shared=str(run / "shared"),
                reserve_bytes=resources["reserve_bytes"],
                seconds=resources["seconds"],
                ct_config=dict(
                    profile=case["ct_profile"],
                    cache_dir=str(run / "ct-cache" / case["field_id"]),
                    cache_bytes=resources["cache_bytes"],
                    decoded_cache_bytes=resources["decoded_cache_bytes"],
                    reserve_bytes=resources["reserve_bytes"],
                ),
            )
            if config.get("protected_inference"):
                request["protected_inference"] = config["protected_inference"]
            if row["method"] == "legacy-lscm" and source.get("legacy_map"):
                for key in ("legacy_map", "legacy_map_sha256", "legacy_unit_um"):
                    request[key] = source[key]
            output = safe(run, row["output"])
            prior_active = (
                read_json(output / "TIME.json")["active_seconds"]
                if (output / "TIME.json").exists()
                else 0.0
            )
            remaining = resources["seconds"] - prior_active
            if workers and remaining <= 0:
                row.update(status="RESOURCE_LIMIT", reason="parent cell wall time budget exhausted")
                atomic_json(run / "QUEUE.json", queue)
                continue
            row["status"] = "RUNNING"
            atomic_json(run / "QUEUE.json", queue)
            worker_started = time.monotonic()
            try:
                if workers:
                    output.mkdir(parents=True, exist_ok=True)
                    atomic_json(output / "WORKER-REQUEST.json", request)
                    supervise(
                        [
                            sys.executable,
                            "-u",
                            __file__,
                            "worker",
                            "--run",
                            str(run),
                            "--output",
                            str(output),
                        ],
                        lambda: (
                            protect_inference(config["protected_inference"])
                            if config.get("protected_inference")
                            else None
                        ),
                        unit="field-flattening-cell-" + uuid.uuid4().hex,
                        slice_name=envelope["slice"],
                        timeout=remaining,
                    )
                    row["status"] = verify(output, "CELL.json")["report"]["status"]
                else:
                    row["status"] = run_cell(request, output)["status"]
            except InterruptedError as error:
                row.update(status="PAUSED", reason=str(error))
                atomic_json(run / "QUEUE.json", queue)
                break
            except TimeoutError as error:
                row.update(status="RESOURCE_LIMIT", reason=str(error))
                atomic_json(
                    output / "TIME.json",
                    dict(
                        active_seconds=max(
                            resources["seconds"], prior_active + time.monotonic() - worker_started
                        )
                    ),
                )
            except subprocess.CalledProcessError as error:
                receipt = output / "WORKER-EXIT.json"
                status = read_json(receipt) if receipt.exists() else {}
                expected = {75: "PAUSED", 76: "RESOURCE_LIMIT"}.get(error.returncode)
                if (
                    expected
                    and status.get("status") == expected
                    and status.get("request_sha256") == digest(request)
                ):
                    row.update(status=expected, reason=status["reason"])
                    if expected == "PAUSED":
                        atomic_json(run / "QUEUE.json", queue)
                        break
                else:
                    row.update(status="FAILED", reason=str(error))
                    atomic_json(
                        output / "FAILED.json", dict(error=str(error), type=type(error).__name__)
                    )
            except (
                ValueError,
                RuntimeError,
                OSError,
                MemoryError,
                subprocess.SubprocessError,
            ) as error:
                row.update(
                    status="RESOURCE_LIMIT" if isinstance(error, MemoryError) else "FAILED",
                    reason=str(error),
                )
                output.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    output / "FAILED.json", dict(error=str(error), type=type(error).__name__)
                )
            finally:
                if workers:
                    timing = output / "TIME.json"
                    recorded = (
                        read_json(timing)["active_seconds"] if timing.exists() else prior_active
                    )
                    atomic_json(
                        timing,
                        dict(
                            active_seconds=max(
                                recorded, prior_active + time.monotonic() - worker_started
                            )
                        ),
                    )
            atomic_json(run / "QUEUE.json", queue)
        missing = [
            e
            for e in entries
            if e["status"] != "READY" and (e["stage"] == stage or stage == "main")
        ]
        queue["status"] = "FINISHED_WITH_GAPS" if missing else queue_status(chosen)
        atomic_json(run / "QUEUE.json", queue)
        if queue["status"] != "RUNNING":
            atomic_json(
                run / ("END-" + stage + ".json"),
                dict(
                    status=queue["status"],
                    rows=chosen,
                    missing=missing,
                    next_gate="Human CT texture review; do not invent a winner or unfreeze holdout",
                ),
            )
    return queue


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "run", "worker", "verify"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=["pilot", "main", "holdout"], default="pilot")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "inventory":
        if args.config is None or args.output is None:
            parser.error("inventory requires --config and --output")
        result = inventory(read_json(args.config), args.output)
    elif args.action == "worker":
        if args.output is None:
            parser.error("worker requires --output")
        with stop_signals(args.output):
            request = read_json(args.output / "WORKER-REQUEST.json")
            try:
                result = run_cell(request, args.output)
            except (InterruptedError, TimeoutError, MemoryError) as error:
                paused = isinstance(error, InterruptedError)
                atomic_json(
                    args.output / "WORKER-EXIT.json",
                    dict(
                        status="PAUSED" if paused else "RESOURCE_LIMIT",
                        reason=str(error),
                        request_sha256=digest(request),
                    ),
                )
                raise SystemExit(75 if paused else 76)
    elif args.action == "run":
        if args.run is None:
            parser.error("run requires --run")
        # Heavy remote execution must have a real hard cgroup, not only a Python guard.
        require_benchmark_envelope()
        result = run_queue(args.run, args.stage, workers=True)
    else:
        if args.run is None:
            parser.error("verify requires --run")
        saved = read_json(args.run / "RUN.json")
        if saved["code"] != tool_identity():
            raise ValueError("runtime/code changed")
        for entry in read_json(args.run / "INVENTORY.json")["entries"]:
            if entry["status"] == "READY":
                case_arrays(entry["case"])
        rows = read_json(args.run / "QUEUE.json")["rows"]
        for row in rows:
            if row["status"] == "COMPLETE":
                root = safe(args.run, row["output"])
                record = verify(root, "CELL.json")
                from flattening_ct import CheckedCT

                with CheckedCT(record["identity"]["request"]["ct_config"], root / "ct/ct-ledger"):
                    pass
        result = dict(
            status="VERIFIED_RETAINED_ARTIFACTS", cells=len(rows), physical_acceptance=False
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
