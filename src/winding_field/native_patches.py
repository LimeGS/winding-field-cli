"""Verified native replay and Linux process-tree bounds for patch exports."""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import threading
import time

import numpy as np

from .surface_export import read_flatboi_uv, scale_uv_to_world, validate_uv
from .util import atomic_json, file_sha


def admit_memory(required_bytes, *, phase, operator_bytes=None):
    """Observe additional allocation headroom; admission is not a reservation."""
    from field_runtime import ResourceUnavailable
    from .tifxyz_runtime import memory_report

    if type(required_bytes) is not int or required_bytes < 0:
        raise ValueError("required memory must be nonnegative bytes")
    if operator_bytes is not None and (type(operator_bytes) is not int or operator_bytes <= 0):
        raise ValueError("operator memory budget must be positive bytes")
    if sys.platform.startswith("linux"):
        memory = memory_report()
        for bound in memory["bounds"]:
            path = Path(bound["path"]) / "memory.high"
            try:
                high = path.read_text().strip()
                if high != "max":
                    high = int(high)
                    if high < 0:
                        raise ValueError("negative memory.high")
                    head = max(0, high - bound["usage"] + bound.get("reclaim", 0))
                    bound["high_headroom"] = head
                    memory["effective_headroom_bytes"] = min(
                        memory["effective_headroom_bytes"], head
                    )
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as error:
                memory["known"] = False
                memory["errors"].append(str(error))
    else:
        from .original_pipeline import available_memory

        available = available_memory()
        memory = dict(
            known=available is not None,
            effective_headroom_bytes=available or 0,
            bounds=[],
            errors=[],
        )
    available = memory["effective_headroom_bytes"] if memory["known"] else 0
    limits = []
    for kind, key in (("RLIMIT_AS", "VmSize"), ("RLIMIT_DATA", "VmData")):
        cap = min(
            (n for n in resource.getrlimit(getattr(resource, kind)) if n != resource.RLIM_INFINITY),
            default=None,
        )
        if cap is not None:
            try:
                status = Path("/proc/self/status").read_text()
                used = int(re.search(r"^" + key + r":\s+(\d+)\s+kB\s*$", status, re.M)[1]) * 1024
            except (OSError, TypeError):
                used = None
            head = max(0, cap - used) if used is not None else 0
            limits.append(dict(kind=kind, limit=cap, usage=used, headroom=head))
            available = min(available, head)
    if operator_bytes is not None:
        available = min(available, operator_bytes)
    report = dict(
        phase=phase,
        required_bytes=required_bytes,
        available_bytes=available,
        memory=memory,
        process_limits=limits,
        operator_bytes=operator_bytes,
        admitted=memory["known"] and available > 0 and required_bytes <= available,
    )
    if not report["admitted"]:
        error = ResourceUnavailable(
            f"{phase}: requires {required_bytes} additional bytes; "
            f"available {available} bytes under effective memory limits"
        )
        error.report = report
        raise error
    return report


def process_tree_rss():
    """Include descendant process groups: native children start new sessions."""
    if not Path("/proc/self/stat").exists():
        raise InterruptedError("parallel native RSS guard requires Linux /proc")
    rows = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            rows[int(path.parent.name)] = (
                int(fields[1]),
                int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
    owned = {os.getpid()}
    while True:
        children = {pid for pid, (parent, rss) in rows.items() if parent in owned} - owned
        if not children:
            break
        owned.update(children)
    return sum(rows[pid][1] for pid in owned if pid in rows)


class ResourceGuard:
    def __init__(self, path, cap_gib, reserve_gib):
        self.path, self.cap, self.reserve = (
            path,
            None if cap_gib is None else cap_gib * 2**30,
            reserve_gib * 2**30,
        )
        self.lock = threading.Lock()
        self.last = 0.0
        self.peak = 0

    def __call__(self):
        with self.lock:
            now = time.monotonic()
            if now - self.last < 0.1:
                return
            self.last = now
            rss = process_tree_rss()
            self.peak = max(self.peak, rss)
            if self.cap is not None and rss > self.cap:
                raise InterruptedError("native process-tree RSS cap exceeded")
            if self.cap is None:
                admit_memory(0, phase="native process-tree observation")
            if shutil.disk_usage(self.path).free < self.reserve + (64 << 20):
                raise InterruptedError("native disk reserve reached")


class NativeReuse:
    """Snapshot completed receipts, then validate and copy each exact input on use.

    Failed/incomplete attempts are counted, never presented as reusable UVs.
    The caller's new identity pins this inventory before launching any child.
    """

    def __init__(self, root, identity_sha256, identity):
        self.root = Path(root).expanduser().resolve()
        self.identity_sha256 = identity_sha256
        if file_sha(self.root / "IDENTITY.json") != identity_sha256:
            raise ValueError("reuse identity SHA256 mismatch")
        old = json.loads((self.root / "IDENTITY.json").read_bytes())
        for key in ("source", "mesh_sha256", "native", "runtime"):
            if old.get(key) != identity.get(key):
                raise ValueError("reuse " + key + " identity differs")
        for key in ("output_frame", "max_patch_faces", "max_depth", "timeout_seconds"):
            if old["settings"].get(key) != identity["settings"].get(key):
                raise ValueError("reuse configuration differs: " + key)
        previous_budget = old["settings"].get("max_charts")
        budget = identity["settings"].get("max_charts")
        # A larger attempt budget extends DFS without changing an exact-input native result.
        if (
            type(previous_budget) is not int
            or type(budget) is not int
            or not 1 <= previous_budget <= budget <= 100000
        ):
            raise ValueError("reuse chart budget must not decrease")
        self.native = identity["native"]
        self.index = {}
        self.count = 0
        self.lock = threading.Lock()
        self.inventory = dict(complete=0, failed=0, incomplete=0, receipts={})
        for directory in sorted(self.root.glob("level-*/patches/patch-*")):
            receipt = directory / "receipt.json"
            if not receipt.is_file():
                self.inventory["incomplete"] += 1
                continue
            if directory.is_symlink() or receipt.is_symlink():
                raise ValueError("reuse symlink is not an immutable artifact")
            data = json.loads(receipt.read_bytes())
            digest = file_sha(receipt)
            if (
                data.get("returncode") != 0
                or data.get("timed_out")
                or not data.get("output_obj_sha256")
                or "scaling" not in data
            ):
                self.inventory["failed"] += 1
                continue
            self.validate_recipe(data)
            level = int(directory.parent.parent.name.removeprefix("level-"))
            self.index.setdefault((level, data["input_mesh_sha256"]), []).append(
                (directory, digest)
            )
            self.inventory["receipts"][str(receipt.relative_to(self.root))] = digest
            self.inventory["complete"] += 1

    def validate_recipe(self, data):
        if data.get("method") != "native_flatboi":
            raise ValueError("reuse native method differs")
        for key in ("iterations", "energy", "relative_energy_tolerance"):
            if data.get(key) != self.native[key]:
                raise ValueError("reuse native recipe differs: " + key)
        if (
            data.get("app_run_sha256") != self.native["launcher_sha256"]
            or data.get("expected_app_run_sha256") != self.native["launcher_sha256"]
        ):
            raise ValueError("reuse native launcher differs")
        command = data.get("command", [])
        if (
            len(command) != 6
            or command[1] != "flatboi"
            or Path(command[2]).name != "inputmesh.obj"
            or command[3:] != ["200", "symmetric_dirichlet", "--tol=1e-6"]
        ):
            raise ValueError("reuse native command differs")

    def get(self, level, v, f, destination):
        stream = io.StringIO()
        np.savetxt(stream, v, fmt="v %.17g %.17g %.17g")
        np.savetxt(stream, np.asarray(f) + 1, fmt="f %d %d %d")
        digest = hashlib.sha256(stream.getvalue().encode()).hexdigest()
        matches = self.index.get((level, digest))
        if not matches:
            return None
        if (
            len(
                {
                    json.loads((path / "receipt.json").read_bytes())["output_obj_sha256"]
                    for path, _ in matches
                }
            )
            > 1
        ):
            raise ValueError(
                "ambiguous saved native outputs for the same input; chart-specific provenance required"
            )
        source, receipt_sha = matches[0]
        receipt_path = source / "receipt.json"
        if (
            file_sha(self.root / "IDENTITY.json") != self.identity_sha256
            or file_sha(receipt_path) != receipt_sha
        ):
            raise ValueError("reuse provenance changed")
        data = json.loads(receipt_path.read_bytes())
        self.validate_recipe(data)
        artifacts = data["artifacts"]
        for name, sha in (
            ("inputmesh.obj", digest),
            ("inputmesh_flatboi.obj", data["output_obj_sha256"]),
        ):
            if artifacts.get(name, {}).get("sha256") != sha:
                raise ValueError("reuse artifact receipt mismatch")
        for name, record in artifacts.items():
            path = source / name
            if (
                Path(name).name != name
                or path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != record["bytes"]
                or file_sha(path) != record["sha256"]
            ):
                raise ValueError("reuse artifact changed: " + name)
        raw, binding = read_flatboi_uv(source / "inputmesh_flatboi.obj", v, f)
        uv, scaling = scale_uv_to_world(v, f, raw)
        gate = validate_uv(v, f, uv)
        if (
            binding != data.get("binding")
            or scaling != data.get("scaling")
            or gate != data.get("uv_gate")
            or data.get("status") != gate["status"]
            or data.get("reasons") != gate["reasons"]
        ):
            raise ValueError("reuse UV validation differs")
        destination = Path(destination)
        destination.mkdir(exist_ok=False)
        for name, record in artifacts.items():
            shutil.copyfile(source / name, destination / name)
            if file_sha(destination / name) != record["sha256"]:
                raise ValueError("reuse artifact changed while copying")
        shutil.copyfile(receipt_path, destination / "source-receipt.json")
        if file_sha(destination / "source-receipt.json") != receipt_sha:
            raise ValueError("reuse receipt changed while copying")
        data = dict(
            data,
            reused_from=dict(
                export=str(self.root),
                identity_sha256=self.identity_sha256,
                receipt=str(receipt_path),
                receipt_sha256=receipt_sha,
            ),
        )
        atomic_json(destination / "receipt.json", data)
        with self.lock:
            self.count += 1
        return uv, data
