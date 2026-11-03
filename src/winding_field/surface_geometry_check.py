#!/usr/bin/env python3
"""Export a declared-world pointmap, then run the unchanged geometry gate.

No new intersection algorithm. --output must be new; use run_watch.py timeout180.
The underlying gate's exit 0 can mean UNMEASURED, so inspect its receipt.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

from dataset_config import frame as validate_frame
from dataset_config import positive

TOOLS = ("helena_tifxyz_geometry_gate.py", "helena_audit_mesh_integrity.py")


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def export_tifxyz(grid_path, metadata, output):
    frame = validate_frame(metadata.get("frame"))
    unit = positive(metadata.get("voxel_um"), "voxel_um")
    with np.load(grid_path, allow_pickle=False) as grid:
        points, valid, pitch_um = grid["points_xyz"], grid["valid"], np.asarray(grid["pitch_um"], float)
    if (
        points.ndim != 3
        or points.shape[-1] != 3
        or min(points.shape[:2]) < 2
        or valid.shape != points.shape[:2]
        or valid.dtype != np.bool_
        or not valid.any()
    ):
        raise ValueError("invalid pointmap/boolean validity shape")
    if not np.isfinite(points[valid]).all() or (points[valid] < 0).any():
        raise ValueError("valid cells must contain finite nonnegative XYZ")
    if pitch_um.shape != (2,) or not np.isfinite(pitch_um).all() or (pitch_um <= 0).any():
        raise ValueError("pitch_um must be positive [row,column]")
    encoded = np.full(points.shape, -1, dtype=np.float32)
    encoded[valid] = points[valid]
    if not np.isfinite(encoded).all():
        raise ValueError("coordinates cannot be represented as finite float32")
    error = float(np.abs(encoded[valid].astype(float) - points[valid]).max())
    output = Path(output)
    output.mkdir(exist_ok=False)
    for i, axis in enumerate("xyz"):
        tifffile.imwrite(output / f"{axis}.tif", encoded[..., i], photometric="minisblack")
    meta = dict(
        format="tifxyz",
        type="seg",
        uuid=output.name,
        frame=frame,
        coordinate_units="declared_world_units",
        voxel_um=unit,
        scale=(unit / pitch_um[::-1]).tolist(),
        scale_convention="[column,row] cells per declared world unit",
        shape_hw=list(valid.shape),
        invalid_xyz_sentinel=-1.0,
        source_grid_sha256=sha(grid_path),
    )
    save_json(output / "meta.json", meta)
    receipt = dict(
        frame=frame,
        voxel_um=unit,
        pointmap=str(Path(grid_path).resolve()),
        source_grid_sha256=meta["source_grid_sha256"],
        valid_points=int(valid.sum()),
        invalid_points=int((~valid).sum()),
        pixel_um_row_column=pitch_um.tolist(),
        scale=meta["scale"],
        maximum_float32_error_world=error,
        maximum_float32_error_um=error * unit,
        xyz_transform="identity; no origin/scale applied to XYZ",
        invalid_xyz_sentinel=-1.0,
        files={n: sha(output / n) for n in ("x.tif", "y.tif", "z.tif", "meta.json")},
    )
    save_json(output / "conversion.json", receipt)
    return receipt


def gate_outcome(surface):
    if surface.get("measurement_complete") is not True:
        return "UNMEASURED"
    if surface.get("geometry_qc_state") == "GEOMETRY_CERTIFIED" and surface.get("status") == "PASS":
        return "PASS"
    if surface.get("geometry_qc_state", "").startswith("GEOMETRY_REJECTED_"):
        return "FAIL"
    return "UNMEASURED"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("grid", "source-receipt", "metadata", "registration-report", "tools", "origin-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--gate-sha256", required=True)
    p.add_argument("--policy-sha256", required=True)
    args = p.parse_args(argv)
    args.output.mkdir(exist_ok=False)
    start = time.monotonic()
    result = dict(
        status="UNMEASURED",
        source_sha256=sha(__file__),
        scope="existing pipeline sampled raster-artifact geometry certification",
        inter_lamina_spacing_measured=False,
        resolution_not_measured=True,
        fallback_inter_lamina_um=150.0,
        resolution_note="inter-lamina resolution threshold is unmeasured; actual raster grid step is measured by the gate",
        physical_sheet_acceptance="UNMEASURED",
        writing_face="UNMEASURED",
        ink="NOT_MEASURED",
        non_claims=["not CGAL or whole-mesh exact proof", "not physical-sheet or ink acceptance"],
    )
    try:
        expected = dict(zip(TOOLS, (args.gate_sha256, args.policy_sha256)))
        result["tools"] = {
            name: dict(
                origin_path=str(args.origin_dir / name),
                deployed_path=str((args.tools / name).resolve()),
                sha256=sha(args.tools / name),
            )
            for name in TOOLS
        }
        if any(result["tools"][name]["sha256"] != expected[name] for name in TOOLS):
            raise ValueError("deployed pipeline tool SHA256 differs from source")
        source = json.loads(args.source_receipt.read_text())
        relative = str(args.grid.resolve().relative_to(args.source_receipt.resolve().parent))
        grid_sha = sha(args.grid)
        if source["artifacts"][relative]["sha256"] != grid_sha:
            raise ValueError("pointmap hash does not match original atlas receipt")
        result["inputs"] = {
            str(x): sha(x) for x in (args.grid, args.source_receipt, args.metadata, args.registration_report)
        }
        metadata = json.loads(args.metadata.read_text())
        if (
            source.get("frame", metadata["frame"]) != metadata["frame"]
            or source.get("voxel_um", metadata["voxel_um"]) != metadata["voxel_um"]
        ):
            raise ValueError("pointmap receipt and metadata coordinate frames/units differ")
        result["conversion"] = export_tifxyz(args.grid, metadata, args.output / "tifxyz")
        registration = json.loads(args.registration_report.read_text())
        expected_metadata = registration.get("inputs", {}).get("metadata_sha256")
        if expected_metadata is not None and expected_metadata != sha(args.metadata):
            raise ValueError("registration metadata identity differs")
        result["physical_registration"] = {
            k: registration[k] for k in ("human_seed", "seed", "seams", "validation") if k in registration
        }
        gate_json = args.output / "gate.json"
        command = [
            sys.executable,
            str(args.tools / TOOLS[0]),
            "--surface-dir",
            str(args.output / "tifxyz"),
            "--voxel-um",
            str(metadata["voxel_um"]),
            "--require-meta-json",
            "--write-masks",
            "--maximum-candidate-pairs",
            "20000000",
            "--output",
            str(gate_json),
        ]
        result["command"] = command
        print("RUN EXISTING GATE", " ".join(command), flush=True)
        with (
            (args.output / "gate.stdout.log").open("xb") as stdout,
            (args.output / "gate.stderr.log").open("xb") as stderr,
        ):
            run = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=160, check=False)
        result["returncode"] = run.returncode
        report = json.loads(gate_json.read_text())
        if len(report["surfaces"]) != 1:
            raise ValueError("expected exactly one measured raster artifact")
        measured = report["surfaces"][0]
        if measured["inputs"] != {
            n: {"sha256": digest, "size_bytes": (args.output / "tifxyz" / n).stat().st_size}
            for n, digest in result["conversion"]["files"].items()
        }:
            raise ValueError("gate inputs do not match exported TIFXYZ")
        result["gate_surface"] = measured
        result["gate_receipt_sha256"] = sha(gate_json)
        if any(sha(args.tools / n) != expected[n] for n in TOOLS) or any(
            sha(path) != digest for path, digest in result["inputs"].items()
        ):
            raise ValueError("inputs/tools changed during geometry check")
        result["status"] = gate_outcome(measured)
        if run.returncode not in (0, 1):
            result["status"] = "UNMEASURED"
            result["error"] = f"gate process exit {run.returncode}"
    except subprocess.TimeoutExpired:
        result.update(status="UNMEASURED", timed_out=True, error="existing gate exceeded 160-second inner deadline")
    except Exception as error:
        result.update(status="UNMEASURED", error=f"{type(error).__name__}: {error}")
    result["seconds"] = time.monotonic() - start
    result["artifacts"] = {
        str(f.relative_to(args.output)): dict(sha256=sha(f), bytes=f.stat().st_size)
        for f in sorted(args.output.rglob("*"))
        if f.is_file()
    }
    save_json(args.output / "result.json", result)
    print(json.dumps(dict(status=result["status"], seconds=result["seconds"], error=result.get("error"))), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
