"""Offline installed-library example: prepared inputs, durable stop/resume, prefetch.

Run ``python -I examples/block_field.py --output NEW_DIRECTORY`` after installing
both packages. This is analytic synthetic truth, not validation of a real scroll.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import torch

import block_solver
import constraint_templates
import field_constraints
import field_observations
import field_prefetch
import field_runtime
import field_solver
import field_sources
import field_store
from winding_field import field_workflow
from constraint_templates import StencilConstraints
from field_observations import density_normals
from field_runtime import ResourceUnavailable, run_supervised
from field_sources import CertifiedPresence, NpyObservations
from field_store import BlockStore, atomic_json


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


MODULES = (
    block_solver,
    constraint_templates,
    field_constraints,
    field_observations,
    field_prefetch,
    field_runtime,
    field_solver,
    field_sources,
    field_store,
    field_workflow,
)


def sources():
    return {module.__name__: sha(module.__file__) for module in MODULES}


def create_case(folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    shape = (3, 4, 6)
    arrays = dict(
        target=np.broadcast_to(np.array([0.125, 0.0, 0.0], np.float32), (*shape, 3)),
        weights=np.ones(shape, np.float32),
        observed=np.ones(shape, bool),
    )
    for name, value in arrays.items():
        np.save(folder / (name + ".npy"), value)

    # Exercise certified input and the estimator separately from analytic solver truth.
    presence = folder / "presence"
    (presence / "0/0").mkdir(parents=True)
    volume = np.broadcast_to(np.where(np.arange(88) % 16 < 2, 255, 0).astype(np.uint8), (88, 88, 88)).copy()
    (presence / "0/0/0").write_bytes(volume.tobytes())
    atomic_json(
        presence / ".zarray",
        dict(
            zarr_format=2,
            shape=[88] * 3,
            chunks=[88] * 3,
            dtype="|u1",
            fill_value=0,
            order="C",
            dimension_separator="/",
            filters=None,
            compressor=None,
        ),
    )
    manifest = folder / "presence.json"
    atomic_json(
        manifest,
        dict(
            schema="winding-presence-v1",
            metadata_sha256=sha(presence / ".zarray"),
            shape_zyx=[88] * 3,
            coverage_zyx=[[[0] * 3, [88] * 3]],
            chunks={"0/0/0": sha(presence / "0/0/0")},
        ),
    )
    reader = CertifiedPresence(presence, manifest, sha(manifest))
    data, receipts = reader.read([0] * 3, [88] * 3)
    density, normals, support = density_normals(data)
    assert bool(support.all()) and bool((density > 0).all())
    usable = torch.linalg.vector_norm(normals, dim=-1) > 0.5
    assert bool(usable.any())
    normal_error = float((normals[..., 0][usable].abs() - 1).abs().max())
    assert normal_error < 1e-5
    atomic_json(
        folder / "CASE.json",
        dict(
            schema="synthetic-block-example-v1",
            shape_zyx=list(shape),
            origin_xyz=[11.0, -4.0, 19.0],
            spacing_xyz=[2.0, 3.0, 5.0],
            hashes={name: sha(folder / (name + ".npy")) for name in arrays},
            observation_control=dict(
                backend="torch/cpu",
                normal_max_error=normal_error,
                support_fraction=float(support.float().mean()),
                director_fraction=float(usable.float().mean()),
                receipts=receipts,
            ),
        ),
    )
    options = case_options(folder)
    options["constraints"].save(folder / "constraints.npz")
    atomic_json(
        folder / "field-run.json",
        dict(
            schema="winding-field-run-v1",
            frame=options["identity"]["frame"],
            voxel_um=8.64,
            sheet_step=1.0,
            calibration=options["identity"]["calibration"],
            grid=dict(
                origin_xyz=options["origin_xyz"].tolist(),
                spacing_xyz=options["spacing_xyz"].tolist(),
                shape_zyx=list(shape),
            ),
            observations={
                name: dict(path=name + ".npy", sha256=sha(folder / (name + ".npy")))
                for name in ("target", "weights", "observed")
            },
            constraints=dict(path="constraints.npz", sha256=sha(folder / "constraints.npz")),
            solver=dict(device="cpu", dtype="float64", rtol=1e-10, maxiter=1000),
            execution=dict(
                chunk=4, halo=2, sweeps=2, cache_bytes=1 << 20, read_workers=2, constraint_cache_bytes=1 << 20
            ),
        ),
    )


def case_options(folder, *, cache_bytes=0, read_workers=1, pair_cache_bytes=0):
    folder = Path(folder)
    case = json.loads((folder / "CASE.json").read_text())
    if case.get("schema") != "synthetic-block-example-v1":
        raise ValueError("not a synthetic example case")
    loader = NpyObservations(*(folder / (name + ".npy") for name in ("target", "weights", "observed")), case["hashes"])
    origin, spacing = np.asarray(case["origin_xyz"]), np.asarray(case["spacing_xyz"])
    constraints = StencilConstraints.pairs(
        origin + spacing * [[0.25, 1.0, 1.0]],
        origin + spacing * [[4.75, 2.0, 1.0]],
        origin,
        spacing,
        case["shape_zyx"],
        targets=[-1.125],
        weight=2.0,
        cache_bytes=pair_cache_bytes,
    )
    return dict(
        shape=case["shape_zyx"],
        chunk=4,
        identity=dict(
            frame="synthetic-performance/world",
            calibration="analytic phi = 0.125 * (world_X - origin_X)",
            sheet_step=1.0,
            observations=loader.identity,
            code_sha256=sources(),
            versions=dict(
                python=platform.python_version(),
                numpy=np.__version__,
                scipy=scipy.__version__,
                torch=str(torch.__version__),
            ),
        ),
        load_observations=loader,
        origin_xyz=origin,
        spacing_xyz=spacing,
        order=[[0, 0, 0], [0, 0, 1]],
        constraints=constraints,
        halo=2,
        sweeps=2,
        rtol=1e-10,
        maxiter=1000,
        cache_bytes=cache_bytes,
        read_workers=read_workers,
    )


def read_result(root):
    metadata = json.loads((root / "CHECKPOINT.json").read_text())
    with BlockStore(root, metadata["shape"], metadata["chunk"], metadata["identity"], resume=True) as store:
        return store.read([0] * 3, store.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prefetch", action="store_true", help="one CPU observation worker")
    parser.add_argument("--supervised", action="store_true", help="owned worker with resource-only retries")
    parser.add_argument("--stop-after-blocks", type=int)
    args = parser.parse_args()
    if args.stop_after_blocks is not None and args.stop_after_blocks < 1:
        parser.error("stop-after-blocks must be positive")
    if not args.output.exists():
        if args.resume:
            parser.error("resume requires an existing example")
        create_case(args.output)
    source_before = sources()
    root = args.output / "field"
    if args.supervised:
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output.resolve())]
        if args.prefetch:
            command.append("--prefetch")
        if args.stop_after_blocks is not None:
            command += ["--stop-after-blocks", str(args.stop_after_blocks)]
        inventory = {str(Path(module.__file__).resolve()): sha(module.__file__) for module in MODULES}
        inventory[str(Path(__file__).resolve())] = sha(__file__)
        inventory[str((args.output / "CASE.json").resolve())] = sha(args.output / "CASE.json")
        inventory[str((args.output / "field-run.json").resolve())] = sha(args.output / "field-run.json")
        return run_supervised(
            command,
            [*command, "--resume"],
            root.resolve(),
            (args.output / "service").resolve(),
            source_sha256=inventory,
            resume=args.resume,
        )

    def stop():
        if args.stop_after_blocks is None or not (root / "CHECKPOINT.json").exists():
            return False
        progress = json.loads((root / "CHECKPOINT.json").read_text())["progress"]
        return (progress.get("sweep", 1) - 1) * 2 + progress.get("next_block", 0) >= args.stop_after_blocks

    try:
        receipt = field_workflow.run_field(
            args.output / "field-run.json", root, resume=args.resume, stop=stop, prefetch=args.prefetch
        )
    except ResourceUnavailable:
        return 75
    if sources() != source_before:
        raise ValueError("example library sources changed during execution")
    if receipt["status"] == "SWEEP_BUDGET_REACHED":
        field = read_result(root)
        error = float(np.max(abs(field - np.broadcast_to(np.arange(6) * 0.25, field.shape))))
        assert error < 2e-7, error
        np.save(args.output / "synthetic-field.npy", field)
        receipt["synthetic_max_error"] = error
    atomic_json(args.output / "RESULT.json", receipt)
    print(json.dumps(receipt, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
