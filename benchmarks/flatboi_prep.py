"""PYTHONPATH=src python benchmarks/flatboi_prep.py — host-side costs, no native SLIM."""

import subprocess
import tempfile
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from statistics import median
from time import perf_counter

import igl
import numpy as np

from winding_field import surface_export as export
from winding_field.flattening import flatboi, matrix, native_sheet, original


def baseline(module):
    path = "src/" + module.__name__.replace(".", "/") + ".py"
    source = subprocess.check_output(["git", "show", f"eea0cbb:{path}"])
    with tempfile.TemporaryDirectory() as folder:
        snapshot = Path(folder) / "baseline.py"
        snapshot.write_bytes(source)
        spec = spec_from_file_location(module.__name__ + "_baseline", snapshot)
        old = module_from_spec(spec)
        spec.loader.exec_module(old)
    return old


def run(label, old, new, key):
    times = [[], []]
    ref = None
    for order in [(0, 1), (1, 0)] * 2:
        for i in order:
            start = perf_counter()
            value = (old, new)[i]()
            wall = perf_counter() - start
            value = key(value)
            if ref is None:
                ref = value
            assert ref == value, label
            times[i].append(wall)
    a, b = map(median, times)
    print(f"{label:30} {a:.4f}s -> {b:.4f}s   {a / b:.2f}x", flush=True)


def grid(n):
    y, x = np.indices((n, n))
    v = np.column_stack((x.ravel(), y.ravel(), np.sin(x.ravel() / 9) * np.cos(y.ravel() / 12)))
    k = (y[:-1, :-1] * n + x[:-1, :-1]).ravel()
    f = np.concatenate([np.stack([k, k + 1, k + n + 1], 1), np.stack([k, k + n + 1, k + n], 1)])
    return v, f, v[:, :2] + 0.25


def array_receipt(result):
    return sha256(result[0].tobytes()).hexdigest(), result[1]


def main():
    e, b, s, o = map(baseline, (export, flatboi, native_sheet, original))
    v, f, uv = grid(256)
    print(f"{len(v):,} vertices, {len(f):,} triangles; four interleaved runs per median")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "mesh.obj"

        def write(module):
            module._write_obj(path, v, uv, f)
            return path

        run(
            "Seed OBJ serialization",
            lambda: write(b),
            lambda: write(flatboi),
            lambda p: sha256(p.read_bytes()).hexdigest(),
        )
        print(f"Seed OBJ SHA256 {sha256(path.read_bytes()).hexdigest()}")
        for layout in ("seed", "per-corner"):
            if layout == "per-corner":
                tf = np.arange(3 * len(f)).reshape(-1, 3)
                with path.open("w") as out:
                    np.savetxt(out, v, fmt="v %.17g %.17g %.17g")
                    np.savetxt(out, uv[f.ravel()], fmt="vt %.17g %.17g")
                    np.savetxt(
                        out,
                        np.stack([f + 1, tf + 1], axis=2).reshape(-1, 6),
                        fmt="f %d/%d %d/%d %d/%d",
                    )
            run(
                "OBJ readback: " + layout,
                lambda: e.read_flatboi_uv(path, v, f, vertex_atol=0),
                lambda: export.read_flatboi_uv(path, v, f, vertex_atol=0),
                array_receipt,
            )
        run(
            "surface_export UV calibration",
            lambda: e.scale_uv_to_world(v, f, uv),
            lambda: export.scale_uv_to_world(v, f, uv),
            array_receipt,
        )
        run(
            "matrix/sheet UV calibration",
            lambda: o.scale_uv_to_world(v, f, uv),
            lambda: matrix.scale_uv_to_world(v, f, uv),
            array_receipt,
        )
        run(
            "Cut component and topology",
            lambda: s.cut_component(v, f),
            lambda: native_sheet.cut_component(v, f),
            lambda r: ({k: sha256(a.tobytes()).hexdigest() for k, a in r[0].items()}, r[1]),
        )
    v, f, _ = grid(40)
    run(
        "Original LSCM map (3,042 faces)",
        lambda: o.make_map(v, f),
        lambda: original.make_map(v, f),
        lambda r: {
            k: sha256(a.tobytes()).hexdigest() if k != "report" else a for k, a in r.items()
        },
    )
    cut = {"vertices": v, "faces": f, "boundary": igl.boundary_loop(f), "parent": np.arange(len(v))}
    run(
        "MVC initializer (3,042 faces)",
        lambda: s.mean_value_map(cut),
        lambda: native_sheet.mean_value_map(cut),
        array_receipt,
    )


if __name__ == "__main__":
    main()
