"""Flatboi (VC3D) SLIM: pinned runtime provisioning and batched multi-chart solves.

SLIM in every pipeline is Flatboi's symmetric-Dirichlet optimizer. The runtime is the
hash-pinned VC3D AppImage recorded in ``provenance/flatboi.json`` (recipe ``historical``):
the AppImage bundles its own libraries and loader, so it runs on any Linux x86_64 host.
A bare binary without those libraries (the vendored ``installed-2f89bfc9`` copy) is only
usable through an explicit ``app_run`` whose installation provides them.

Resolution order when no ``app_run`` is given: ``WINDING_FLATBOI_APP_RUN``, the container
install at ``/opt/vc3d/AppRun``, the user cache, then a verified download of the AppImage
(disabled by ``offline=True`` or ``WINDING_FLATBOI_OFFLINE=1``).
"""

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import urllib.request

import numpy as np

APPIMAGE_URL = (
    "https://github.com/ScrollPrize/villa/releases/download/vc3d-archive-stable-363877681/"
    "VC3D-fc25b4d-2026-07-31-linux-x86_64.AppImage"
)
APPIMAGE_SHA256 = "5da27c3df44c0958390ea752b550756f9d7e2c104840a95c0ba8b8651c7aee9f"
CONTAINER_APP_RUN = Path("/opt/vc3d/AppRun")
ENERGY = "symmetric_dirichlet"


class FlatboiUnavailable(Exception):
    """Flatboi could not start (exit 127: missing shared libraries or loader).

    Not a ``RuntimeError`` on purpose: callers that bisect a failing batch or fall back
    to another route for a ``RuntimeError`` must stop instead, because every later call
    would fail the same way and the run would finish with an invalid result.
    """


def supported_platform():
    return platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64")


def require_supported_platform():
    if not supported_platform():
        raise ValueError(
            "SLIM flattening uses Flatboi, which is available only on Linux x86_64; "
            "use --flattening-method default (LSCM) on this platform"
        )


def cache_root():
    configured = os.environ.get("WINDING_FLATBOI_CACHE")
    return Path(configured or "~/.cache/winding-field/flatboi").expanduser()


def cached_app_run():
    return cache_root() / APPIMAGE_SHA256[:16] / "squashfs-root" / "AppRun"


def resolve_app_run(app_run=None, recipe="historical", *, offline=False):
    """Return the AppRun to verify; provisions the pinned AppImage when allowed."""
    if app_run:
        return Path(app_run).expanduser()
    if recipe != "historical":
        raise ValueError(
            f"Flatboi recipe {recipe} requires an explicit app_run whose installation provides "
            "its libraries; omit native_recipe to use the pinned historical AppImage"
        )
    configured = os.environ.get("WINDING_FLATBOI_APP_RUN")
    if configured:
        return Path(configured).expanduser()
    if CONTAINER_APP_RUN.is_file():
        return CONTAINER_APP_RUN
    require_supported_platform()
    app = cached_app_run()
    if app.is_file():
        return app
    if offline or os.environ.get("WINDING_FLATBOI_OFFLINE") == "1":
        raise ValueError(
            "Flatboi runtime is not cached and downloads are disabled; run "
            "`python -m winding_field.flattening.flatboi --provision` while online"
        )
    provision()
    return app


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def provision(url=APPIMAGE_URL, sha256=APPIMAGE_SHA256):
    """Download, verify and extract the pinned AppImage into the user cache (idempotent)."""
    require_supported_platform()
    root = cache_root() / sha256[:16]
    app = root / "squashfs-root" / "AppRun"
    with _lock(cache_root() / (sha256[:16] + ".lock")):
        if app.is_file():
            return app
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".provision-", dir=root) as temporary:
            image = Path(temporary) / "vc3d.AppImage"
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response, image.open("wb") as out:
                while chunk := response.read(1 << 20):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != sha256:
                raise ValueError("Flatboi AppImage SHA256 mismatch")
            image.chmod(0o755)
            subprocess.run(
                [str(image), "--appimage-extract"],
                cwd=temporary,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.replace(Path(temporary) / "squashfs-root", root / "squashfs-root")
    return app


def _write_obj(path, vertices, uv, faces):
    from .obj import write_obj

    with open(path, "w") as stream:
        write_obj(stream, vertices, faces, uv)


def solve_batch(
    charts, app_run, *, iterations=200, tolerance=1e-6, check=None, threads=1, timeout=1800.0
):
    """Run one Flatboi SLIM solve over many disconnected charts.

    ``charts`` is a sequence of ``(vertices, faces, seed_uv)`` with local indices. Each chart
    keeps its own seed; Flatboi's energy is separable across components. Returns per-chart
    UV arrays and the shared solver receipt. Raises on native failure, ignored warm starts or
    a non-monotone energy history; per-chart validity is the caller's gate.

    One PaStiX thread by default: multithreaded factorization is not bitwise reproducible
    (every chart of a 300-chart batch differed by up to 3e-7 between identical runs), and a
    resumed run must recompute an interrupted group to the same bytes. It costs no time at
    the native batch size (65k faces: 2.7 s with one thread, 2.8-3.1 s with two).
    """
    from ..surface_export import read_flatboi_uv

    if not charts:
        return [], dict(steps=0, converged=True)
    offsets = np.cumsum([0] + [len(v) for v, _, _ in charts])
    vertices = np.concatenate([np.asarray(v, float) for v, _, _ in charts])
    # Disjoint seed boxes: overlapping seeds make Flatboi's global UV work ~30x slower
    # (16.2 s vs 0.49 s for 50 small charts). Translation does not change the energy.
    shifts, cursor = [], 0.0
    for _, _, uv in charts:
        uv = np.asarray(uv, float)
        low, high = uv.min(0), uv.max(0)
        gap = max(float((high - low).max()), 1e-6) * 0.25
        shifts.append(np.array([cursor - low[0], -low[1]]))
        cursor += float(high[0] - low[0]) + gap
    seeds = np.concatenate([np.asarray(uv, float) + d for (_, _, uv), d in zip(charts, shifts)])
    faces = np.concatenate([np.asarray(f, np.int64) + o for (_, f, _), o in zip(charts, offsets)])
    env = dict(
        os.environ,
        PASTIX_NUM_THREADS=str(threads),
        OMP_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    with tempfile.TemporaryDirectory(prefix="flatboi-") as temporary:
        work = Path(temporary)
        mesh = work / "inputmesh.obj"
        _write_obj(mesh, vertices, seeds, faces)
        command = [
            str(app_run),
            "flatboi",
            str(mesh),
            str(int(iterations)),
            ENERGY,
            f"--tol={tolerance:g}",
        ]
        with (work / "stdout.log").open("wb") as out, (work / "stderr.log").open("wb") as err:
            child = subprocess.Popen(
                command, cwd=work, env=env, stdout=out, stderr=err, start_new_session=True
            )
            deadline = time.monotonic() + timeout
            try:
                while True:
                    if check:
                        check()
                    try:
                        code = child.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    # PaStiX reports this and then blocks forever instead of exiting.
                    if b"Negative diagonal term" in (work / "stderr.log").read_bytes()[-65536:]:
                        raise RuntimeError("Flatboi reported a negative diagonal term")
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"Flatboi exceeded {timeout:g} s")
            except BaseException:
                child.kill()
                child.wait()
                raise
        log = (work / "stdout.log").read_text(errors="replace")
        errors = (work / "stderr.log").read_text(errors="replace")
        if code == 127 or "error while loading shared libraries" in errors:
            raise FlatboiUnavailable(
                f"Flatboi could not start (exit {code}); install its system libraries: "
                + errors[-2000:]
            )
        if code:
            raise RuntimeError(f"Flatboi exited with {code}: {errors[-2000:]}")
        if "No per-corner UVs" in log:
            raise RuntimeError("Flatboi ignored the warm-start UVs")
        if "Negative diagonal term" in errors:
            raise RuntimeError("Flatboi reported a negative diagonal term")
        energies = np.atleast_1d(np.loadtxt(work / "inputmesh_energies_flatboi.txt"))
        if energies.ndim != 1 or not len(energies) or not np.isfinite(energies).all():
            raise RuntimeError("invalid Flatboi energy history")
        if (np.diff(energies) > 1e-7 * np.maximum(1.0, energies[:-1])).any():
            raise RuntimeError("Flatboi energy increased")
        uv, _ = read_flatboi_uv(work / "inputmesh_flatboi.obj", vertices, faces, vertex_atol=0)
    receipt = dict(
        steps=int(len(energies) - 1),
        converged="Converged:" in log,
        energy_first=float(energies[0]),
        energy_last=float(energies[-1]),
        charts=len(charts),
        faces=int(len(faces)),
    )
    return [uv[offsets[i] : offsets[i + 1]] - shifts[i] for i in range(len(charts))], receipt


def smoke_test(app_run):
    """Solve one tiny chart so a Flatboi that cannot start fails before any real work."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.1], [1.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    try:
        solve_batch([(vertices, faces, vertices[:, :2].copy())], app_run, iterations=2)
    except RuntimeError:
        pass  # a numerical outcome on this toy chart says nothing about the runtime
    return app_run


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Provision the pinned Flatboi runtime")
    parser.add_argument("--provision", action="store_true", required=True)
    parser.parse_args()
    print(provision())
