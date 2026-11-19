#!/usr/bin/env python3
"""Sequential, source-preserving benchmarks of local imported prepare/fibres/exports."""

import argparse
import cProfile
from contextlib import ExitStack, closing
import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from unittest.mock import patch


def _inputs(config, base):
    paths = []
    for key in (
        "source",
        "local",
        "fibres",
        "fibres_geometry",
        "lasagna_input",
        "lasagna_geometry",
    ):
        if not config.get(key):
            continue
        if "://" in config[key]:
            raise ValueError("benchmark requires purely local inputs")
        config[key] = str((base / config[key]).resolve())
        paths.append(Path(config[key]))
    if not config.get("fibres") or not config.get("lasagna_input"):
        raise ValueError("benchmark requires imported fibres and Lasagna")
    manifest = json.loads(Path(config["lasagna_input"]).read_bytes())
    for group in manifest["groups"].values():
        path = group["zarr"]
        if "://" in path:
            raise ValueError("benchmark requires local prediction arrays")
        channel = (Path(config["lasagna_input"]).parent / path).resolve()
        paths.append(channel)
        if (channel.parent / ".zattrs").is_file():
            paths.append(channel.parent / ".zattrs")
    if manifest.get("provenance"):
        paths.append((Path(config["lasagna_input"]).parent / manifest["provenance"]).resolve())
    return paths


def _snapshot(paths):
    from winding_field.util import file_sha

    return {
        str(p): (file_sha(p), p.stat().st_mtime_ns)
        for path in paths
        for p in (sorted(path.rglob("*")) if path.is_dir() else [path])
        if p.is_file()
    }


def _logical(root):
    from winding_field.artifacts import Tiles
    from bench_afv_import import logical_sha

    h = hashlib.sha256()
    with closing(Tiles(root / "maps")) as maps:
        for key in maps.keys():
            for name, array in sorted(maps.load(key).items()):
                h.update(name.encode())
                h.update(str(array.dtype).encode())
                h.update(array.tobytes())
    receipt = json.loads((root / "FIBRES.json").read_bytes())
    # Receipts contain transport/run identities; science parity is the logical rows.
    h.update(
        logical_sha(
            root,
            {
                k: receipt[k]
                for k in ("count", "input_count", "imported_sources", "excluded_sources")
            },
        ).encode()
    )
    return h.hexdigest()


def _worker(config, output, workers, profiled, resume_case=False):
    from winding_field import (
        afv,
        artifacts,
        exports,
        io,
        lasagna_input,
        lasagna_stream,
        predict,
        workflow,
        util,
    )

    metrics = dict(decodes=0, support_loads=0, hashed_bytes=0, commits=0, stage_seconds={})
    connect = sqlite3.connect

    class Counted(sqlite3.Connection):
        def commit(self):
            writing = self.in_transaction
            result = super().commit()
            metrics["commits"] += int(writing)
            return result

    def opened(database, *args, **kwargs):
        kwargs.setdefault("factory", Counted)
        return connect(database, *args, **kwargs)

    def timed(fn, name):
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                metrics["stage_seconds"][name] = (
                    metrics["stage_seconds"].get(name, 0.0) + time.perf_counter() - start
                )

        return call

    decode = io.Volume._decode_chunk
    load = artifacts.Tiles.load
    sha = util.file_sha
    record = io.Volume._record

    def decoded(self, *args, **kwargs):
        before = self._stats["decodes"]
        try:
            return decode(self, *args, **kwargs)
        finally:
            metrics["decodes"] += self._stats["decodes"] - before

    def loaded(self, key, names=None):
        metrics["support_loads"] += int(names == ("support",))
        return load(self, key, names)

    def hashed(path):
        result = sha(path)
        metrics["hashed_bytes"] += Path(path).stat().st_size
        return result

    def recorded(self, *args, **kwargs):
        before = self._stats["hashed_bytes"]
        try:
            return record(self, *args, **kwargs)
        finally:
            metrics["hashed_bytes"] += self._stats["hashed_bytes"] - before

    done = threading.Event()
    peak = [0]

    def memory():
        while not done.is_set():
            pids = [os.getpid(), *[p.pid for p in multiprocessing.active_children()]]
            try:
                if sys.platform == "darwin":
                    raw = subprocess.check_output(
                        ["ps", "-o", "rss=", "-p", ",".join(map(str, pids))], text=True
                    )
                    rss = sum(int(v) * 1024 for v in raw.split())
                else:
                    from winding_field.tifxyz_runtime import process_rss

                    rss = sum(process_rss(pid) for pid in pids)
                peak[0] = max(peak[0], rss)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                pass
            done.wait(0.05)

    sampler = threading.Thread(target=memory, daemon=True)
    sampler.start()
    config = dict(config, output=str(output), cache=str(output / "cache"), import_workers=workers)
    profiler = cProfile.Profile() if profiled else None
    start = time.perf_counter()
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(sqlite3, "connect", opened))
            stack.enter_context(patch.object(io.Volume, "_decode_chunk", decoded))
            stack.enter_context(patch.object(io.Volume, "_record", recorded))
            stack.enter_context(patch.object(artifacts.Tiles, "load", loaded))
            stack.enter_context(
                patch.object(
                    io.Volume, "validate_used", timed(io.Volume.validate_used, "validation")
                )
            )
            stack.enter_context(
                patch.object(
                    artifacts.np,
                    "savez_compressed",
                    timed(artifacts.np.savez_compressed, "map_compression"),
                )
            )
            for module, label in ((afv, "afv_validation"), (lasagna_input, "lasagna_validation")):
                stack.enter_context(
                    patch.object(module, "source_identity", timed(module.source_identity, label))
                )
            for module in (
                afv,
                artifacts,
                exports,
                io,
                lasagna_input,
                lasagna_stream,
                workflow,
                util,
            ):
                if hasattr(module, "file_sha"):
                    stack.enter_context(patch.object(module, "file_sha", hashed))
            for module, name, label in (
                (predict, "prepare", "prepare"),
                (afv, "import_fibres", "fibres"),
                (exports, "export_afv", "export_afv"),
                (exports, "export_lasagna", "export_lasagna"),
            ):
                stack.enter_context(patch.object(module, name, timed(getattr(module, name), label)))
            if profiler:
                profiler.enable()
            if resume_case:
                prepare = predict.prepare
                first = output / "maps/0_0_0.npz"

                def pause(volume, run, config, stop=None, **kwargs):
                    return prepare(
                        volume,
                        run,
                        config,
                        stop=lambda: first.exists() or (stop and stop()),
                        **kwargs,
                    )

                with patch.object(predict, "prepare", pause):
                    paused = workflow.run(config, until="fibres")
                if paused["status"] != "PAUSED" or not first.exists():
                    raise ValueError("benchmark did not pause at a retained tile")
                pinned = (sha(first), first.stat().st_mtime_ns)
                metrics["partial_pause_seconds"] = time.perf_counter() - start
                resumed = time.perf_counter()
                state = workflow.run(config, resume=True, until="fibres")
                metrics["partial_resume_seconds"] = time.perf_counter() - resumed
                if pinned != (sha(first), first.stat().st_mtime_ns):
                    raise ValueError("partial resume rewrote a retained tile")
            else:
                state = workflow.run(config, until="fibres")
            if profiler:
                profiler.disable()
            elapsed = time.perf_counter() - start
    finally:
        done.set()
        sampler.join()
    if set(state["stages"]) != {"prepare", "fibres"}:
        raise ValueError("benchmark pipeline did not complete")
    if profiler:
        profiler.dump_stats(output / "PROFILE.pstats")
    receipt = json.loads((output / "FIBRES.json").read_bytes())
    status = json.loads((output / "fibres/STATUS.json").read_bytes())
    metrics["execution"] = status.get("execution", dict(workers=1))
    metrics["worker_support_loads"] = metrics["execution"].get("worker_support_loads", 0)
    for name in ("lasagna-source", "lasagna-import"):
        path = output / name / "PERFORMANCE.json"
        if path.exists():
            metrics["prepare_execution"] = json.loads(path.read_bytes())
    child = metrics.get("prepare_execution", {})
    metrics.update(
        worker_decodes=child.get("decodes", 0),
        worker_hashed_bytes=child.get("hashed_bytes", 0),
        worker_chunk_commits=child.get("chunk_commits", 0),
    )
    metrics.update(
        total_decodes=metrics["decodes"] + metrics["worker_decodes"],
        total_commits=metrics["commits"] + metrics["worker_chunk_commits"],
    )
    logical = _logical(output)
    resumed = time.perf_counter()
    workflow.run(config, resume=True, until="fibres")
    metrics["completed_resume_seconds"] = time.perf_counter() - resumed
    if logical != _logical(output):
        raise ValueError("completed resume changed logical observations/curves")
    metrics.update(
        seconds=elapsed,
        pid=os.getpid(),
        workers=workers,
        profiled=profiled,
        resume_case=resume_case,
        peak_rss_bytes=peak[0]
        or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        curves=receipt["count"],
        logical_sha256=logical,
        output_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
    )
    util.atomic_json(output / "METRICS.json", metrics)


def run(output, *, config, repeats=3, workers=(1,), profile=False):
    from winding_field import workflow
    from winding_field.util import atomic_json, digest, file_sha

    if (
        type(repeats) is not int
        or not 1 <= repeats <= 20
        or not workers
        or any(type(n) is not int or n < 0 for n in workers)
    ):
        raise ValueError("invalid benchmark repetitions/workers")
    config = Path(config).resolve()
    settings = json.loads(config.read_bytes())
    paths = _inputs(settings, config.parent)
    paths.append(config)
    output = Path(output).resolve()
    for path in paths:
        if output == path or output in path.parents or (path.is_dir() and path in output.parents):
            raise ValueError("benchmark output overlaps an input")
    if output.exists():
        raise FileExistsError(output)
    before = _snapshot(paths)
    output.mkdir(parents=True)
    effective = output / "config.json"
    atomic_json(effective, settings)

    def invoke(n, index, profiled=False, resume_case=False):
        root = output / f"workers-{n}-{index}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            str(effective),
            "--output",
            str(root),
            "--workers",
            str(n),
        ]
        if profiled:
            command.append("--profile")
        if resume_case:
            command.append("--resume-case")
        child = subprocess.run(command, capture_output=True, text=True)
        (output / f"workers-{n}-{index}.log").write_text(child.stdout + child.stderr)
        if child.returncode:
            raise RuntimeError("benchmark worker failed: " + child.stderr[-3000:])
        return json.loads((root / "METRICS.json").read_bytes())

    records = [invoke(n, index) for index in range(repeats) for n in workers]
    result = dict(
        schema="local-import-benchmark-v1",
        measurements=records,
        median_seconds={
            str(n): statistics.median(r["seconds"] for r in records if r["workers"] == n)
            for n in workers
        },
        input_sha256=digest(before),
        config_sha256=file_sha(effective),
        implementation_sha256=workflow.implementation_sha(),
        logical_results_equal=len({r["logical_sha256"] for r in records}) == 1,
        runtime=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            core_version=importlib.metadata.version("winding-field"),
        ),
        cache_policy="Sequential processes; input hashing warms OS cache. No cache drops. Profile/resume trials excluded from medians.",
        counter_scope="Parent operations; worker counters reported separately when available.",
    )
    if profile:
        result["profile"] = invoke(workers[0], "profile", True)
    result["resume"] = invoke(workers[0], "resume", resume_case=True)
    if result["resume"]["logical_sha256"] != records[0]["logical_sha256"]:
        raise ValueError("partial resume logical results differ")
    if before != _snapshot(paths):
        raise ValueError("benchmark inputs changed")
    atomic_json(output / "RESULT.json", result)
    if not result["logical_results_equal"]:
        raise ValueError("worker logical results differ")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", nargs="+", type=int, default=[1])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume-case", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(
            json.loads(args.config.read_bytes()),
            args.output,
            args.workers[0],
            args.profile,
            args.resume_case,
        )
    else:
        print(
            json.dumps(
                run(
                    args.output,
                    config=args.config,
                    repeats=args.repeats,
                    workers=tuple(args.workers),
                    profile=args.profile,
                ),
                indent=2,
            )
        )
