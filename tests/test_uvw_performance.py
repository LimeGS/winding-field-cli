from contextlib import closing
import json
from pathlib import Path

import numpy as np
import pytest

from test_io import served, make_volume
from test_uvw import field_array
from winding_field.io import Volume
from winding_field.original_pipeline import sample_ct
from winding_field.tifxyz_field import PhysicalField
from winding_field.surface_sources import FieldSource


def test_decoded_ct_cache_reuses_and_evicts(tmp_path):
    data = make_volume(tmp_path / "ct")
    with closing(
        Volume(tmp_path / "ct", cache=tmp_path / "cache", decoded_cache_bytes=524)
    ) as volume:
        first = volume.decoded_chunk("0/0.0.0")
        second = volume.decoded_chunk("0/0.0.0")
        assert first is second and not first.flags.writeable
        np.testing.assert_array_equal(first, data[:2, :2, :3])
        volume.decoded_chunk("0/0.0.1")
        volume.decoded_chunk("0/0.0.0")
        stats = volume.stats()
        assert stats["decoded_hits"] == 1 and stats["decodes"] == 3
        assert stats["decoded_cache_bytes"] <= 524 and stats["decoded_evictions"] == 2


def test_decoded_cache_local_mutation_preserves_ledger(tmp_path):
    make_volume(tmp_path / "ct")
    with closing(
        Volume(tmp_path / "ct", cache=tmp_path / "cache", decoded_cache_bytes=1024)
    ) as volume:
        volume.bind_ledger(tmp_path / "source.sqlite")
        volume.decoded_chunk("0/0.0.0")
        (tmp_path / "ct/0/0.0.0").write_bytes(bytes([255]) * 12)
        with pytest.raises(ValueError, match="changed"):
            volume.decoded_chunk("0/0.0.0")


def test_ledger_bound_crops_avoid_redundant_file_stats(tmp_path, monkeypatch):
    data = make_volume(tmp_path / "ct")
    chunk = tmp_path / "ct/0/0.0.0"
    with closing(
        Volume(tmp_path / "ct", cache=tmp_path / "cache", decoded_cache_bytes=1024)
    ) as volume:
        volume.bind_ledger(tmp_path / "source.sqlite")
        stat = Path.stat
        calls = []

        def counted(path, *args, **kwargs):
            if path == chunk:
                calls.append(path)
            return stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", counted)
        for _ in range(2):
            np.testing.assert_array_equal(volume.read([0, 0, 0], [2, 2, 3]), data[:2, :2, :3])
        assert len(calls) <= 2
        assert volume.stats()["decodes"] == 1 and volume.stats()["decoded_hits"] == 1
        chunk.write_bytes(bytes([255]) * 12)
        with pytest.raises(ValueError, match="source chunk changed"):
            volume.read([0, 0, 0], [2, 2, 3])


def test_validate_used_revalidates_warm_remote_decoded_cache(served):
    root, url, _ = served
    with closing(Volume(url, cache=root / "cache", decoded_cache_bytes=1024)) as volume:
        volume.bind_ledger(root / "source.sqlite")
        volume.decoded_chunk("0/0.0.0")
        assert volume.decoded_chunk("0/0.0.0") is not None
        (root / "ct.zarr/0/0.0.0").write_bytes(bytes([255]) * 12)
        with pytest.raises(ValueError, match="changed"):
            volume.validate_used()
        assert volume.stats()["decoded_cache_bytes"] == 0


def test_binding_ledger_discards_previously_decoded_bytes(tmp_path):
    make_volume(tmp_path / "ct")
    with closing(
        Volume(tmp_path / "ct", cache=tmp_path / "cache", decoded_cache_bytes=1024)
    ) as volume:
        volume.decoded_chunk("0/0.0.0")
        (tmp_path / "ct/0/0.0.0").write_bytes(bytes([255]) * 12)
        volume.bind_ledger(tmp_path / "source.sqlite")
        assert volume.stats()["decoded_cache_bytes"] == 0
        np.testing.assert_array_equal(
            volume.decoded_chunk("0/0.0.0"), np.full((2, 2, 3), 255, dtype="u1")
        )


def test_verified_sampling_reuses_blocks_across_ledgers_and_pins_each(tmp_path, monkeypatch):
    import hashlib
    import sqlite3

    make_volume(tmp_path / "ct")
    key = "0/0.0.0"
    with closing(
        Volume(tmp_path / "ct", cache=tmp_path / "cache", decoded_cache_bytes=1024)
    ) as volume:
        reads = []
        read = volume._chunk_bytes

        def counted(key):
            reads.append(key)
            return read(key)

        monkeypatch.setattr(volume, "_chunk_bytes", counted)
        for tile in range(2):
            volume.bind_ledger(tmp_path / f"{tile}.sqlite")
            with volume.verified_chunks():
                first = volume.decoded_chunk(key)
                assert volume.decoded_chunk(key) is first
            with sqlite3.connect(tmp_path / f"{tile}.sqlite") as db:
                assert db.execute("SELECT key,sha FROM chunks").fetchall() == [
                    (key, hashlib.sha256((tmp_path / "ct" / key).read_bytes()).hexdigest())
                ]
        assert volume.stats()["decodes"] == 1
        # One initial read and one full payload check at each publication boundary.
        assert reads == [key, key, key]


def test_verified_sampling_allows_reader_callback_on_another_thread(tmp_path):
    import threading

    data = make_volume(tmp_path / "ct")
    finished = threading.Event()
    result = []
    errors = []
    with closing(
        Volume(tmp_path / "ct", cache=tmp_path / "cache", decoded_cache_bytes=1024)
    ) as volume:
        volume.bind_ledger(tmp_path / "source.sqlite")

        def callback():
            try:
                result.append(volume.decoded_chunk("0/0.0.0"))
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        worker = threading.Thread(target=callback, daemon=True)
        try:
            with volume.verified_chunks():
                worker.start()
                assert finished.wait(2), "reader callback blocked by the verification scope"
        finally:
            worker.join(timeout=5)
        assert not worker.is_alive() and not errors
        np.testing.assert_array_equal(result[0], data[:2, :2, :3])


def test_verified_sampling_detects_stat_collision_before_acceptance(tmp_path, monkeypatch):
    make_volume(tmp_path / "ct")
    key = "0/0.0.0"
    with closing(
        Volume(tmp_path / "ct", cache=tmp_path / "cache", decoded_cache_bytes=1024)
    ) as volume:
        volume.bind_ledger(tmp_path / "source.sqlite")
        with pytest.raises(ValueError, match="changed"):
            with volume.verified_chunks():
                first = volume.decoded_chunk(key)
                signature = volume._encoded_signature(key)
                monkeypatch.setattr(volume, "_encoded_signature", lambda key: signature)
                (tmp_path / "ct" / key).write_bytes(bytes([255]) * 12)
                assert volume.decoded_chunk(key) is first


def test_verified_sampling_rejects_conflicting_next_ledger(tmp_path):
    make_volume(tmp_path / "ct")
    key = "0/0.0.0"
    with closing(
        Volume(tmp_path / "ct", cache=tmp_path / "cache", decoded_cache_bytes=1024)
    ) as volume:
        volume.bind_ledger(tmp_path / "first.sqlite")
        with volume.verified_chunks():
            volume.decoded_chunk(key)
        volume.bind_ledger(tmp_path / "second.sqlite")
        volume._ledger.execute("INSERT INTO chunks VALUES (?,?)", (key, "incorrect"))
        volume._ledger.commit()
        with pytest.raises(ValueError, match="changed"):
            with volume.verified_chunks():
                volume.decoded_chunk(key)


def test_verified_sampling_keeps_eviction_bounded_across_tiles(tmp_path):
    data = make_volume(tmp_path / "ct")
    with closing(
        Volume(tmp_path / "ct", cache=tmp_path / "cache", decoded_cache_bytes=524)
    ) as volume:
        for tile, key, expected in [
            (0, "0/0.0.0", data[:2, :2, :3]),
            (1, "0/0.0.1", data[:2, :2, 3:]),
            (2, "0/0.0.0", data[:2, :2, :3]),
        ]:
            volume.bind_ledger(tmp_path / f"{tile}.sqlite")
            with volume.verified_chunks():
                np.testing.assert_array_equal(volume.decoded_chunk(key), expected)
            assert volume.stats()["decoded_cache_bytes"] <= 524
        assert volume.stats()["decoded_evictions"] == 2


def test_verified_sampling_rejects_corrupt_remote_cache_on_exit(served):
    root, url, _ = served
    with closing(Volume(url, cache=root / "cache", decoded_cache_bytes=1024)) as volume:
        volume.bind_ledger(root / "source.sqlite")
        with pytest.raises(ValueError, match="corrupt"):
            with volume.verified_chunks():
                volume.decoded_chunk("0/0.0.0")
                volume._native.path(url + "/0/0.0.0").write_bytes(bytes([255]) * 12)
        assert volume.stats()["decoded_cache_bytes"] == 0


def test_ledger_bound_cache_remains_bounded_and_rejects_missing_chunk(tmp_path):
    data = make_volume(tmp_path / "ct")
    with closing(
        Volume(tmp_path / "ct", cache=tmp_path / "cache", decoded_cache_bytes=524)
    ) as volume:
        volume.bind_ledger(tmp_path / "source.sqlite")
        for key, expected in [
            ("0/0.0.0", data[:2, :2, :3]),
            ("0/0.0.1", data[:2, :2, 3:]),
            ("0/0.0.0", data[:2, :2, :3]),
        ]:
            np.testing.assert_array_equal(volume.decoded_chunk(key), expected)
            assert volume.stats()["decoded_cache_bytes"] <= 524
        assert volume.stats()["decoded_evictions"] == 2
        (tmp_path / "ct/0/0.0.0").unlink()
        with pytest.raises(FileNotFoundError, match="missing local CT chunk"):
            volume.decoded_chunk("0/0.0.0")


def test_publication_snapshot_does_not_refetch_but_fresh_validation_detects_change(
    served, monkeypatch
):
    from winding_field import io

    root, url, _ = served
    fetch = io.fetch
    requests = []

    def counted(url, **kwargs):
        requests.append(url)
        return fetch(url, **kwargs)

    monkeypatch.setattr(io, "fetch", counted)
    with closing(Volume(url, cache=root / "cache", decoded_cache_bytes=1024)) as volume:
        volume.bind_ledger(root / "source.sqlite")
        volume.validate_used()
        volume.decoded_chunk("0/0.0.0")
        count = len(requests)
        for _ in range(4):
            volume.validate_used(fresh=False)
        assert len(requests) == count
        (root / "ct.zarr/0/0.0.0").write_bytes(bytes([255]) * 12)
        volume.validate_used(fresh=False)
        with pytest.raises(ValueError, match="changed"):
            volume.validate_used()
    with closing(Volume(url, cache=root / "cache")) as resumed:
        resumed.bind_ledger(root / "source.sqlite")
        with pytest.raises(ValueError, match="changed"):
            resumed.validate_used()


def test_snapshot_validation_checks_local_bytes_despite_decoded_hit(tmp_path):
    make_volume(tmp_path / "ct")
    with closing(
        Volume(tmp_path / "ct", cache=tmp_path / "cache", decoded_cache_bytes=1024)
    ) as volume:
        volume.bind_ledger(tmp_path / "source.sqlite")
        volume.decoded_chunk("0/0.0.0")
        (tmp_path / "ct/0/0.0.0").write_bytes(bytes([255]) * 12)
        with pytest.raises(ValueError, match="changed"):
            volume.validate_used(fresh=False)


def test_export_publications_share_verification_epoch(tmp_path, monkeypatch):
    from test_uvw_pipeline import inputs
    from winding_field.uvw_pipeline import export_existing

    calls = []
    validate = Volume.validate_used

    def checked(self, *, fresh=True):
        calls.append(fresh)
        return validate(self, fresh=fresh)

    monkeypatch.setattr(Volume, "validate_used", checked)
    args = inputs(tmp_path)
    export_existing(args)
    assert calls == [True, False, False, False, False]
    calls.clear()
    args.resume = True
    export_existing(args)
    assert calls[0] is True


def test_advisory_cache_hits_do_not_commit_each_read(served):
    root, url, _ = served
    volume = Volume(url, cache=root / "cache", cache_bytes=128 << 20)
    volume._chunk_bytes("0/0.0.0")
    statements = []
    volume._native.db.set_trace_callback(statements.append)
    for _ in range(20):
        volume._chunk_bytes("0/0.0.0")
    assert not any(s == "COMMIT" for s in statements)
    volume.close()
    import sqlite3

    with sqlite3.connect(root / "cache/vc-reader/receipts.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == 1
    with closing(Volume(url, cache=root / "cache", offline=True, cache_bytes=128 << 20)) as resumed:
        assert resumed._chunk_bytes("0/0.0.0") is not None


def test_shuffled_chunk_local_sampling_uses_one_decode_per_chunk(tmp_path):
    make_volume(tmp_path / "ct")
    points = np.array([[2.5, 1.5, 0.5], [0.5, 0.5, 0.5], [3.0, 2.0, 2.0], [2.5, 1.5, 0.5]])
    physical = points * [2.0, 3.0, 4.0] + [10.0, 20.0, 30.0]
    with closing(Volume(tmp_path / "ct", cache=tmp_path / "cache-a")) as cold:
        expected, mask = sample_ct(cold, physical)
    with closing(
        Volume(tmp_path / "ct", cache=tmp_path / "cache-b", decoded_cache_bytes=8192)
    ) as warm:
        order = [3, 1, 0, 2]
        actual, valid = sample_ct(warm, physical[order])
        count = warm.stats()["decodes"]
        again, again_valid = sample_ct(warm, physical)
        np.testing.assert_array_equal(actual, expected[order])
        np.testing.assert_array_equal(valid, mask[order])
        np.testing.assert_array_equal(again, expected)
        np.testing.assert_array_equal(again_valid, mask)
        assert warm.stats()["decodes"] == count and warm.stats()["decoded_hits"] >= count


def test_region_cache_reuses_cells_preserves_support_and_detects_mutation(tmp_path):
    mask = np.ones((12, 12, 12), bool)
    mask[5, 5, 5] = False
    original = field_array(tmp_path, lambda p: p[..., 2], mask=mask)
    descriptor = original.source.identity
    original.close()
    with PhysicalField(
        FieldSource(descriptor["field"], descriptor["geometry"]), region_cache_bytes=1 << 20
    ) as field:
        cells = np.array([[2, 2, 2], [4, 4, 4]])
        first, valid = field.read_cells(cells)
        again, supported = field.read_cells(cells)
        np.testing.assert_array_equal(first, again)
        assert supported.tolist() == valid.tolist() == [True, False]
        assert field.stats["read_calls"] == 1 and field.stats["region_hits"] == 1
        (tmp_path / "field.npy").write_bytes(b"changed")
        with pytest.raises(ValueError):
            field.read_cells(cells)
        assert field.stats["region_cache_bytes"] == 0
    assert field.stats["region_cache_bytes"] == 0


def test_region_cache_eviction_and_nonfinite_nodes(tmp_path):
    original = field_array(
        tmp_path, lambda p: np.where(p[..., 0] == 65, np.nan, p[..., 0]), shape=(4, 4, 130)
    )
    descriptor = original.source.identity
    original.close()
    with PhysicalField(
        FieldSource(descriptor["field"], descriptor["geometry"]), region_cache_bytes=10000
    ) as field:
        for x in (1, 65, 1):
            _, valid = field.read_cells(np.array([[1, 1, x]]))
            assert valid[0] == (x != 65)
        assert field.stats["region_evictions"] == 2 and field.stats["region_cache_bytes"] <= 10000


def test_spawn_workers_match_serial_and_do_not_write_sources(tmp_path):
    import os
    from test_uvw_pipeline import inputs
    from winding_field.uvw_workers import GeometryWorkers
    from winding_field.util import file_sha

    args = inputs(tmp_path)
    descriptor = json.loads(Path(args.field_config).read_text())
    before = {str(p): file_sha(p) for p in (tmp_path / "reference").rglob("*") if p.is_file()}
    uv = np.array([[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
    jobs = [dict(uv=uv, w=w, reference_data=None) for w in (34.0, 35.0, 36.0)]
    with GeometryWorkers(
        descriptor, args.reference_sheet, args.levels, 7, workers=1, region_cache_bytes=1 << 20
    ) as pool:
        serial = list(pool.map(jobs))
    with GeometryWorkers(
        descriptor, args.reference_sheet, args.levels, 7, workers=2, region_cache_bytes=1 << 20
    ) as pool:
        parallel = list(pool.map(jobs))
    for (a, _), (b, metrics) in zip(serial, parallel):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        assert metrics["pid"] != os.getpid() and metrics["native_threads"] == "1"
        assert metrics["geometry_seconds"] >= 0 and metrics["field"]["region_hits"] > 0
    assert before == {
        str(p): file_sha(p) for p in (tmp_path / "reference").rglob("*") if p.is_file()
    }


def test_parallel_publication_resume_and_worker_error(tmp_path):
    import tifffile
    from test_uvw_pipeline import inputs
    from winding_field.uvw_pipeline import export_existing
    from winding_field.uvw_workers import GeometryWorkers

    args = inputs(tmp_path)
    args.workers = 2
    args.cache_mib = 8
    args.canvas_samples = 33
    data = np.zeros((32, 32, 32), np.uint8)
    z, y, x = np.indices((12, 12, 12))
    data[:12, :12, :12] = x + 3 * y + 5 * z
    (tmp_path / "ct.zarr/0/0.0.0").write_bytes(data.tobytes())
    assert export_existing(args, stop=lambda: True)["status"] == "PAUSED"
    args.resume = True
    result = export_existing(args)
    assert result["quality"]["performance"]["workers"] == 2
    assert export_existing(args) == result
    parallel = tifffile.imread(Path(args.output) / "level-0000/tifxyz/x.tif")
    args.workers = 1
    args.output = str(tmp_path / "serial")
    args.resume = False
    export_existing(args)
    np.testing.assert_array_equal(
        parallel, tifffile.imread(Path(args.output) / "level-0000/tifxyz/x.tif")
    )
    from PIL import Image

    for level in range(3):
        for name in (
            "image.png",
            "geometry-mask.png",
            "ct-mask.png",
            "cause-mask.png",
            "cell-mask.png",
        ):
            with (
                Image.open(tmp_path / f"output/level-{level:04d}" / name) as a,
                Image.open(tmp_path / f"serial/level-{level:04d}" / name) as b,
            ):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        for axis in "xyz":
            np.testing.assert_array_equal(
                tifffile.imread(tmp_path / f"output/level-{level:04d}/tifxyz/{axis}.tif"),
                tifffile.imread(tmp_path / f"serial/level-{level:04d}/tifxyz/{axis}.tif"),
            )
    descriptor = json.loads(Path(args.field_config).read_text())
    with pytest.raises(ValueError, match="bounded"):
        with GeometryWorkers(
            descriptor, args.reference_sheet, args.levels, 7, workers=2, region_cache_bytes=0
        ) as pool:
            list(pool.map([dict(uv=np.zeros((16385, 2)), w=35.0, reference_data=None)]))


@pytest.mark.parametrize(
    "option,value", [("workers", 0), ("workers", 9), ("cache_mib", -1), ("cache_mib", float("nan"))]
)
def test_invalid_performance_settings_precede_output(tmp_path, option, value):
    from test_uvw_pipeline import inputs
    from winding_field.uvw_pipeline import export_existing

    args = inputs(tmp_path)
    setattr(args, option, value)
    with pytest.raises(ValueError):
        export_existing(args)
    assert not Path(args.output).exists()


def test_spawn_cancellation_restores_environment_and_reaps_children(tmp_path, monkeypatch):
    import multiprocessing, os
    from test_uvw_pipeline import inputs
    from winding_field.uvw_workers import GeometryWorkers

    args = inputs(tmp_path)
    descriptor = json.loads(Path(args.field_config).read_text())
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    before = {p.pid for p in multiprocessing.active_children()}
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(InterruptedError):
        with GeometryWorkers(
            descriptor, args.reference_sheet, args.levels, 7, workers=2, stop=stop
        ) as pool:
            list(
                pool.map(
                    [dict(uv=np.tile([3.0, 4.0], (1024, 1)), w=36.0, reference_data=None)] * 20
                )
            )
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_remote_decoded_cache_corruption_rechecks_encoded_bytes(served):
    root, url, data = served
    with closing(
        Volume(url, cache=root / "cache", decoded_cache_bytes=1024, cache_bytes=128 << 20)
    ) as volume:
        volume.bind_ledger(root / "source.sqlite")
        volume.decoded_chunk("0/0.0.0")
        path = volume._native.path(url + "/0/0.0.0")
        path.write_bytes(bytes([255]) * 12)
        with pytest.raises(ValueError, match="corrupt"):
            volume.decoded_chunk("0/0.0.0")


def test_workers_pin_reference_before_spawn(tmp_path):
    from concurrent.futures.process import BrokenProcessPool
    import sqlite3
    from test_uvw_pipeline import inputs
    from winding_field.uvw_workers import GeometryWorkers
    from winding_field.util import atomic_json, file_sha

    args = inputs(tmp_path)
    descriptor = json.loads(Path(args.field_config).read_text())
    with GeometryWorkers(descriptor, args.reference_sheet, args.levels, 7, workers=2) as pool:
        path = Path(args.reference_sheet)
        record = json.loads((path / "LEVEL.json").read_text())
        record["identity"]["level"] = 35.1
        atomic_json(path / "LEVEL.json", record)
        sha = file_sha(path / "LEVEL.json")
        with sqlite3.connect(path.parent / "publications.sqlite") as db:
            db.execute("UPDATE levels SET receipt_sha=?", (sha,))
        atomic_json(path.parent / "ORIGINAL.json", dict(level_receipt_sha256={path.name: sha}))
        with pytest.raises((ValueError, BrokenProcessPool)):
            list(pool.map([dict(uv=np.array([[3.0, 4.0]]), w=35.0, reference_data=None)]))


def test_region_hit_checks_source_after_copy(tmp_path, monkeypatch):
    original = field_array(tmp_path, lambda p: p[..., 2])
    descriptor = original.source.identity
    original.close()
    with PhysicalField(
        FieldSource(descriptor["field"], descriptor["geometry"]), region_cache_bytes=1 << 20
    ) as field:
        cells = np.array([[2, 2, 2]])
        field.read_cells(cells)
        check = field.source.validate_region

        def mutate_after_check(*args):
            check(*args)
            (tmp_path / "field.npy").write_bytes(b"concurrent replacement")

        monkeypatch.setattr(field.source, "validate_region", mutate_after_check)
        with pytest.raises(ValueError):
            field.read_cells(cells)
        assert field.stats["region_cache_bytes"] == 0


def test_completed_worker_timeout_is_an_error_not_a_wait_loop(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from test_uvw_pipeline import inputs
    from winding_field.uvw_workers import GeometryWorkers

    args = inputs(tmp_path)
    descriptor = json.loads(Path(args.field_config).read_text())
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls > 3

    with GeometryWorkers(
        descriptor, args.reference_sheet, args.levels, 7, workers=2, stop=stop
    ) as pool:
        failed = Future()
        failed.set_exception(TimeoutError("worker IO timeout"))
        monkeypatch.setattr(pool.pool, "submit", lambda *args: failed)
        with pytest.raises(TimeoutError, match="worker IO timeout"):
            list(pool.map([dict(uv=np.array([[3.0, 4.0]]), w=35.0, reference_data=None)]))


def test_cached_reference_batch_shape_is_validated(tmp_path):
    from test_uvw_pipeline import inputs
    from winding_field.uvw_workers import GeometryWorkers

    args = inputs(tmp_path)
    descriptor = json.loads(Path(args.field_config).read_text())
    reference_data = dict(
        base=np.array([[12.0, 22.0, 35.0], [13.0, 23.0, 35.0]]),
        reference_valid=np.ones(2, bool),
        cause=np.zeros(2, np.uint8),
    )
    with GeometryWorkers(descriptor, args.reference_sheet, args.levels, 7) as pool:
        with pytest.raises(ValueError, match="reference batch"):
            list(pool.map([dict(uv=np.array([[3.0, 4.0]]), w=35.0, reference_data=reference_data)]))
