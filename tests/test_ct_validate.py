"""validate_used: a ledger-pinned CT cache needs no remote request; remote checks overlap."""

from contextlib import closing
import sqlite3
import threading

import numpy as np
import pytest

from test_io import served  # noqa: F401 (pytest fixtures)

CHUNKS = 8  # the served 4x4x6 array in 2x2x3 chunks


def warm(root, url):
    """An online run that pinned every chunk (one of them an authoritative remote miss)."""
    from winding_field.io import Volume

    (root / "ct.zarr/0/1.1.1").unlink()
    with closing(Volume(url, cache=root / "cache")) as volume:
        volume.bind_ledger(root / "source.sqlite")
        data = volume.read([0, 0, 0], [4, 4, 6])
    with sqlite3.connect(root / "source.sqlite") as db:
        assert db.execute("SELECT count(*) FROM chunks").fetchone()[0] == CHUNKS
    return data


def counted_fetch(monkeypatch, allow=True):
    from winding_field import io

    fetch, requests = io.fetch, []

    def counted(url, **kwargs):
        if url.rsplit("/", 1)[-1].startswith(".z"):
            return fetch(url, **kwargs)  # array metadata, read once per Volume
        requests.append(url)
        if not allow:
            raise AssertionError("remote request for a ledger-pinned cached chunk: " + url)
        return fetch(url, **kwargs)

    monkeypatch.setattr(io, "fetch", counted)
    return requests


def test_trusted_ledger_accepts_matching_cache_without_remote_requests(served, monkeypatch):
    from winding_field.io import Volume

    root, url, _ = served
    data = warm(root, url)
    requests = counted_fetch(monkeypatch, allow=False)
    with closing(Volume(url, cache=root / "cache", trust_ledger=True)) as volume:
        volume.bind_ledger(root / "source.sqlite")
        volume.validate_used()
        assert requests == []
        assert volume.stats()["ledger_hits"] == CHUNKS
        assert volume.stats()["downloaded_bytes"] == 0
        # The run consumed exactly the pinned bytes; a later source change does not reach it.
        (root / "ct.zarr/0/0.0.0").write_bytes(bytes([255]) * 12)
        volume.validate_used()
        np.testing.assert_array_equal(volume.read([0, 0, 0], [4, 4, 6]), data)
        assert requests == []


def test_default_volume_still_rechecks_every_chunk_remotely(served, monkeypatch):
    from winding_field.io import Volume

    root, url, _ = served
    warm(root, url)
    requests = counted_fetch(monkeypatch)
    with closing(Volume(url, cache=root / "cache")) as volume:
        volume.bind_ledger(root / "source.sqlite")
        volume.validate_used()
        assert len(requests) == CHUNKS - 1  # the pinned miss has no cached payload to check
        assert volume.stats()["ledger_hits"] == 0
        (root / "ct.zarr/0/0.0.0").write_bytes(bytes([255]) * 12)
        with pytest.raises(ValueError, match="source chunk changed"):
            volume.validate_used()


def test_trusted_ledger_rejects_a_cache_that_differs_from_its_pin(served, monkeypatch):
    from winding_field.io import Volume

    root, url, _ = served
    warm(root, url)
    with sqlite3.connect(root / "source.sqlite") as db:
        db.execute("UPDATE chunks SET sha=? WHERE key='0/0.0.0'", ("0" * 64,))
    requests = counted_fetch(monkeypatch)
    with closing(Volume(url, cache=root / "cache", trust_ledger=True)) as volume:
        volume.bind_ledger(root / "source.sqlite")
        with pytest.raises(ValueError, match="source chunk changed: 0/0.0.0"):
            volume.validate_used()
        # Only the mismatching chunk went back to the source.
        assert requests == [url + "/0/0.0.0"]


def test_trusted_ledger_still_hashes_the_cached_payload(served):
    from winding_field.io import Volume

    root, url, _ = served
    warm(root, url)
    with closing(Volume(url, cache=root / "cache", trust_ledger=True)) as volume:
        volume.bind_ledger(root / "source.sqlite")
        volume._native.path(url + "/0/0.0.0").write_bytes(bytes([255]) * 12)
        with pytest.raises(ValueError, match="corrupt native CT cache"):
            volume.validate_used()


def test_offline_validation_of_a_complete_cache_matches_online(served, monkeypatch):
    from winding_field.io import Volume

    root, url, _ = served
    data = warm(root, url)
    counted_fetch(monkeypatch, allow=False)
    with closing(Volume(url, cache=root / "cache", offline=True, trust_ledger=True)) as volume:
        volume.bind_ledger(root / "source.sqlite")
        volume.validate_used()
        np.testing.assert_array_equal(volume.read([0, 0, 0], [4, 4, 6]), data)
    # An incomplete cache is an offline miss, never a fetch.
    for blob in (root / "cache/vc-reader/encoded").glob("*.blob"):
        blob.unlink()
    with closing(Volume(url, cache=root / "cache", offline=True, trust_ledger=True)) as volume:
        volume.bind_ledger(root / "source.sqlite")
        with pytest.raises(FileNotFoundError, match="offline"):
            volume.validate_used()


def test_remote_checks_run_concurrently_on_reused_transports(served, monkeypatch):
    from winding_field.ct_reader import EncodedTransport
    from winding_field.io import VALIDATE_WORKERS, Volume

    root, url, _ = served
    warm(root, url)
    assert CHUNKS < VALIDATE_WORKERS
    read = EncodedTransport.read
    # Every check must be in flight at once before any may finish: a sequential loop breaks
    # the barrier instead of passing. The wait bound only turns a hang into a failure.
    barrier = threading.Barrier(CHUNKS)
    transports, threads = set(), set()

    def gated(self, url, *args):
        transports.add(id(self))
        threads.add(threading.get_ident())
        barrier.wait(60)
        return read(self, url, *args)

    monkeypatch.setattr(EncodedTransport, "read", gated)
    with closing(Volume(url, cache=root / "cache")) as volume:
        volume.bind_ledger(root / "source.sqlite")
        volume.validate_used()
        assert len(threads) == len(transports) == CHUNKS
        pool = list(volume._pool_transports)
        monkeypatch.setattr(EncodedTransport, "read", read)
        for _ in range(3):
            volume.validate_used()
        # Spare transports are reused by later calls instead of opening new connections.
        assert len(volume._pool_transports) <= CHUNKS
        assert volume._pool_transports[: len(pool)] == pool


def test_parallel_validation_raises_the_first_error_in_ledger_order(served, monkeypatch):
    from winding_field.io import Volume

    root, url, _ = served
    warm(root, url)
    with closing(Volume(url, cache=root / "cache")) as volume:
        volume.bind_ledger(root / "source.sqlite")
        keys = [key for (key,) in volume._ledger.execute("SELECT key FROM chunks")]
        failing = {keys[2], keys[-1]}
        checked = []
        lock = threading.Lock()

        def check(key):
            with lock:
                checked.append(key)
            if key in failing:
                raise ValueError("bad " + key)

        monkeypatch.setattr(volume, "_chunk_bytes", check)
        for _ in range(3):
            with pytest.raises(ValueError, match="bad " + keys[2]):
                volume.validate_used()
