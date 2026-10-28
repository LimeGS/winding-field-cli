"""Read-only, SHA-bound field crops over real tiny NPY and checkpoint files."""

import gc
import hashlib
import importlib
import io
import itertools
import json
import os
import tracemalloc
import weakref
from copy import deepcopy

import numpy as np
import pytest

import field_store


def api():
    return importlib.import_module("winding_field.surface_sources")


def geometry(shape=(3, 3, 5)):
    return dict(
        frame="synthetic/custom-XYZ",
        voxel_um=7.25,
        origin_xyz=[11.0, -8.0, 4.0],
        spacing_xyz=[2.0, 3.0, 5.0],
        shape_zyx=list(shape),
        sheet_step=6.0,
    )


def descriptor(path, **extra):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **extra)


def npy(tmp_path, values=None, mask=None):
    if values is None:
        values = np.arange(45, dtype=np.float32).reshape(3, 3, 5)
    path = tmp_path / "field.npy"
    np.save(path, values)
    result = descriptor(path, kind="npy")
    if mask is not None:
        path = tmp_path / "mask.npy"
        np.save(path, mask)
        result["mask"] = descriptor(path)
    return result


def native(tmp_path, *, observed=True, all_chunks=False):
    root = tmp_path / "native"
    root.mkdir()
    meta = geometry()
    chunks = {}
    for key in itertools.product(range(2), range(2), range(3)):
        if not all_chunks and key not in ((0, 0, 0), (1, 1, 2)):
            continue
        shape = tuple(min(2, n - k * 2) for k, n in zip(key, meta["shape_zyx"]))
        z, y, x = np.indices(shape)
        values = 100 * (z + 2 * key[0]) + 10 * (y + 2 * key[1]) + x + 2 * key[2]
        support = values % 2 == 0
        stream = io.BytesIO()
        np.savez_compressed(
            stream,
            delta=values.astype(np.float32),
            offset=np.float64(1e12),
            observed=np.packbits(support) if observed else np.empty(0, np.uint8),
        )
        label = "-".join(map(str, key))
        path = root / f"{label}.g1792680000000000000.npz"
        path.write_bytes(stream.getvalue())
        chunks[label] = dict(descriptor(path), bytes=path.stat().st_size)
        chunks[label]["path"] = path.name
    manifest = dict(
        identity=dict(meta, prepared_sha256="a" * 64),
        shape=meta["shape_zyx"],
        chunk=2,
        chunks=chunks,
        progress={},
        initialization="numerical zero, not an observed value",
    )
    path = root / "CHECKPOINT.json"
    path.write_text(json.dumps(manifest))
    return descriptor(path, kind="chunks", support_policy="observed"), manifest


def publish(field, manifest):
    from pathlib import Path

    path = Path(field["path"])
    path.write_text(json.dumps(manifest))
    field["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def payload(field, manifest, label="0-0-0"):
    from pathlib import Path

    return Path(field["path"]).parent / manifest["chunks"][label]["path"]


@pytest.mark.parametrize("dtype,order", [(np.float32, "C"), (np.float64, "F"), (">f8", "F"), (np.int16, "C")])
def test_npy_crops_keep_geometry_precision_support_and_owned_storage(tmp_path, dtype, order):
    values = np.array(np.arange(45).reshape(3, 3, 5), dtype=dtype, order=order)
    field, meta = npy(tmp_path, values), geometry()
    with api().FieldSource(field, meta) as source:
        assert source.shape == (3, 3, 5)
        np.testing.assert_array_equal(source.origin, [11, -8, 4])
        np.testing.assert_array_equal(source.spacing, [2, 3, 5])
        assert source.origin.dtype == source.spacing.dtype == np.float64
        assert (source.frame, source.voxel_um, source.sheet_step) == (meta["frame"], 7.25, 6.0)
        result, valid = source.read((1, 1, 2), (3, 3, 5))
        np.testing.assert_array_equal(result, [[[22, 23, 24], [27, 28, 29]], [[37, 38, 39], [42, 43, 44]]])
        assert result.dtype == np.float64 and valid.dtype == np.bool_
        assert result.flags.owndata and valid.flags.owndata
        assert result.flags.c_contiguous and valid.all()
        result[:] = -1
        valid[:] = False
        again, supported = source.read((1, 1, 2), (3, 3, 5))
        assert again[0, 0, 0] == 22 and supported.all()
        source.validate()
    source.close()
    for action in (source.validate, lambda: source.read((0, 0, 0), (1, 1, 1)), source.__enter__):
        with pytest.raises(RuntimeError):
            action()


def test_npy_mask_and_nonfinite_holes_survive_cropping(tmp_path):
    values = np.arange(45, dtype=float).reshape(3, 3, 5)
    values[1, 1, 2:5] = [np.nan, np.inf, -np.inf]
    mask = np.ones(values.shape, bool)
    mask[2, 2, 2] = False
    with api().FieldSource(npy(tmp_path, values, mask), geometry()) as source:
        result, valid = source.read((1, 1, 2), (3, 3, 5))
        np.testing.assert_array_equal(result, values[1:, 1:, 2:])
        np.testing.assert_array_equal(
            valid, [[[False, False, False], [True, True, True]], [[True, True, True], [False, True, True]]]
        )
        assert np.isfinite(result[valid]).all()


def test_identity_is_json_detached_from_inputs_and_each_return(tmp_path):
    field, manifest = native(tmp_path)
    meta = geometry()
    with api().FieldSource(field, meta) as source:
        expected = deepcopy(source.identity)
        assert isinstance(expected, dict)
        assert json.loads(json.dumps(expected, allow_nan=False)) == expected
        assert expected["field"] == field and expected["geometry"] == meta
        assert expected["source_identity"] == manifest["identity"]
        field["support_policy"] = "stored"
        meta["spacing_xyz"][0] = 900
        exposed = source.identity
        exposed["field"]["path"] = "changed"
        exposed["source_identity"]["frame"] = "changed"
        assert source.identity == expected
        _, support = source.read((0, 0, 0), (2, 2, 2))
        assert support.sum() == 4


@pytest.mark.parametrize("policy", ["observed", "stored"])
@pytest.mark.parametrize("observed", [True, False])
def test_native_and_public_share_crop_and_support_contract(tmp_path, policy, observed):
    field, _ = native(tmp_path, observed=observed)
    field["support_policy"] = policy
    root = tmp_path / "public"
    with field_store.BlockStore(root, (3, 3, 5), 2, {"grid": geometry()}) as store:
        for key in ((0, 0, 0), (1, 1, 2)):
            shape = tuple(min(2, n - k * 2) for k, n in zip(key, (3, 3, 5)))
            z, y, x = np.indices(shape)
            v = 100 * (z + 2 * key[0]) + 10 * (y + 2 * key[1]) + x + 2 * key[2]
            store.commit(key, 1e12 + v, {}, v % 2 == 0 if observed else None)
    public = descriptor(root / "CHECKPOINT.json", kind="chunks", support_policy=policy)
    with api().FieldSource(field, geometry()) as a, api().FieldSource(public, geometry()) as b:
        for lo, hi in [((0, 0, 0), (3, 3, 5)), ((1, 1, 1), (3, 3, 5)), ((2, 2, 4), (3, 3, 5))]:
            av, am = a.read(lo, hi)
            bv, bm = b.read(lo, hi)
            np.testing.assert_array_equal(av, bv)
            np.testing.assert_array_equal(am, bm)
            assert av.dtype == np.float64 and am.dtype == bool and av.flags.owndata and am.flags.owndata
        v, m = a.read((0, 0, 0), (3, 3, 5))
        assert v[0, 0, 0] == 1e12 and v[2, 2, 4] == 1e12 + 224
        assert m.sum() == (9 if policy == "stored" else 5 if observed else 0)
        assert not m[0, 0, 2] and v[0, 0, 2] == 0


@pytest.mark.parametrize("policy", ["observed", "stored"])
def test_absent_chunks_never_gain_support_and_observed_zero_is_preserved(tmp_path, policy):
    root = tmp_path / "zero"
    with field_store.BlockStore(root, (2, 2, 4), 2, {}) as store:
        store.commit((0, 0, 0), np.zeros((2, 2, 2)), {}, np.ones((2, 2, 2), bool))
    field = descriptor(root / "CHECKPOINT.json", kind="chunks", support_policy=policy)
    with api().FieldSource(field, geometry((2, 2, 4))) as source:
        v, m = source.read((0, 0, 0), (2, 2, 4))
        assert not v.any() and m[:, :, :2].all() and not m[:, :, 2:].any()


@pytest.mark.parametrize(
    "lo,hi",
    [
        ((-1, 0, 0), (1, 1, 1)),
        ((0, 0, 0), (4, 3, 5)),
        ((0, 0, 0), (0, 1, 1)),
        ((0.0, 0, 0), (1, 1, 1)),
        ((False, 0, 0), (1, 1, 1)),
        ((0, 0), (1, 1)),
        ((0, 0, 0), (1, np.nan, 1)),
    ],
)
def test_invalid_bounds_fail_for_both_backends(tmp_path, lo, hi):
    field, _ = native(tmp_path)
    for item in (field, npy(tmp_path)):
        with api().FieldSource(item, geometry()) as source, pytest.raises(ValueError):
            source.read(lo, hi)


@pytest.mark.parametrize(
    "key,value",
    [
        ("frame", ""),
        ("frame", None),
        ("voxel_um", 0),
        ("voxel_um", True),
        ("sheet_step", -1),
        ("sheet_step", np.inf),
        ("origin_xyz", [0, np.nan, 0]),
        ("voxel_um", 10**400),
        ("origin_xyz", [False, 2, 3]),
        ("spacing_xyz", [1, 0, 1]),
        ("origin_xyz", [0, 0]),
        ("shape_zyx", [3.0, 3, 5]),
        ("shape_zyx", [3, True, 5]),
        ("shape_zyx", [3, 0, 5]),
        ("shape_zyx", [2**63, 3, 5]),
    ],
)
def test_invalid_explicit_geometry_is_rejected(tmp_path, key, value):
    meta = geometry()
    meta[key] = value
    with pytest.raises(ValueError):
        api().FieldSource(npy(tmp_path), meta)


@pytest.mark.parametrize("key", list(geometry()))
def test_every_geometry_field_is_required(tmp_path, key):
    meta = geometry()
    del meta[key]
    with pytest.raises(ValueError):
        api().FieldSource(npy(tmp_path), meta)


@pytest.mark.parametrize(
    "key,value",
    [
        ("frame", "foreign"),
        ("voxel_um", 1),
        ("sheet_step", 1),
        ("origin_xyz", [0, 0, 0]),
        ("spacing_xyz", [1, 1, 1]),
        ("shape_zyx", [2, 3, 5]),
    ],
)
@pytest.mark.parametrize("container", ["identity", "source_identity", "grid", "geometry"])
def test_declared_embedded_geometry_cannot_be_overridden(tmp_path, key, value, container):
    field, manifest = native(tmp_path)
    if container == "identity":
        target = manifest["identity"]
    else:
        manifest["identity"][container] = {}
        target = manifest["identity"][container]
    target[key] = value
    publish(field, manifest)
    with pytest.raises(ValueError):
        api().FieldSource(field, geometry())


@pytest.mark.parametrize(
    "change", ["missing_hash", "bad_hash", "wrong_hash", "relative", "kind", "mask_hash", "policy"]
)
def test_invalid_descriptors_are_rejected(tmp_path, change):
    field = npy(tmp_path, mask=np.ones((3, 3, 5), bool))
    if change == "missing_hash":
        del field["sha256"]
    elif change == "bad_hash":
        field["sha256"] = "z" * 64
    elif change == "wrong_hash":
        field["sha256"] = "0" * 64
    elif change == "relative":
        field["path"] = "field.npy"
    elif change == "kind":
        field["kind"] = "zarr"
    elif change == "mask_hash":
        field["mask"]["sha256"] = "0" * 64
    else:
        field, _ = native(tmp_path)
        del field["support_policy"]
    with pytest.raises(ValueError):
        api().FieldSource(field, geometry())


@pytest.mark.parametrize(
    "change", ["complex", "object", "boolean", "shape", "mask_dtype", "mask_shape", "npz", "trailing", "truncated"]
)
def test_hash_does_not_excuse_invalid_npy_header_dtype_or_length(tmp_path, change):
    values = np.ones((3, 3, 5), dtype={"complex": complex, "object": object, "boolean": bool}.get(change, float))
    if change == "shape":
        values = values[:, :, :2]
    mask = np.ones((3, 3, 5), np.uint8) if change == "mask_dtype" else None
    if change == "mask_shape":
        mask = np.ones((1, 1, 1), bool)
    field = npy(tmp_path, values, mask)
    from pathlib import Path

    path = Path(field["path"])
    if change == "npz":
        with path.open("wb") as stream:
            np.savez(stream, data=values)
    elif change == "trailing":
        path.write_bytes(path.read_bytes() + b"extra")
    elif change == "truncated":
        path.write_bytes(path.read_bytes()[:-8])
    field["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        api().FieldSource(field, geometry())


@pytest.mark.parametrize(
    "change",
    [
        "path",
        "symlink",
        "key",
        "out_of_bounds",
        "bytes",
        "oversize",
        "missing",
        "hash",
        "record",
        "progress",
        "chunk",
        "shape",
        "quantization",
    ],
)
def test_all_manifest_paths_sizes_and_keys_are_checked_before_first_output(tmp_path, change):
    field, manifest = native(tmp_path)
    record = manifest["chunks"]["1-1-2"]
    if change == "path":
        record["path"] = "../escape.npz"
    elif change == "symlink":
        path = payload(field, manifest, "1-1-2")
        other = tmp_path / "outside.npz"
        path.rename(other)
        path.symlink_to(other)
    elif change in ("key", "out_of_bounds"):
        manifest["chunks"]["01-1-2" if change == "key" else "2-1-2"] = manifest["chunks"].pop("1-1-2")
    elif change == "bytes":
        record["bytes"] += 1
    elif change == "oversize":
        record["bytes"] = 2**40
    elif change == "missing":
        payload(field, manifest, "1-1-2").unlink()
    elif change == "hash":
        record["sha256"] = "bad"
    elif change == "record":
        del record["bytes"]
    elif change == "progress":
        manifest["progress"] = []
    elif change == "chunk":
        manifest["chunk"] = True
    elif change == "shape":
        manifest["shape"] = [3, 3, 6]
    else:
        record["quantization_max"] = -1
    publish(field, manifest)
    with pytest.raises((ValueError, OSError)):
        api().FieldSource(field, geometry())


@pytest.mark.parametrize("change", ["digest", "delta", "padding", "members"])
def test_payload_hash_and_safe_decoder_are_enforced_on_read(tmp_path, change):
    field, manifest = native(tmp_path)
    path = payload(field, manifest)
    if change == "digest":
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    else:
        stream = io.BytesIO()
        arrays = dict(
            delta=np.zeros((2, 2, 2), np.float64 if change == "delta" else np.float32),
            offset=np.float64(4),
            observed=np.array([255], np.uint8),
        )
        label = "0-0-0"
        if change == "padding":
            label = "1-1-2"
            path = payload(field, manifest, label)
            arrays["delta"] = np.zeros((1, 1, 1), np.float32)
            arrays["observed"] = np.array([129], np.uint8)
        if change == "members":
            arrays["foreign"] = np.zeros(1)
        np.savez_compressed(stream, **arrays)
        path.write_bytes(stream.getvalue())
        manifest["chunks"][label].update(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        publish(field, manifest)
    with api().FieldSource(field, geometry()) as source, pytest.raises(ValueError):
        source.read((0, 0, 0), (3, 3, 5))


@pytest.mark.parametrize("backend", ["field", "mask", "manifest", "payload"])
@pytest.mark.parametrize("change", ["mutate", "replace", "truncate", "header"])
def test_immutable_sources_reject_changes_even_with_restored_mtime(tmp_path, backend, change):
    from pathlib import Path

    if backend in ("field", "mask"):
        field = npy(tmp_path, mask=np.ones((3, 3, 5), bool))
        path = Path(field["path"] if backend == "field" else field["mask"]["path"])
    else:
        field, manifest = native(tmp_path)
        path = Path(field["path"]) if backend == "manifest" else payload(field, manifest)
    with api().FieldSource(field, geometry(), cache_bytes=4096) as source:
        source.read((0, 0, 0), (2, 2, 2))
        before = path.stat()
        raw = bytearray(path.read_bytes())
        if change == "replace":
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(raw)
            replacement.replace(path)
        elif change == "truncate":
            path.write_bytes(raw[:-8])
        else:
            raw[10 if change == "header" else -1] ^= 1
            path.write_bytes(raw)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with pytest.raises((ValueError, OSError)):
            source.read((0, 0, 0), (2, 2, 2))
        with pytest.raises((ValueError, OSError)):
            source.validate()


def test_hash_and_decode_use_same_payload_bytes_during_replacement(tmp_path, monkeypatch):
    field, manifest = native(tmp_path)
    path = payload(field, manifest)
    original, real_decode = path.read_bytes(), field_store._decode

    def replace_after_read(raw, shape):
        assert raw == original
        path.write_bytes(b"changed after read")
        return real_decode(raw, shape)

    with api().FieldSource(field, geometry()) as source:
        monkeypatch.setattr(field_store, "_decode", replace_after_read)
        try:
            values, _ = source.read((0, 0, 0), (2, 2, 2))
        except ValueError:
            pass  # Detecting the concurrent mutation is also valid.
        else:
            assert values[0, 0, 0] == 1e12
        with pytest.raises((ValueError, OSError)):
            source.read((0, 0, 0), (2, 2, 2))


def test_npy_hash_and_memmap_share_open_inode(tmp_path, monkeypatch):
    from pathlib import Path

    field = npy(tmp_path)
    path = Path(field["path"])
    digest = hashlib.file_digest

    def replace_after_hash(stream, *args, **kwargs):
        result = digest(stream, *args, **kwargs)
        replacement = path.with_suffix(".replacement.npy")
        np.save(replacement, np.full((3, 3, 5), 999.0, np.float32))
        replacement.replace(path)
        return result

    monkeypatch.setattr(hashlib, "file_digest", replace_after_hash)
    with pytest.raises(ValueError):
        api().FieldSource(field, geometry())


@pytest.mark.parametrize("cache_bytes", [0, 1, 600, 1200])
def test_default_retains_nothing_and_optional_lru_bounds_live_chunks(tmp_path, monkeypatch, cache_bytes):
    field, _ = native(tmp_path, all_chunks=True)
    refs, loads = [], []
    decode = field_store._decode

    def track(raw, shape):
        decoded = decode(raw, shape)
        refs.append(weakref.ref(decoded[0]))
        loads.append(1)
        live = [ref() for ref in refs if ref() is not None]
        assert sum(a.nbytes + (a.size + 7) // 8 + 512 for a in live) <= cache_bytes + 545
        return decoded

    monkeypatch.setattr(field_store, "_decode", track)
    with api().FieldSource(field, geometry(), cache_bytes=cache_bytes) as source:
        for _ in range(2):
            for key in itertools.product(range(2), range(2), range(3)):
                lo = tuple(k * 2 for k in key)
                hi = tuple(min(a + 2, n) for a, n in zip(lo, (3, 3, 5)))
                source.read(lo, hi)
                gc.collect()
                live = [ref() for ref in refs if ref() is not None]
                assert sum(a.nbytes + (a.size + 7) // 8 + 512 for a in live) <= cache_bytes
                del live
        count = len(loads)
        source.read((2, 2, 4), (3, 3, 5))
        assert len(loads) == count + (1 if cache_bytes < 517 else 0)
    gc.collect()
    assert not any(ref() is not None for ref in refs)


def test_npy_open_and_small_crop_do_not_materialize_the_full_volume(tmp_path):
    field = npy(tmp_path, np.zeros((128, 128, 128), np.float64))
    tracemalloc.start()
    try:
        with api().FieldSource(field, geometry((128, 128, 128))) as source:
            values, valid = source.read((100, 100, 100), (102, 103, 104))
            assert values.shape == (2, 3, 4) and not values.any() and valid.all()
            _, peak = tracemalloc.get_traced_memory()
            assert peak < 2 * 1024**2  # The pinned source is 16 MiB, not a resident array.
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("cache_bytes", [-1, True, 1.5])
def test_invalid_cache_budget_is_rejected(tmp_path, cache_bytes):
    with pytest.raises(ValueError):
        api().FieldSource(npy(tmp_path), geometry(), cache_bytes=cache_bytes)


def test_read_only_sources_never_create_locks_publish_or_modify_any_bytes(tmp_path, monkeypatch):
    field, _ = native(tmp_path)
    npy_field = npy(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    real_open = os.open

    def read_only_open(path, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", read_only_open)
    for item in (field, npy_field):
        with api().FieldSource(item, geometry()) as source:
            source.read((0, 0, 0), (3, 3, 5))
            source.validate()
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
