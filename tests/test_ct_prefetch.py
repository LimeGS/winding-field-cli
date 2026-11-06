import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import pytest
from winding_field.original_recipe import sample_points
from winding_field.patch_pipeline import PatchCT


def test_prefetch_preserves_scalar_order_and_missing_support():
    xyz=np.random.default_rng(7).uniform(0,7,(1000,3))
    def chunk(index):
        return None if index==(1,1,1) else np.full((2,2,2),sum(index)*17,dtype='u1')
    expected=sample_points(xyz,(8,8,8),(2,2,2),chunk)
    ready=set();batches=[]
    def prefetch(indices):
        assert 1<=len(indices)<=4
        batches.append(indices);ready.update(indices)
    def checked(index):
        assert index in ready
        return chunk(index)
    actual=sample_points(xyz,(8,8,8),(2,2,2),checked,prefetch=prefetch)
    assert len(batches)>1
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_native_prefetch_four_concurrent_loads_and_publication(tmp_path):
    ct=object.__new__(PatchCT);ct._lock=threading.RLock();ct._chunks={}
    ct._cache_path=tmp_path;ct._disk_reserve=0
    ct.volume=SimpleNamespace(chunks=(2,2,2),dtype=np.dtype('u1'))
    barrier=threading.Barrier(4);published=[]
    def load(index):barrier.wait(timeout=5);published.append(tuple(index));return bytes(8)
    ct._load_chunk=load
    with ThreadPoolExecutor(max_workers=4) as pool:
        ct._ct_pool=pool
        ct._prefetch([(0,0,i) for i in range(4)])
    assert sorted(published)==[(0,0,i) for i in range(4)]


def test_prefetch_reserve_failure_before_network(tmp_path,monkeypatch):
    from winding_field import patch_pipeline as p
    ct=object.__new__(PatchCT);ct._lock=threading.RLock();ct._chunks={}
    ct._cache_path=tmp_path;ct._disk_reserve=100
    ct.volume=SimpleNamespace(chunks=(2,2,2),dtype=np.dtype('u1'))
    monkeypatch.setattr(p.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    with pytest.raises(RuntimeError,match='CT cache disk reserve'):ct._prefetch([(0,0,0)])


def test_prefetch_propagates_publication_failure(tmp_path):
    ct=object.__new__(PatchCT);ct._lock=threading.RLock();ct._chunks={}
    ct._cache_path=tmp_path;ct._disk_reserve=0
    ct.volume=SimpleNamespace(chunks=(2,2,2),dtype=np.dtype('u1'))
    def fail(index):raise RuntimeError('CT prefetch publication failed')
    ct._load_chunk=fail
    with ThreadPoolExecutor(max_workers=4) as pool:
        ct._ct_pool=pool
        with pytest.raises(RuntimeError,match='publication failed'):ct._prefetch([(0,0,i) for i in range(4)])
