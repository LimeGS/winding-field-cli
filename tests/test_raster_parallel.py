import threading
import numpy as np
import pytest
from winding_field import patch_pipeline as p


def mapped_square():
    uv=np.array([[0,0],[10,0],[10,10],[0,10]],float)
    return dict(uv=uv,world=np.column_stack((uv,np.zeros(4))),faces=np.array([[0,1,2],[0,2,3]]),
                display=np.ones(2,bool),source_face_rows=np.array([17,29]))


def test_raster_workers_overlap_but_preserve_tile_order_and_mapping(tmp_path):
    frame=dict(voxel_um=1.,origin_xyz_um=[0,0,0]);mapped=mapped_square()
    def sample(xyz):return xyz[:,0]+xyz[:,1],np.ones(len(xyz),bool)
    serial=p.render_map(mapped,tmp_path/'serial',sample,frame,pitch_um=1.,tile_size=4)
    lock=threading.Lock();barrier=threading.Barrier(2);calls=[0]
    def concurrent(xyz):
        with lock:calls[0]+=1;number=calls[0]
        if number<=2:barrier.wait(timeout=10)
        return sample(xyz)
    parallel=p.render_map(mapped,tmp_path/'parallel',concurrent,frame,pitch_um=1.,tile_size=4,raster_workers=2)
    assert [t['directory'] for t in serial['tiles']]==[t['directory'] for t in parallel['tiles']]
    for tile in serial['tiles']:
        name=tile['directory']
        with np.load(tmp_path/'serial'/name/'mapping.npz') as a,np.load(tmp_path/'parallel'/name/'mapping.npz') as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        assert (tmp_path/'serial'/name/'image.png').read_bytes()==(tmp_path/'parallel'/name/'image.png').read_bytes()


def test_raster_worker_failure_propagates(tmp_path):
    def fail(xyz):raise OSError('injected CT read failure')
    with pytest.raises(OSError,match='injected CT'):
        p.render_map(mapped_square(),tmp_path/'failed',fail,dict(voxel_um=1.),pitch_um=1.,tile_size=4,raster_workers=2)


def test_raster_memory_scales_with_workers():
    cfg=dict(tile_size=1024,cache_mib=512)
    assert p.raster_memory(dict(cfg,raster_workers=3))-p.raster_memory(cfg)>=2*((1024+4)**2*160+100000*1024)


def test_parallel_admission_reserves_dense_sampling_workspace():
    with pytest.raises(ValueError,match='memory budget'):
        p.ct_memory((2,2,2),'u1',dict(tile_size=320,cache_mib=0,raster_workers=4,memory_gib=.25))


def test_failure_cancels_running_worker_without_later_receipts(tmp_path,monkeypatch):
    started=threading.Event();cancelled=threading.Event()
    def raster(*args,guard,**kwargs):
        if args[5]['column_offset']==0:
            assert started.wait(5)
            raise OSError('first tile failed')
        started.set()
        try:
            for _ in range(500):
                guard();cancelled.wait(.01)
            pytest.fail('worker did not observe cancellation')
        except InterruptedError:
            cancelled.set();raise
    monkeypatch.setattr(p,'raster_sheet',raster)
    with pytest.raises(OSError,match='first tile failed'):
        p.render_map(mapped_square(),tmp_path/'failed',None,dict(voxel_um=1.),pitch_um=1.,tile_size=4,raster_workers=2)
    assert cancelled.is_set()
    assert not list((tmp_path/'failed').rglob('TILE.json'))


def test_shared_ct_samples_use_one_verified_load(tmp_path):
    from collections import OrderedDict
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import sqlite3
    ct=object.__new__(p.PatchCT)
    ct._lock=threading.RLock();ct._chunks=OrderedDict();ct._cache_bytes=0;ct._cache_limit=4096
    ct._cache_path=tmp_path;ct._disk_reserve=0
    ct.db=sqlite3.connect(tmp_path/'chunks.sqlite',check_same_thread=False)
    ct.db.execute('CREATE TABLE chunks(key TEXT PRIMARY KEY,sha TEXT)')
    ct.volume=SimpleNamespace(chunks=(2,2,2),shape=(2,2,2),dtype=np.dtype('u1'),metadata={'order':'C'},
        origin_xyz=np.zeros(3),spacing_xyz=np.ones(3))
    calls=[];barrier=threading.Barrier(2)
    def load(index):
        calls.append(index)
        return bytes(range(8))
    ct._load_chunk=load
    from winding_field.native_ct_sampling import NativeCTSampler
    ct._sampler=NativeCTSampler(ct.volume.shape,ct.volume.chunks,ct.volume.dtype,ct._chunk)
    def run():
        barrier.wait(5)
        return ct.sample(np.array([[.5,.5,.5]]),dict(voxel_um=1.,origin_xyz_um=[0,0,0]))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run) for _ in range(2)]
            for future in futures:
                values,valid=future.result()
                np.testing.assert_array_equal(values,[3.5]);assert valid.all()
        assert len(calls)==1 and ct.db.execute('SELECT COUNT(*) FROM chunks').fetchone()==(1,)
    finally:ct._sampler.close();ct.db.close()
