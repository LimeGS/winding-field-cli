"""Native sampling must consume verified chunks and preserve CT support."""
import importlib.util
import itertools
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_io import served

from winding_field.original_recipe import sample_points


def sampler_class():
    assert importlib.util.find_spec('winding_field.native_ct_sampling'), 'verified native sampler not implemented'
    from winding_field.native_ct_sampling import NativeCTSampler
    return NativeCTSampler


@pytest.mark.parametrize('dtype',['u1','<u2','<f4'])
def test_native_sampler_support_values_prefetch_and_concurrency(dtype):
    sampler = sampler_class()
    pytest.importorskip('vc_reader._reader')
    shape=(8,8,8);chunks=(4,4,4)
    arrays={i:np.asfortranarray((np.arange(64).reshape(chunks)+sum(i)*30).astype(dtype))
            for i in itertools.product(range(2),repeat=3)}
    arrays[(1,1,1)]=None
    points=np.r_[np.array([[0,0,0],[3,4,4],[3+1e-10,4,4],[3.5,3.5,3.5],
                           [3,3,3],[7,7,7],[-1e-10,1,1],[np.nan,1,1],[np.inf,1,1]]),
                 np.random.default_rng(12).uniform(0,7,(3000,3))]
    ready=set();batches=[]
    def prefetch(indices):
        assert 1<=len(indices)<=4
        ready.update(indices);batches.append(indices)
    def chunk(index):
        assert index in ready
        return arrays[index]
    expected=sample_points(points,shape,chunks,arrays.get)
    native=sampler(shape,chunks,np.dtype(dtype),chunk)
    try:
        actual=native.sample(points,prefetch=prefetch)
        assert batches and actual[0].dtype==np.float32
        np.testing.assert_array_equal(actual[1],expected[1])
        np.testing.assert_allclose(actual[0],expected[0],rtol=0,atol=2e-5,equal_nan=True)
        assert actual[1][0] and actual[0][0]==0
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda _:native.sample(points,prefetch=prefetch),range(2)))
        for result in results:
            for a,b in zip(result,actual):np.testing.assert_array_equal(a,b)
        values,valid=native.sample(np.empty((0,3)))
        assert values.shape==valid.shape==(0,)
        with pytest.raises(ValueError,match='100000'):native.sample(np.zeros((100001,3)))
    finally:native.close()


def test_native_support_does_not_load_zero_weight_chunk():
    sampler=sampler_class();pytest.importorskip('vc_reader._reader')
    def chunk(index):
        assert index==(0,0,0), 'zero-weight corner requested'
        return np.zeros((4,4,4),'u1')
    native=sampler((8,8,8),(4,4,4),np.dtype('u1'),chunk)
    try:
        values,valid=native.sample(np.array([[3.,3.,3.]]))
        assert valid.tolist()==[True] and values.tolist()==[0.]
    finally:native.close()


def test_patch_native_hashes_detect_change_after_eviction(tmp_path):
    sampler_class();pytest.importorskip('vc_reader._reader')
    from winding_field.patch_pipeline import PatchCT,settings
    source=tmp_path/'source';source.mkdir()
    (source/'.zarray').write_text(json.dumps(dict(zarr_format=2,shape=[4]*3,chunks=[4]*3,dtype='|u1',
        compressor=None,fill_value=0,filters=None,order='C')))
    (source/'0.0.0').write_bytes(bytes([10])*64)
    output=tmp_path/'output';output.mkdir()
    cfg=settings(dict(output_frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0]),
        app_run='/unused',cache_dir=str(tmp_path/'cache'),cache_mib=0,tile_size=8))
    ct=PatchCT(str(source),output,cfg,voxel_um=1.)
    try:
        assert ct.identity['sampler']=='vc_reader-float64-support-v1'
        values,valid=ct.sample(np.array([[1.25,1.5,1.75]]),cfg['output_frame'])
        assert valid.tolist()==[True] and values.tolist()==[10.]
        assert ct.db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]==1
        (source/'0.0.0').write_bytes(bytes([11])*64)
        with pytest.raises(ValueError,match='CT chunk changed'):
            ct.sample(np.array([[1.25,1.5,1.75]]),cfg['output_frame'])
    finally:ct.close()


def test_patch_remote_cache_preserves_missing_and_real_zero(served):
    pytest.importorskip('vc_reader._reader')
    from winding_field.patch_pipeline import PatchCT,settings
    root,url,_=served
    (root/'ct.zarr/0/0.0.0').write_bytes(bytes(12))
    (root/'ct.zarr/0/1.1.1').unlink()
    output=root/'patch';output.mkdir()
    cfg=settings(dict(output_frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0]),
        app_run='/unused',cache_dir=str(root/'patch-cache'),cache_mib=0,ct_workers=4,tile_size=8))
    ct=PatchCT(url,output,cfg)
    try:
        points=np.array([[10.,20.,30.],[20.,29.,42.]])
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda _:ct.sample(points,cfg['output_frame']),range(2)))
        for values,valid in results:
            assert valid.tolist()==[True,False]
            assert values[0]==0 and np.isnan(values[1])
        assert ct.db.execute('SELECT COUNT(*) FROM chunks WHERE sha IS NULL').fetchone()[0]==1
        path=next(Path(ct._chunk_layers[0].root).rglob('0.0.0'))
        path.write_bytes(bytes([1])*12)
        with pytest.raises(ValueError,match='CT chunk changed'):
            ct.sample(points[:1],cfg['output_frame'])
    finally:ct.close()


@pytest.mark.parametrize('order',['C','F'])
def test_patch_local_compressed_chunk_layout(tmp_path,order):
    pytest.importorskip('vc_reader._reader')
    from numcodecs import Blosc
    from winding_field.patch_pipeline import PatchCT,settings
    source=tmp_path/'source';source.mkdir()
    codec=Blosc(cname='zstd',shuffle=Blosc.SHUFFLE)
    data=np.arange(24,dtype='<u2').reshape(2,3,4)
    (source/'.zarray').write_text(json.dumps(dict(zarr_format=2,shape=[2,3,4],chunks=[2,3,4],dtype='<u2',
        compressor=codec.get_config(),fill_value=0,filters=None,order=order)))
    (source/'0.0.0').write_bytes(codec.encode(np.frombuffer(data.tobytes(order=order),dtype='<u2')))
    output=tmp_path/'output';output.mkdir()
    cfg=settings(dict(output_frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0]),app_run='/unused',
        cache_dir=str(tmp_path/'cache'),tile_size=8))
    ct=PatchCT(str(source),output,cfg,voxel_um=1.)
    try:
        values,valid=ct.sample(np.array([[1.5,1.5,.5],[3.,2.,1.]]),cfg['output_frame'])
        np.testing.assert_array_equal(valid,[True,True])
        np.testing.assert_array_equal(values,[13.5,23.])
    finally:ct.close()


def test_patch_failed_ledger_initialization_closes_connections(tmp_path,monkeypatch):
    pytest.importorskip('vc_reader._reader')
    import sqlite3
    from winding_field.patch_pipeline import PatchCT,settings
    source=tmp_path/'source';source.mkdir()
    (source/'.zarray').write_text(json.dumps(dict(zarr_format=2,shape=[2]*3,chunks=[2]*3,dtype='|u1',
        compressor=None,fill_value=0,filters=None,order='C')))
    output=tmp_path/'output';output.mkdir()
    db=sqlite3.connect(output/'ct-chunks.sqlite');db.execute('CREATE TABLE chunks (key TEXT,sha TEXT)');db.close()
    cfg=settings(dict(output_frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0]),app_run='/unused',
        cache_dir=str(tmp_path/'cache'),tile_size=8))
    connect=sqlite3.connect;opened=[]
    def track(*a,**kw):
        connection=connect(*a,**kw);opened.append(connection);return connection
    monkeypatch.setattr(sqlite3,'connect',track)
    with pytest.raises(sqlite3.OperationalError,match='already exists'):
        PatchCT(str(source),output,cfg,voxel_um=1.)
    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError,match='closed'):connection.execute('SELECT 1')


@pytest.mark.parametrize('cached',['stale-missing','lossless-delta','lossy-delta'])
def test_native_cache_validates_payload_before_missing_marker(served,cached):
    pytest.importorskip('vc_reader._reader')
    from vc_reader.reader.writer import compress_bytes
    from winding_field.patch_pipeline import PatchCT,settings
    root,url,expected=served
    cfg=settings(dict(output_frame=dict(id='test',voxel_um=1.,origin_xyz_um=[0,0,0]),app_run='/unused',
        cache_dir=str(root/'cache'),cache_mib=0,tile_size=8))
    output=root/'first';output.mkdir();ct=PatchCT(url,output,cfg)
    try:
        ct._chunk((0,0,0));cache=Path(ct._chunk_layers[0].root)
    finally:ct.close()
    if cached=='stale-missing':(cache/'0.0.0.missing').write_bytes(b'')
    else:
        metadata=json.loads((cache/'.zarray').read_text());metadata['compressor']={'id':'vc-delta3d','quant':1}
        (cache/'.zarray').write_text(json.dumps(metadata))
        data=expected[:2,:2,:3].copy()
        if cached=='lossy-delta':data=data//3*3
        encoded=bytearray(compress_bytes(data,{'id':'vc-delta3d'},shape=(2,2,3)))
        assert encoded[:5]==b'D3D1\x01'
        if cached=='lossy-delta':encoded[6]=3
        (cache/'0.0.0').write_bytes(encoded)
    output=root/'second';output.mkdir();ct=PatchCT(url,output,cfg)
    try:
        if cached=='lossy-delta':
            with pytest.raises(ValueError,match='lossless'):ct._chunk((0,0,0))
            assert ct.db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]==0
        else:np.testing.assert_array_equal(ct._chunk((0,0,0)),expected[:2,:2,:3])
    finally:ct.close()
