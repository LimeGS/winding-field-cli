import numpy as np


def test_joint_map_reads_and_sparse_sampling_load_each_tile_once(tmp_path,monkeypatch):
    from winding_field.artifacts import Tiles
    maps=Tiles(tmp_path/'maps',shape=(4,4,1004),chunk=4,identity={})
    for key in ((0,0,0),(0,0,250)):
        lo,hi=maps.box(key)
        x=np.broadcast_to(np.arange(lo[2],hi[2]),(4,4,4)).astype(np.float32)
        maps.write(key,dict(density=x,support=np.ones_like(x,bool),normals=np.stack([x,x,x],axis=-1)))
    load=maps.load; calls=[]
    def counted(key,names=None):
        calls.append(tuple(key)); return load(key,names)
    monkeypatch.setattr(maps,'load',counted)
    data=maps.read_many(('normals','density','support'),[0,0,0],[4,4,4])
    assert len(calls)==1 and data['support'].all()
    calls.clear()
    data=maps.sample(('density','support'),[np.array([1.5]),np.array([1.5]),np.array([1.5,1001.5])])
    np.testing.assert_allclose(data['density'],[[[1.5,1001.5]]])
    assert data['support'].all()
    assert sorted(calls)==[(0,0,0),(0,0,250)]
    maps.close()
