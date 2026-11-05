import numpy as np
import pytest


def test_region_seed_order_is_global_priority_then_centrality_then_coordinate(tmp_path):
    from winding_field.bounded_seeds import seeds
    class Fields:
        shape=(4,4,8)
        def centrality(self,lo,hi,family):
            cent=np.zeros(tuple(hi-lo),'u2');ridge=np.zeros(tuple(hi-lo),'u1')
            for xyz,priority,value in [((0,0,1),1,200),((2,0,0),2,10),
                                       ((0,0,0),2,10),((0,0,5),2,500)]:
                q=np.asarray(xyz)
                if np.all(q>=lo) and np.all(q<hi):
                    at=tuple(q-lo);cent[at]=value;ridge[at]=priority
            return cent,ridge
    result=list(seeds(Fields(),1,tmp_path,quota_bytes=1<<20,region_size=4,chunk=2))
    assert result==[(0,(0,0,0),2),(1,(2,0,0),2),(2,(0,0,1),1),(3,(0,0,5),2)]
    assert list(tmp_path.iterdir())==[]
    with pytest.raises(OSError,match='quota'):
        list(seeds(Fields(),1,tmp_path,quota_bytes=1,region_size=4,chunk=2))


def test_seed_stencils_and_order_match_dense_reference_across_regions(tmp_path):
    import itertools
    from winding_field.bounded_seeds import seeds
    from winding_field.bounded_qual import QualFields
    from avf_spline_generator.prepare import tensor_axes
    from avf_spline_generator.core import stage1_centrality_and_ridge
    rng=np.random.default_rng(56)
    native=rng.uniform(0,1,(2,72,72,72)).astype('f4')
    fields=QualFields((72,72,72),lambda lo,hi:native[(slice(None),)+tuple(slice(a,b) for a,b in zip(lo,hi))],
                      chunk=8,cache_bytes=4096)
    pooled=np.zeros((2,36,36,36),'f4')
    for z,y,x in itertools.product((0,1),repeat=3):pooled+=native[:,z::2,y::2,x::2]
    pooled=np.rint(pooled/8*255).clip(0,255).astype('u1')
    presence=np.minimum(pooled[0].astype('u2')+pooled[1],255).astype('u1')
    presence[pooled[0]<pooled[1]]=0
    cent,ridge=stage1_centrality_and_ridge(presence,*tensor_axes(pooled,(slice(None),)*3),2)
    coordinates=np.argwhere(ridge>0)
    assert len(coordinates)>0
    expected=sorted((tuple(map(int,q)) for q in coordinates),key=lambda q:(
        *(v//16 for v in q),-int(ridge[q]),-int(cent[q]),*q))
    actual=list(seeds(fields,1,tmp_path,quota_bytes=8<<20,region_size=16,chunk=8))
    assert [q for _,q,_ in actual]==expected
    assert [priority for _,q,priority in actual]==[int(ridge[q]) for q in expected]


def test_seed_preparation_stop_is_bounded_and_abandoned_spools_are_charged(tmp_path):
    from winding_field.bounded_seeds import seeds,SeedPreparationPaused
    class Fields:
        shape=(8,8,8)
        calls=0
        def centrality(self,lo,hi,family):
            self.calls+=1
            return np.zeros(tuple(hi-lo),'u2'),np.zeros(tuple(hi-lo),'u1')
    fields=Fields()
    with pytest.raises(SeedPreparationPaused):
        list(seeds(fields,1,tmp_path,quota_bytes=1<<20,region_size=4,chunk=2,stop=lambda:True))
    assert fields.calls==0
    with pytest.raises(SeedPreparationPaused):
        list(seeds(fields,1,tmp_path,quota_bytes=1<<20,region_size=4,chunk=2,stop=lambda:fields.calls==1))
    assert fields.calls==1 and list(tmp_path.iterdir())==[]
    abandoned=tmp_path/'qual-seeds-abandoned';abandoned.mkdir()
    (abandoned/'seeds.sqlite').write_bytes(b'x'*(1<<20))
    with pytest.raises(OSError,match='quota'):
        list(seeds(fields,1,tmp_path,quota_bytes=1<<20,region_size=4,chunk=2))
    assert (abandoned/'seeds.sqlite').stat().st_size==1<<20
