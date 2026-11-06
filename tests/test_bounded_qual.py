"""Global Qual geometry must not end at a backing-cache boundary."""

import numpy as np
import pytest


def test_paged_field_eviction_preserves_global_scalar_addresses():
    from winding_field.bounded_qual import PagedField
    source=np.arange(9*7*5,dtype=np.uint16).reshape(9,7,5)
    calls=[]
    def load(lo,hi):
        calls.append(tuple(lo))
        return source[tuple(slice(a,b) for a,b in zip(lo,hi))].copy()
    field=PagedField(source.shape,np.uint16,load,chunk=4,cache_bytes=128)
    for ij in [(0,0,0),(8,6,4),(4,3,2),(0,0,0)]:
        assert field[ij]==source[ij]
        assert field.resident_bytes<=128
    assert calls.count((0,0,0))==2
    with pytest.raises(IndexError):field[9,0,0]


def test_trilinear_qual_read_crosses_cache_boundaries_without_changing_values():
    from winding_field.bounded_qual import PagedField,python_kernels
    from avf_spline_generator import core
    source=np.arange(12*12*12,dtype=np.uint16).reshape(12,12,12)%251
    field=PagedField(source.shape,np.uint16,
        lambda lo,hi:source[tuple(slice(a,b) for a,b in zip(lo,hi))].copy(),chunk=4,cache_bytes=128)
    kernels=python_kernels()
    for xyz in [(3.5,3.5,3.5),(7.9,4.2,8.1),(10.8,10.8,10.8)]:
        actual=kernels['trilin_scalar_u16'](field,*xyz)
        assert actual==pytest.approx(core.trilin_scalar_u16(source,*xyz),abs=1e-6)


def test_invalid_or_oversized_field_tile_is_not_cached():
    from winding_field.bounded_qual import PagedField
    with pytest.raises(ValueError,match='cache'):
        PagedField((8,8,8),np.uint8,lambda a,b:None,chunk=4,cache_bytes=63)
    field=PagedField((8,8,8),np.uint8,lambda a,b:np.zeros((2,2,2),np.uint8),chunk=4,cache_bytes=64)
    with pytest.raises(ValueError,match='shape'):field[0,0,0]


def test_global_fibre_history_survives_many_cache_boundaries():
    from winding_field.bounded_qual import PagedField,python_kernels
    from avf_spline_generator import core
    shape=(16,16,64)
    arrays=[np.full(shape,value,dtype) for value,dtype in
            [(10000,np.uint16),(255,np.uint8),(255,np.uint8),(128,np.uint8),
             (0,np.uint8),(128,np.uint8),(128,np.uint8)]]
    fields=[PagedField(shape,a.dtype,lambda lo,hi,a=a:a[tuple(slice(x,y) for x,y in zip(lo,hi))],
                       chunk=4,cache_bytes=128) for a in arrays]
    expected=np.empty((120,3),np.float32); expected_tangent=np.empty_like(expected)
    actual=np.empty_like(expected); actual_tangent=np.empty_like(expected)
    n=core.trace_half_buf(*arrays,2.,8.,8.,1.,0.,0.,expected,expected_tangent)
    m=python_kernels()['trace_half_buf'](*fields,2.,8.,8.,1.,0.,0.,actual,actual_tangent)
    assert n==m and n>60
    np.testing.assert_allclose(actual[:m],expected[:n],atol=1e-5,rtol=0)
    np.testing.assert_allclose(actual_tangent[:m],expected_tangent[:n],atol=1e-5,rtol=0)


def test_lazy_qual_pooling_and_stencils_match_dense_reference():
    from winding_field.bounded_qual import QualFields
    from avf_spline_generator.prepare import tensor_axes
    from avf_spline_generator import core
    import itertools
    rng=np.random.default_rng(12)
    native=rng.uniform(0,.95,(2,40,40,40)).astype('f4')
    read=lambda lo,hi:native[(slice(None),)+tuple(slice(a,b) for a,b in zip(lo,hi))]
    fields=QualFields((40,40,40),read,chunk=8,cache_bytes=4096)
    pooled=np.zeros((2,20,20,20),'f4')
    for z,y,x in itertools.product((0,1),repeat=3):pooled+=native[:,z::2,y::2,x::2]
    pooled=np.rint(pooled/8*255).clip(0,255).astype('u1')
    np.testing.assert_array_equal(fields.pooled(np.array([0,0,0]),np.array([20,20,20])),pooled)
    axes=tensor_axes(pooled,(slice(None),)*3)
    pres=np.minimum(pooled[0].astype('u2')+pooled[1],255).astype('u1')
    pres[pooled[0]<pooled[1]]=0
    cent,seeds=core.stage1_centrality_and_ridge(pres,*axes,2)
    family=fields.family(1)
    for ij in [(0,0,0),(7,7,7),(8,8,8),(16,9,10),(19,19,19)]:
        assert family['presence'][ij]==pres[ij]
        assert family['nx'][ij]==axes[0][ij]
        assert family['ny'][ij]==axes[1][ij]
        assert family['centrality'][ij]==cent[ij]
        assert family['seeds'][ij]==seeds[ij]


def test_raw_curve_replay_preserves_cross_window_coverage_and_family_seed_skip():
    from winding_field.bounded_qual import PagedField,CoverageReplay,trace_one
    from avf_spline_generator import core
    shape=(16,16,64)
    arrays={name:np.full(shape,value,dtype) for name,value,dtype in
            [('centrality',10000,np.uint16),('presence',255,np.uint8),
             ('nx',255,np.uint8),('ny',128,np.uint8)]}
    fields={name:PagedField(shape,a.dtype,
        lambda lo,hi,a=a:a[tuple(slice(x,y) for x,y in zip(lo,hi))],chunk=4,cache_bytes=128)
        for name,a in arrays.items()}
    raw=[]
    replay=CoverageReplay(shape,fields,lambda lo,hi:iter(raw),chunk=4,cache_bytes=128)
    curves=trace_one((8,8,10),2,fields,replay.fields())
    assert len(curves)==1 and len(curves[0][0])>60
    raw.extend(curves)
    actual=replay.fields()
    expected=[np.zeros(shape,'u1') for _ in range(3)]
    for p,t in raw:
        core.mark_curve_coverage(*expected,arrays['centrality'],arrays['presence'],arrays['nx'],arrays['ny'],p,t,len(p),2.05)
    for ij in [(8,8,10),(8,8,32),(8,8,55),(1,1,1)]:
        assert [a[ij] for a in actual]==[a[ij] for a in expected]
    assert trace_one((8,8,50),2,fields,actual)==[]


def test_finished_curves_keep_native_family_validation_across_paging(monkeypatch):
    from winding_field.bounded_qual import PagedField,finish_curves
    from winding_field import bounded_qual
    monkeypatch.setattr(bounded_qual,'python_kernels',lambda:pytest.fail('fitting must use compiled kernels'))
    from avf_spline_generator.tracing import finish,pack
    shape=(16,16,32)
    vertical=np.full(shape,255,'u1');horizontal=np.zeros(shape,'u1')
    fields=[PagedField(shape,a.dtype,lambda lo,hi,a=a:a[tuple(slice(x,y) for x,y in zip(lo,hi))],
                       chunk=4,cache_bytes=64) for a in [vertical,horizontal]]
    class Native:
        def sample(self,xyz):
            good=xyz[:,0]<29
            return np.column_stack([np.where(good,255,0),np.where(good,0,255)]).astype('f4')
    points=np.column_stack([np.arange(2.,28.,.75),np.full(35,8.),np.full(35,8.)]).astype('f4')
    raw,offsets=pack([points])
    expected=finish(raw,offsets,[vertical,horizontal],vertical,1,Native())
    actual=finish_curves([points],fields,fields[0],1,Native())
    np.testing.assert_array_equal(actual[1],expected[1])
    np.testing.assert_allclose(actual[0],expected[0],rtol=0,atol=1e-5)


def test_empty_seed_commits_reuse_coverage_without_replaying_history(tmp_path):
    from winding_field.bounded_qual import trace_seeds
    from winding_field.bounded_journal import CurveJournal
    shape=(16,16,64)
    fields={name:np.full(shape,value,dtype) for name,value,dtype in
            [('centrality',10000,np.uint16),('presence',255,np.uint8),
             ('nx',255,np.uint8),('ny',128,np.uint8)]}
    queries=[];before=[]
    def seeds():
        for i,x in enumerate((10,50,50,50)):
            before.append(len(queries))
            yield i,(8,8,x),2
    with CurveJournal(tmp_path/'raw.sqlite',{},quota_bytes=1<<20) as journal:
        journal.db.set_trace_callback(lambda sql:queries.append(sql) if 'JOIN curves c ON' in sql else None)
        assert trace_seeds(seeds(),1,fields,journal,chunk=16,cache_bytes=65536)=='COMPLETE'
        assert journal.next_seed(1)==4
        assert journal.db.execute('SELECT count(*) FROM curves').fetchone()[0]==1
        assert before[2]>before[1]  # The accepted first curve invalidated cached coverage.
        assert before[3]==before[2]  # The following empty commit did not change coverage.
    with CurveJournal(tmp_path/'raw.sqlite',{},quota_bytes=1<<20) as journal:
        assert trace_seeds([(4,(8,8,50),2)],1,fields,journal,chunk=16,cache_bytes=65536)=='COMPLETE'
        assert journal.next_seed(1)==5
        assert journal.db.execute('SELECT count(*) FROM curves').fetchone()[0]==1
