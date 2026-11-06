"""Compiled access must retain global coordinates and a single bounded cache."""
import gc
import weakref
import numpy as np
import pytest
from numba import njit


@njit
def read_values(field,indices):
    out=np.empty(len(indices),np.uint16)
    for i in range(len(indices)):
        out[i]=field[indices[i,0],indices[i,1],indices[i,2]]
    return out


@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
def test_compiled_tiles_evict_and_share_python_cache(dtype):
    from winding_field.bounded_qual import PagedField
    from winding_field.bounded_numba import as_numba
    a=(np.arange(9*7*5).reshape(9,7,5)%251).astype(dtype);loads=[]
    def load(lo,hi):
        loads.append(tuple(lo))
        return a[tuple(slice(x,y) for x,y in zip(lo,hi))]
    field=PagedField(a.shape,a.dtype,load,chunk=4,cache_bytes=64*a.itemsize)
    assert field[0,0,0]==a[0,0,0]
    compiled=as_numba(field)
    assert as_numba(field) is compiled
    assert read_values(compiled,np.array([[0,0,0]]))[0]==a[0,0,0]
    assert len(loads)==1
    indices=np.array([[8,6,4],[4,3,2],[0,0,0]])
    np.testing.assert_array_equal(read_values(compiled,indices),a[tuple(indices.T)])
    assert field[0,0,0]==a[0,0,0]
    assert loads.count((0,0,0))==2
    assert field.resident_bytes==compiled.resident_bytes<=field.limit
    with pytest.raises(IndexError):read_values(compiled,np.array([[9,0,0]]))
    ref=weakref.ref(field);del field;gc.collect()
    assert ref() is None
    with pytest.raises(RuntimeError,match='released'):read_values(compiled,np.array([[8,0,0]]))


def test_compiled_tile_loader_rejects_invalid_payload():
    from winding_field.bounded_qual import PagedField
    from winding_field.bounded_numba import as_numba
    field=PagedField((4,4,4),np.uint8,lambda lo,hi:np.zeros((4,4,4),'f4'),chunk=4,cache_bytes=64)
    with pytest.raises(ValueError,match='shape or dtype'):
        read_values(as_numba(field),np.array([[0,0,0]]))


@pytest.mark.parametrize('gap',[False,True])
def test_compiled_global_trace_matches_original_dense_across_tiles(gap):
    from avf_spline_generator import core
    from winding_field.bounded_qual import PagedField,trace_one
    from winding_field.bounded_numba import trace_kernel
    shape=(24,24,96)
    rng=np.random.default_rng(71)
    arrays=dict(centrality=rng.integers(9950,10050,shape,dtype=np.uint16),
                presence=np.full(shape,255,'u1'),nx=np.full(shape,255,'u1'),
                ny=np.broadcast_to((128+3*np.sin(np.arange(96)*.06)).astype('u1'),shape).copy())
    if gap:arrays['presence'][:,:,55:]=0
    fields={name:PagedField(shape,a.dtype,lambda lo,hi,a=a:a[tuple(slice(x,y) for x,y in zip(lo,hi))],
                           chunk=8,cache_bytes=2048) for name,a in arrays.items()}
    coverage=[np.zeros(shape,'u1') for _ in range(3)]
    actual=trace_one((12,12,10),2,fields,coverage)
    points=np.empty((131075,3),'f4');tangents=np.empty_like(points);offsets=np.zeros(65538,'i8')
    _,count=core.trace_seed_range(*(arrays[n] for n in ('centrality','presence','nx','ny')),
        *(np.array([v],'i4') for v in (12,12,10)),*coverage,points,tangents,offsets,0,0,0,1,
        min_raw_len=7.5,step_size=.75,max_low_steps=3,core_r=2.05,n_pass1=1)
    assert count==len(actual)>0
    for i,(p,t) in enumerate(actual):
        np.testing.assert_array_equal(p,points[offsets[i]:offsets[i+1]])
        np.testing.assert_array_equal(t,tangents[offsets[i]:offsets[i+1]])
    assert trace_kernel().nopython_signatures
    assert all(f.resident_bytes<=f.limit for f in fields.values())


def test_coverage_replay_uses_compiled_original_with_global_window(monkeypatch):
    from avf_spline_generator import core
    from winding_field import bounded_qual as q
    monkeypatch.setattr(q,'python_kernels',lambda:pytest.fail('coverage must use compiled kernels'))
    shape=(12,12,32)
    arrays={name:np.full(shape,value,dtype) for name,value,dtype in
            [('centrality',10000,'u2'),('presence',255,'u1'),('nx',255,'u1'),('ny',128,'u1')]}
    fields={name:q.PagedField(shape,a.dtype,lambda lo,hi,a=a:a[tuple(slice(x,y) for x,y in zip(lo,hi))],
                             chunk=4,cache_bytes=256) for name,a in arrays.items()}
    points=np.column_stack((np.arange(1,31,.75),np.full(40,6),np.full(40,6))).astype('f4')
    tangents=np.zeros_like(points);tangents[:,0]=1
    replay=q.CoverageReplay(shape,fields,lambda lo,hi:iter([(points,tangents)]),chunk=4,cache_bytes=256)
    expected=[np.zeros(shape,'u1') for _ in range(3)]
    core.mark_curve_coverage(*expected,*(arrays[n] for n in ('centrality','presence','nx','ny')),
                             points,tangents,len(points),2.05)
    lo=np.array([4,4,7]);hi=np.array([9,9,14])
    for a,b in zip(replay.replay(lo,hi),expected):
        np.testing.assert_array_equal(a,b[4:9,4:9,7:14])


@pytest.mark.parametrize('error',[MemoryError,OSError])
def test_nested_coverage_loader_failure_preserves_seed_transaction(tmp_path,error):
    from winding_field.bounded_qual import PagedField,trace_seeds
    from winding_field.bounded_journal import CurveJournal
    shape=(12,12,32)
    def fail(lo,hi):raise error('injected geometry load failure')
    fields=dict(centrality=PagedField(shape,np.uint16,fail,chunk=4,cache_bytes=128),
        presence=np.full(shape,255,'u1'),nx=np.full(shape,255,'u1'),ny=np.full(shape,128,'u1'))
    points=np.column_stack((np.arange(1,31,.75),np.full(40,6),np.full(40,6))).astype('f4')
    tangents=np.zeros_like(points);tangents[:,0]=1
    path=tmp_path/'journal.sqlite'
    with CurveJournal(path,{},quota_bytes=1<<20) as journal:
        journal.commit(1,0,[(points,tangents)])
        with pytest.raises(error,match='injected geometry'):
            trace_seeds([(1,(6,6,15),2)],1,fields,journal,chunk=4,cache_bytes=128)
        assert journal.next_seed(1)==1
        assert journal.db.execute('SELECT count(*) FROM commits').fetchone()[0]==1
    with CurveJournal(path,{},quota_bytes=1<<20) as journal:
        assert journal.next_seed(1)==1
        stored=list(journal.curves(1,[0]*3,shape))
        assert len(stored)==1
        np.testing.assert_array_equal(stored[0][0],points)


def test_curved_paged_fitting_matches_dense_presence_gates_and_split_pieces():
    from avf_spline_generator import core
    from avf_spline_generator.tracing import finish,pack
    from winding_field.bounded_qual import PagedField,finish_curves
    from winding_field.bounded_numba import as_numba,fit_kernel
    shape=(24,24,48);x=np.linspace(3,42,60,dtype='f4')
    curve=np.column_stack((x,8+2*np.sin(x*.7),10+np.cos(x*.25))).astype('f4')
    curves=[curve,curve+np.array([0,6,0],'f4')]
    points,offsets=pack(curves)
    presence=np.random.default_rng(101).integers(30,256,shape,dtype=np.uint8)
    field=PagedField(shape,np.uint8,lambda lo,hi:presence[tuple(slice(x,y) for x,y in zip(lo,hi))],
                     chunk=4,cache_bytes=128)
    options=dict(min_final_len=0.,target_step=.8,smooth_iters=10,max_dev=.3)
    dense=core.fit_and_resample_splines(points,offsets,presence,**options)
    actual=fit_kernel()(points,offsets,as_numba(field),**options)
    np.testing.assert_array_equal(actual[1],dense[1])
    np.testing.assert_allclose(actual[0],dense[0],rtol=0,atol=1e-5)
    unconstrained=core.fit_and_resample_splines(points,offsets,np.full(shape,255,'u1'),**options)
    assert np.max(np.abs(dense[0]-unconstrained[0]))>1e-3
    class Native:
        def sample(self,xyz):
            x=xyz[:,0]/2-.25
            valid=(x<18)|(x>23)
            return np.column_stack((np.where(valid,255,0),np.where(valid,0,255))).astype('f4')
    tracing=[np.full(shape,255,'u1'),np.zeros(shape,'u1')]
    expected=finish(points,offsets,tracing,presence,1,Native())
    result=finish_curves(curves,tracing,field,1,Native())
    assert len(expected[1])==5
    np.testing.assert_array_equal(result[1],expected[1])
    np.testing.assert_allclose(result[0],expected[0],rtol=0,atol=1e-5)
    assert field.resident_bytes<=field.limit
