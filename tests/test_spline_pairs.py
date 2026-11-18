import numpy as np
import pytest

from winding_field import field


def scalar_pairs(points,low,high,step):
    """Frozen pre-vectorization semantics, including scalar norm/linspace."""
    start=points[:-1];delta=np.diff(points,axis=0)
    near=np.divide(low-start,delta,out=np.full_like(start,-np.inf),where=delta!=0)
    far=np.divide(high-start,delta,out=np.full_like(start,np.inf),where=delta!=0)
    enter=np.maximum(0,np.minimum(near,far).max(1))
    leave=np.minimum(1,np.maximum(near,far).min(1))
    valid=(leave>enter)&np.all((delta!=0)|((start>=low)&(start<=high)),axis=1)
    a=[];b=[]
    for i in np.flatnonzero(valid):
        first=np.clip(start[i]+enter[i]*delta[i],low,high)
        last=np.clip(start[i]+leave[i]*delta[i],low,high)
        length=np.linalg.norm(last-first)
        if length:
            samples=np.linspace(first,last,max(1,int(np.ceil(length/step)))+1)
            a.extend(samples[:-1]);b.extend(samples[1:])
    return np.array(a).reshape(-1,3),np.array(b).reshape(-1,3)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('step',[.125,1.,172.8])
@pytest.mark.parametrize('seed',range(5))
def test_spline_pairs_matches_frozen_scalar_endpoints_and_order(dtype,step,seed):
    points=np.random.default_rng(seed).uniform(-1,5,(60,3)).astype(dtype)
    points[5]=points[4]
    low=np.array([.1,-.2,.3]);high=np.array([2.,3.,4.])
    expected=scalar_pairs(points,low,high,step)
    actual=field.spline_pairs(points,low,high,step)
    for got,want in zip(actual,expected):
        np.testing.assert_array_equal(got,want)
        if len(want):
            assert got.dtype==want.dtype
            assert got.tobytes()==want.tobytes()


@pytest.mark.parametrize('points',[
    [],[[0.,0.,0.]],[[0.,0.,0.],[0.,0.,0.]],
    [[-0.,0.,0.],[.1,-0.,0.],[1.,0.,0.],[np.nextafter(2.,3.),0.,0.]],
    [[0.,0.,0.],[1.,0.,0.],[np.nextafter(2.,3.),0.,0.],[2.,3.,4.]],
    [[0.,0.,0.],[1e-300,0.,0.],[2e-300,0.,0.]],
])
def test_spline_pairs_degenerate_boundary_and_underflow_cases(points):
    points=np.array(points,dtype=float).reshape(-1,3)
    expected=scalar_pairs(points,np.zeros(3),np.array([2.,3.,4.]),1.)
    actual=field.spline_pairs(points,np.zeros(3),np.array([2.,3.,4.]),1.)
    for got,want in zip(actual,expected):
        np.testing.assert_array_equal(got,want)
        assert got.tobytes()==want.tobytes()


def test_short_segments_use_one_batched_linspace(monkeypatch):
    original=np.linspace
    points=np.column_stack([original(.1,.4,1000),np.zeros(1000),np.zeros(1000)])
    expected=scalar_pairs(points,np.zeros(3),np.ones(3),1.)
    calls=[]
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(np,'linspace',counted)
    result=field.spline_pairs(points,np.zeros(3),np.ones(3),1.)
    assert len(calls)==1
    for got,want in zip(result,expected):np.testing.assert_array_equal(got,want)
