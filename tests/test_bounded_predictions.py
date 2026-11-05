import numpy as np
import pytest


class Volume:
    def __init__(self,shape):self.shape=shape;self.calls=[]
    def sample(self,axes):
        self.calls.append(tuple(a.copy() for a in axes))
        return (axes[0][:,None,None]+axes[1][None,:,None]+axes[2][None,None,:]).astype('u1')


def test_qual_windows_are_global_and_eviction_does_not_shift_normalization():
    from winding_field.bounded_predictions import QualWindows
    volume=Volume((200,200,200)); windows=[]
    class Model:
        def predict(self,ct):
            windows.append(ct.copy())
            values=np.stack([ct.astype('f4')/255,1-ct.astype('f4')/255],axis=-1)
            return None,None,values
    reader=QualWindows(volume,Model(),cache_bytes=48**3*8)
    result=reader.read(np.array([47,47,47]),np.array([50,50,50]))
    axes=np.arange(47,50)
    expected=(axes[:,None,None]+axes[None,:,None]+axes[None,None,:]).astype('u1')/255
    np.testing.assert_allclose(result[0],expected,atol=1e-7)
    assert len(windows)==8 and all(w.shape==(128,)*3 for w in windows)
    assert reader.resident_bytes<=48**3*8
    first=volume.calls[0]
    np.testing.assert_array_equal(first[0],np.clip(np.arange(-40,88),0,199))


def test_lasagna_last_shifted_tile_blends_raw_channels_before_encoding():
    from winding_field.bounded_predictions import LasagnaWindows
    volume=Volume((65,65,65))
    class Model:
        def predict_raw(self,ct):
            # Different canonical windows produce different normalized values.
            return np.full((8,64,64,64),float(ct[16,16,16])/255,np.float32)
    reader=LasagnaWindows(volume,Model(),cache_bytes=32**3*8*4)
    value=reader.read_raw(np.array([33,33,33]),np.array([34,34,34]))
    # Position33 belongs to cores beginning32 AND33 on every axis.
    assert value.shape==(8,1,1,1)
    np.testing.assert_allclose(value,97.5/255,atol=1e-7)
    assert all(len(a)==64 for call in volume.calls for a in call)
    assert all((a>=0).all() and (a<65).all() for call in volume.calls for a in call)


def test_prediction_reads_reject_native_volume_materialization():
    from winding_field.bounded_predictions import QualWindows
    reader=QualWindows(Volume((10000,10000,10000)),None,cache_bytes=48**3*8)
    with pytest.raises(ValueError,match='bounded'):
        reader.read(np.array([0,0,0]),np.array([10000,10000,10000]))


def test_qual_complete_halos_and_context_sensitive_predictions_survive_eviction():
    from winding_field.bounded_predictions import QualWindows
    shape=(101,99,103)
    z,y,x=np.indices(shape);source=((z*z+3*y+7*x)%256).astype('u1')
    padded=np.pad(source,((40,128),)*3,mode='edge')
    class Source:
        shape=source.shape
        def sample(self,axes):return source[np.ix_(*axes)]
    class Model:
        def predict(self,ct):
            probability=np.float32(ct.mean()/255)
            return None,None,np.broadcast_to([probability,1-probability],(*ct.shape,2)).astype('f4')
    reader=QualWindows(Source(),Model(),cache_bytes=48**3*8)
    for lo,hi in [([47]*3,[50]*3),([99,97,101],list(shape)),([47]*3,[50]*3)]:
        result=reader.read(np.array(lo),np.array(hi))
        for ij in np.ndindex(tuple(np.array(hi)-lo)):
            start=((np.asarray(lo)+ij)//48)*48
            expected=padded[tuple(slice(s,s+128) for s in start)].mean()/255
            assert result[(0,*ij)]==pytest.approx(expected,abs=1e-7)


def test_lasagna_complete_reflected_windows_match_dense_padding_at_last_tiles():
    from winding_field.bounded_predictions import LasagnaWindows
    shape=(65,67,69);z,y,x=np.indices(shape);source=((z*z+3*y+7*x)%256).astype('u1')
    padded=np.pad(source,((16,16),)*3,mode='reflect')
    class Source:
        shape=source.shape
        def sample(self,axes):return source[np.ix_(*axes)]
    windows=[]
    class Model:
        def predict_raw(self,ct):
            windows.append(ct.copy());return np.full((8,64,64,64),ct.mean()/255,'f4')
    reader=LasagnaWindows(Source(),Model(),cache_bytes=32**3*8*4)
    for key in [(0,0,0),(32,32,32),(33,35,37),(0,0,0)]:
        actual=reader.window(key)
        expected=padded[tuple(slice(s,s+64) for s in key)]
        np.testing.assert_array_equal(windows[-1],expected)
        np.testing.assert_allclose(actual,expected.mean()/255,atol=1e-7)
