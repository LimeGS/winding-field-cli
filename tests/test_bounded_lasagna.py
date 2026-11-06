import io
import numpy as np
import pytest


def test_worker_does_not_shadow_installed_lasagna(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    from winding_field import lasagna_worker
    package=tmp_path/'lasagna';package.mkdir()
    (package/'__init__.py').write_text("raise RuntimeError('installed Lasagna reached')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(Path(lasagna_worker.__file__).parent))
    monkeypatch.delitem(sys.modules,'lasagna',raising=False)
    with pytest.raises(RuntimeError,match='installed Lasagna reached'):
        lasagna_worker.bounded_main('unused','cpu')


def test_worker_frames_preserve_channels_and_reject_truncated_or_oversized_payload():
    from winding_field.bounded_lasagna import read_frame,write_frame
    raw=np.arange(8*4*4*4,dtype='f4').reshape(8,4,4,4)
    stream=io.BytesIO();write_frame(stream,raw)
    wire=stream.getvalue();np.testing.assert_array_equal(read_frame(io.BytesIO(wire)),raw)
    with pytest.raises(EOFError):read_frame(io.BytesIO(wire[:-3]))
    with pytest.raises(ValueError,match='bounded'):read_frame(io.BytesIO((1<<40).to_bytes(8,'little')))
    with pytest.raises(ValueError):write_frame(io.BytesIO(),np.array([object()],object))
    class ShortWriter(io.BytesIO):
        def write(self,value):return super().write(value[:7])
    stream=ShortWriter();write_frame(stream,raw)
    np.testing.assert_array_equal(read_frame(io.BytesIO(stream.getvalue())),raw)


def test_finalization_dispatch_preserves_raw_and_encoded_channel_order():
    from winding_field.bounded_lasagna import finalize
    class Product:
        def __init__(self,name):self.name=name
    class Adapter:
        cos_product=Product('cos');normal_product=Product('normal')
        def finalize_product_slab(self,product,slab):
            if product.name=='cos':return {'cos':(slab[0]*255).astype('u1')}
            assert slab.shape[0]==7
            return {k:(slab[i]*255).astype('u1') for i,k in enumerate(['grad_mag','nx','ny'])}
    raw=np.full((8,2,2,2),.5,'f4')
    result=finalize(Adapter(),raw)
    assert result.shape==(4,2,2,2) and result.dtype==np.uint8
    assert (result==127).all()


def test_stalled_worker_request_times_out_and_reaps_child(tmp_path,monkeypatch):
    import subprocess
    import sys
    import time
    from winding_field.bounded_lasagna import LasagnaModel
    original=subprocess.Popen
    def stalled(command,**kwargs):
        return original([sys.executable,'-c','import time; time.sleep(1)'],**kwargs)
    monkeypatch.setattr(subprocess,'Popen',stalled)
    start=time.monotonic()
    with LasagnaModel(sys.executable,'unused','cpu',tmp_path/'worker.log',timeout=.1) as model:
        with pytest.raises(TimeoutError):model.predict_raw(np.zeros((64,)*3,'u1'))
        assert model.child.poll() is not None
    assert time.monotonic()-start<3
