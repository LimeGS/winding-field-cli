import numpy as np
import pytest


def test_checkpoint_survives_pause_and_rejects_changed_committed_payload(tmp_path):
    from winding_field.native_store import Journal,Paused
    stopped=False
    with Journal(tmp_path/'work',{'source':'fixed'},stop=lambda:stopped) as store:
        store.save_arrays('charts/one.npz',xyz=np.array([[1.,2.,3.]]))
        before=(store.root/'charts/one.npz').read_bytes()
        stopped=True
        with pytest.raises(Paused):
            store.check()
    with Journal(tmp_path/'work',{'source':'fixed'}) as store:
        np.testing.assert_array_equal(store.load_arrays('charts/one.npz')['xyz'],[[1,2,3]])
        assert (store.root/'charts/one.npz').read_bytes()==before
    (tmp_path/'work/charts/one.npz').write_bytes(b'broken')
    with pytest.raises(ValueError,match='changed|corrupt'):
        Journal(tmp_path/'work',{'source':'fixed'})


def test_journal_rejects_identity_change_and_unsafe_payload_path(tmp_path):
    from winding_field.native_store import Journal
    with Journal(tmp_path/'work',{'source':'fixed'}) as store:
        with pytest.raises(ValueError):
            store.save_arrays('../escape.npz',a=np.zeros(1))
    with pytest.raises(ValueError,match='identity'):
        Journal(tmp_path/'work',{'source':'other'})


def test_uncommitted_payload_can_be_recovered_but_committed_payload_is_immutable(tmp_path):
    from winding_field.native_store import Journal
    with Journal(tmp_path/'work',{'source':'fixed'}) as store:
        (store.root/'orphan.npz').write_bytes(b'interrupted write')
        store.save_arrays('orphan.npz',a=np.array([2]))
        with pytest.raises(ValueError,match='immutable|changed'):
            store.save_arrays('orphan.npz',a=np.array([3]))
    with Journal(tmp_path/'work',{'source':'fixed'}) as store:
        assert store.load_arrays('orphan.npz')['a'].tolist()==[2]


def test_scratch_admission_preserves_committed_units(tmp_path,monkeypatch):
    import winding_field.native_store as module
    from winding_field.native_store import Journal,Paused
    with Journal(tmp_path/'work',{'source':'fixed'}) as store:
        store.save_arrays('charts/first.npz',a=np.array([1.,2.]))
        before=(store.root/'charts/first.npz').read_bytes()
        monkeypatch.setattr(module,'SCRATCH_LIMIT',200_000,raising=False)
        with pytest.raises(Paused,match='SCRATCH_LIMIT'):
            store.save_arrays('charts/large.npz',a=np.arange(200_000,dtype=float))
        assert (store.root/'charts/first.npz').read_bytes()==before
        assert not (store.root/'charts/large.npz').exists()
