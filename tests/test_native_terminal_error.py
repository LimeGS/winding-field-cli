"""Only the proven binary's irreversible PaStiX exit may bypass its timeout."""
import numpy as np
import pytest


@pytest.mark.parametrize('pinned,message,early', [
    (True, 'Negative diagonal term', True),
    (False, 'Negative diagonal term', False),
    (True, 'warning: Negative diagonal term', False),
])
def test_terminal_error_is_binary_bound_and_exact(tmp_path, monkeypatch, pinned, message, early):
    from winding_field import surface_export as s
    app=tmp_path/'AppRun'
    app.write_text(f'#!/bin/sh\nprintf "%s\\n" "{message}" >&2\nsleep 60\n')
    app.chmod(0o700)
    binary=tmp_path/'usr/bin/flatboi';binary.parent.mkdir(parents=True);binary.write_bytes(b'fixture')
    original=s._sha
    monkeypatch.setattr(s, '_sha', lambda p: '2f89bfc96512bebf7ef44b25de01c5afd01c7af058e452515b03bc52cf6afd6a'
        if pinned and p==binary else original(p))
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    uv,receipt=s.flatten_native(v,np.array([[0,1,2]]),app,tmp_path/'native',timeout=.5)
    assert uv is None
    assert receipt['timed_out'] is (not early)
    assert bool(receipt.get('terminal_native_error')) is early
    assert (tmp_path/'native/stderr.log').read_text().strip()==message


def test_cache_budget_can_increase_without_changing_saved_uv(tmp_path):
    from test_patch_parallel import cache_fixture
    from winding_field.native_patches import NativeReuse
    from winding_field.util import file_sha
    root,identity,v,f,uv,patch=cache_fixture(tmp_path)
    identity['settings']['max_charts']=8192
    reuse=NativeReuse(root,file_sha(root/'IDENTITY.json'),identity)
    saved,_=reuse.get(0,v,f,tmp_path/'larger-budget')
    np.testing.assert_array_equal(saved,uv)
    identity['settings']['max_charts']=2048
    with pytest.raises(ValueError,match='budget'):
        NativeReuse(root,file_sha(root/'IDENTITY.json'),identity)
