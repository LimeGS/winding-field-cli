import ctypes
from fractions import Fraction as Q
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_tifxyz_native import artifact, coverage, orientation


@pytest.mark.parametrize('kind',['numpy','cpu'])
def test_filters_preserve_exact_uncertainty_and_rational_centers(kind,artifact,monkeypatch):
    from winding_field.tifxyz_predicates import Backend
    monkeypatch.setenv('WINDING_TIFXYZ_ACCELERATOR',str(artifact/'accelerator.json'))
    cases=[coverage(),coverage(point=(2.,0.,0.)),coverage(point=(.5,0.,0.)),
           coverage(center=(Q(1,3),0,0),half2=Q(1,9),tau=2/3),coverage(unit=1e100)]
    oriented=[orientation(),orientation(corners=(1.,)*4+(0.,)*4),orientation(corners=(0.,)*8)]
    with Backend(kind) as backend:
        for mode,items in ((0,cases),(1,oriented)):
            result=backend.classify(mode,np.array([r for r,_ in items]))
            assert all(v==-1 or v==expected for v,(_,expected) in zip(result,items))
        assert backend.classify(0,np.array([cases[2][0],cases[4][0]])).tolist()==[-1,-1]


def test_portable_backend_does_not_load_native_environment(monkeypatch):
    from winding_field.tifxyz_predicates import Backend,identity
    monkeypatch.setenv('WINDING_TIFXYZ_ACCELERATOR','/absent/foreign.json')
    monkeypatch.setattr(ctypes,'CDLL',lambda *a:pytest.fail('portable backend loaded native code'))
    assert identity('numpy')=={'backend':'numpy'}
    with Backend('numpy') as backend:
        assert backend.classify(0,np.array([coverage()[0]])).tolist()==[1]


def test_missing_explicit_accelerator_is_blocked(monkeypatch):
    from winding_field.tifxyz_predicates import Backend
    from winding_field.tifxyz_contract import Blocked
    monkeypatch.delenv('WINDING_TIFXYZ_ACCELERATOR',raising=False)
    with pytest.raises(Blocked,match='ACCELERATOR_UNAVAILABLE'): Backend('cpu')


def test_cuda_device_permission_precedes_library_loading(artifact,monkeypatch):
    from winding_field import tifxyz_predicates as code
    from winding_field.tifxyz_contract import Blocked
    monkeypatch.delenv('WINDING_TIFXYZ_CUDA_DEVICE',raising=False)
    monkeypatch.setattr(code,'_native',lambda kind:(artifact/'accelerator.json',artifact/'predicates.so',{'backend':'cuda'},{}))
    monkeypatch.setattr(ctypes,'CDLL',lambda *args:pytest.fail('CUDA library loaded before device opt-in'))
    with pytest.raises(Blocked,match='ACCELERATOR_UNAVAILABLE'):code.Backend('cuda')


@pytest.mark.parametrize('damage',['hash','source','abi','platform','path','size','extra','duplicate'])
def test_manifest_is_validated_before_native_load(tmp_path,artifact,monkeypatch,damage):
    from winding_field.tifxyz_predicates import identity
    import shutil
    shutil.copyfile(artifact/'predicates.so',tmp_path/'predicates.so')
    data=json.loads((artifact/'accelerator.json').read_text())
    if damage=='hash':data['library_sha256']='0'*64
    elif damage=='source':data['sources']['predicates.h']='0'*64
    elif damage=='abi':data['abi']=2
    elif damage=='platform':data['platform']['machine']='foreign'
    elif damage=='path':data['library']='../predicates.so'
    elif damage=='size':data['library_bytes']=1<<40
    elif damage=='extra':data['load_from_export']='/foreign.so'
    text=json.dumps(data)
    if damage=='duplicate':text=text[:-1]+',"abi":1}'
    path=tmp_path/'accelerator.json';path.write_text(text)
    monkeypatch.setenv('WINDING_TIFXYZ_ACCELERATOR',str(path))
    monkeypatch.setattr(ctypes,'CDLL',lambda *a:pytest.fail('invalid manifest executed code'))
    with pytest.raises(ValueError): identity('cpu')


def test_backend_checks_artifact_mutation_between_batches(tmp_path,artifact,monkeypatch):
    from winding_field.tifxyz_predicates import Backend
    import shutil
    shutil.copytree(artifact,tmp_path/'copy')
    path=tmp_path/'copy/accelerator.json'
    monkeypatch.setenv('WINDING_TIFXYZ_ACCELERATOR',str(path))
    with pytest.raises(ValueError,match='changed'):
        with Backend('cpu') as backend:
            path.write_text(path.read_text()+' ')
            backend.classify(0,np.array([coverage()[0]]))


@pytest.mark.parametrize('rows',[np.zeros((0,24)),np.zeros((65537,24)),np.zeros((1,23)),np.zeros((1,24),dtype='float32')])
def test_batch_envelope_precedes_allocation(rows):
    from winding_field.tifxyz_predicates import Backend
    with Backend('numpy') as backend,pytest.raises(ValueError):backend.classify(0,rows)


def test_native_identity_is_content_bound_not_path_bound(tmp_path,artifact,monkeypatch):
    from winding_field.tifxyz_predicates import identity
    import shutil
    monkeypatch.setenv('WINDING_TIFXYZ_ACCELERATOR',str(artifact/'accelerator.json'))
    first=identity('cpu')
    shutil.copytree(artifact,tmp_path/'relocated')
    monkeypatch.setenv('WINDING_TIFXYZ_ACCELERATOR',str(tmp_path/'relocated/accelerator.json'))
    assert identity('cpu')==first
    assert first['library_sha256']==json.loads((artifact/'accelerator.json').read_text())['library_sha256']


def test_accelerator_module_import_stays_lightweight():
    run=subprocess.run([sys.executable,'-c',"import sys; import winding_field.tifxyz_predicates; assert 'numpy' not in sys.modules and 'torch' not in sys.modules"],capture_output=True,text=True)
    assert run.returncode==0,run.stderr


@pytest.mark.parametrize('kind',['numpy','cpu','cuda'])
def test_audit_backend_is_explicit_normalized_and_effective(kind):
    from winding_field.tifxyz_contract import validate_request,load_settings
    r=validate_request(dict(levels=[0.],audit_backend=kind))
    assert r['audit_backend']==kind
    frame=dict(id='fixture',origin_xyz_um=[0,0,0],voxel_um=1.,shape_zyx=[3]*3,source_identity_sha256='a'*64)
    settings=load_settings(r,dict(shape_zyx=[3]*3,spacing_xyz=[1.]*3,voxel_um=1.),frame)
    assert settings.audit_backend==kind


def test_no_library_or_device_path_accepted_from_export_request():
    from winding_field.tifxyz_contract import validate_request
    assert validate_request(dict(levels=[0.]))['audit_backend']=='numpy'
    for entry in ({'audit_backend':'unknown'},{'audit_backend':True},
                  {'audit_library':'/untrusted.so'},{'cuda_device':'GPU-other'}):
        with pytest.raises(ValueError):validate_request(dict(levels=[0.],**entry))
