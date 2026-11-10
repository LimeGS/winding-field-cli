import hashlib
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def readonly_models(monkeypatch,tmp_path):
    from winding_field import models,lasagna
    monkeypatch.setenv('HOME',str(tmp_path/'home'))
    def forbidden(*args,**kwargs):raise AssertionError('doctor must not download, load weights or infer')
    monkeypatch.setattr(models,'urlopen',forbidden)
    monkeypatch.setattr('torch.load',forbidden)
    monkeypatch.setattr(lasagna,'run_child',forbidden)
    return models,lasagna


def doctor(capsys,*arguments):
    from winding_field.cli import main
    code=main(['doctor',*map(str,arguments)])
    return code,json.loads(capsys.readouterr().out)


def test_bare_doctor_keeps_cli_only_contract(capsys,readonly_models,monkeypatch):
    models,lasagna=readonly_models
    def forbidden(*args,**kwargs):raise AssertionError('bare doctor does not select an inference pipeline')
    monkeypatch.setattr(lasagna,'runtime_identity',forbidden)
    monkeypatch.setattr(models,'checkpoint',forbidden)
    code,result=doctor(capsys)
    assert code==0 and 'version' in result and 'preflight' not in result


def test_doctor_reports_missing_runtime_and_checkpoints_without_creating_cache(capsys,readonly_models,tmp_path):
    code,result=doctor(capsys,'--prediction','qual-v2')
    assert code==1 and result['preflight']['ready'] is False
    checks=result['preflight']['checks']
    assert checks['lasagna_runtime']['status']=='MISSING'
    assert checks['lasagna_checkpoint']['status']==checks['fibre_checkpoint']['status']=='MISSING'
    assert not (tmp_path/'home').exists()


def test_doctor_respects_explicit_venv_symlink_and_reports_incompatibility(capsys,readonly_models,tmp_path):
    python=tmp_path/'venv/bin/python';python.parent.mkdir(parents=True);python.symlink_to(sys.executable)
    code,result=doctor(capsys,'--prediction','qual-v2','--lasagna-python',python)
    check=result['preflight']['checks']['lasagna_runtime']
    assert code==1 and check['status']=='INCOMPATIBLE'
    assert check['executable']==str(python) and 'Python 3.14' in check['error']


@pytest.mark.parametrize('valid', [False,True])
def test_doctor_checks_pinned_weights_without_loading_them(capsys,readonly_models,monkeypatch,tmp_path,valid):
    models,lasagna=readonly_models;python=str(tmp_path/'selected/bin/python')
    def runtime(selected):
        assert selected==python
        return dict(executable=selected,python='3.14.0')
    monkeypatch.setattr(lasagna,'runtime_identity',runtime)
    payload=b'fixture pinned checkpoint'
    expected=hashlib.sha256(payload if valid else b'different checkpoint').hexdigest()
    monkeypatch.setattr(models,'LASAGNA_MODEL_SHA',expected)
    monkeypatch.setattr(models,'QUAL_MODEL_SHA',expected)
    weights=tmp_path/'weights.pt';weights.write_bytes(payload)
    code,result=doctor(capsys,'--prediction','qual-v2','--lasagna-python',python,
                       '--model-path',weights,'--lasagna-model-path',weights)
    checks=result['preflight']['checks'];status='READY' if valid else 'INVALID'
    assert checks['lasagna_runtime']['status']=='READY'
    assert checks['lasagna_checkpoint']['status']==checks['fibre_checkpoint']['status']==status
    assert result['preflight']['ready'] is valid and code==(0 if valid else 1)
    assert weights.read_bytes()==payload


@pytest.mark.parametrize('prediction', ['classical','qual-v2'])
def test_imported_lasagna_does_not_probe_surface_runtime_or_checkpoint(capsys,readonly_models,monkeypatch,tmp_path,prediction):
    models,lasagna=readonly_models
    def forbidden(*args,**kwargs):raise AssertionError('imported surfaces must bypass Lasagna inference')
    monkeypatch.setattr(lasagna,'runtime_identity',forbidden)
    payload=b'qual fixture';weights=tmp_path/'qual.pt';weights.write_bytes(payload)
    monkeypatch.setattr(models,'QUAL_MODEL_SHA',hashlib.sha256(payload).hexdigest())
    checkpoint=models.checkpoint
    def check(path=None,offline=False,*,prediction):
        assert prediction=='qual-v2' and offline is True
        return checkpoint(path,offline=offline,prediction=prediction)
    monkeypatch.setattr(models,'checkpoint',check)
    args=['--prediction',prediction,'--lasagna-input',str(tmp_path/'import.lasagna.json')]
    if prediction=='qual-v2':args+=['--model-path',str(weights)]
    code,result=doctor(capsys,*args)
    checks=result['preflight']['checks']
    assert code==0 and result['preflight']['ready'] is True
    assert checks['lasagna_runtime']['status']==checks['lasagna_checkpoint']['status']=='NOT_REQUIRED'
    assert checks['fibre_checkpoint']['status']==('READY' if prediction=='qual-v2' else 'NOT_REQUIRED')


def test_external_fibres_with_imported_surfaces_require_no_inference_dependencies(capsys,readonly_models,monkeypatch,tmp_path):
    models,lasagna=readonly_models
    def forbidden(*args,**kwargs):raise AssertionError('imported fibres and surfaces must not check models')
    monkeypatch.setattr(lasagna,'runtime_identity',forbidden)
    monkeypatch.setattr(models,'checkpoint',forbidden)
    code,result=doctor(capsys,'--prediction','qual-v2','--fibres',tmp_path/'fibres.afv',
                       '--lasagna-input',tmp_path/'input.lasagna.json')
    assert code==0 and result['preflight']['ready'] is True
    assert all(check['status']=='NOT_REQUIRED' for check in result['preflight']['checks'].values())
