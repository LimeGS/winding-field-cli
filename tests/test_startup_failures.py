import json
from pathlib import Path

import numpy as np
import pytest


def source_config(tmp_path):
    from ct_fixture import write_ct
    source=write_ct(tmp_path/'ct',np.ones((8,)*3,np.uint8),spacing=(9.,)*3)
    return dict(source=str(source),output=str(tmp_path/'run'),prediction='qual-v2',
                lasagna_python=str(tmp_path/'missing-python'),reserve_bytes=0,
                cache=str(tmp_path/'cache'),threads=1)


def assert_failed(output, phrase):
    state=json.loads((output/'RUN.json').read_bytes())
    assert state['schema']=='winding-field-run-v1'
    assert state['status']=='FAILED' and phrase in state['error']
    assert state['stages']=={} and state['physical_acceptance'] is False
    assert not (output/'PREPARED.json').exists()
    return state


def test_cli_missing_runtime_has_failure_receipt_and_nonzero_exit(tmp_path,capsys):
    from winding_field.cli import main
    config=source_config(tmp_path)
    assert main(['run','--source',config['source'],'--output',config['output'],
                 '--lasagna-python',config['lasagna_python'],'--reserve-gib','0'])==1
    assert 'Lasagna Python runtime missing' in capsys.readouterr().err
    state=assert_failed(tmp_path/'run','Lasagna Python runtime missing')
    assert state['config']['lasagna_python']==config['lasagna_python']


@pytest.mark.parametrize('failure', ['source','plan','checkpoint'])
def test_startup_io_and_dependency_errors_use_existing_receipt_contract(tmp_path,monkeypatch,failure):
    from winding_field import workflow,lasagna
    config=source_config(tmp_path)
    def fail(*args,**kwargs):raise OSError('startup '+failure+' unavailable')
    if failure=='source':monkeypatch.setattr(workflow,'open_volume',fail)
    elif failure=='plan':monkeypatch.setattr(workflow,'plan',fail)
    else:
        monkeypatch.setattr(lasagna,'runtime_identity',lambda python=None:dict(executable=python))
        monkeypatch.setattr(lasagna,'checkpoint',fail)
    with pytest.raises(OSError,match='startup '+failure):workflow.run(config,until='prepare')
    assert_failed(tmp_path/'run','startup '+failure)


@pytest.mark.parametrize('kind', ['empty','populated','file','dangling-symlink'])
def test_existing_outputs_are_never_adopted_or_replaced(tmp_path,capsys,kind):
    from winding_field.cli import main
    config=source_config(tmp_path);root=tmp_path/'run'
    if kind=='file':root.write_bytes(b'user data')
    elif kind=='dangling-symlink':root.symlink_to(tmp_path/'absent-target',target_is_directory=True)
    else:
        root.mkdir()
        if kind=='populated':(root/'RUN.json').write_bytes(b'existing receipt')
    before={p.name:p.read_bytes() for p in root.iterdir()} if root.is_dir() else None
    assert main(['run','--source',config['source'],'--output',str(root),
                 '--lasagna-python',config['lasagna_python'],'--reserve-gib','0'])==1
    assert 'output' in capsys.readouterr().err
    if before is not None:assert {p.name:p.read_bytes() for p in root.iterdir()}==before
    elif kind=='file':assert root.read_bytes()==b'user data'
    else:
        assert root.is_symlink() and not (tmp_path/'absent-target').exists()


def test_uncreatable_output_reports_stderr_without_a_receipt(tmp_path,capsys):
    from winding_field.cli import main
    config=source_config(tmp_path);parent=tmp_path/'not-a-directory';parent.write_bytes(b'keep')
    assert main(['run','--source',config['source'],'--output',str(parent/'run'),
                 '--lasagna-python',config['lasagna_python'],'--reserve-gib','0'])==1
    assert capsys.readouterr().err and parent.read_bytes()==b'keep'


@pytest.mark.parametrize('existing', [False,True])
def test_resume_without_receipt_never_creates_or_adopts_output(tmp_path,existing):
    from winding_field.workflow import run
    config=source_config(tmp_path);root=tmp_path/'run'
    if existing:root.mkdir()
    with pytest.raises(FileNotFoundError):run(config,resume=True)
    assert list(root.iterdir())==[] if existing else not root.exists()


def test_inference_failures_still_record_failed_stage_and_preserve_resume_identity(tmp_path,monkeypatch):
    from winding_field import workflow,lasagna,predict
    config=source_config(tmp_path)
    monkeypatch.setattr(lasagna,'runtime_identity',lambda python=None:dict(executable=python))
    monkeypatch.setattr(lasagna,'checkpoint',lambda *a,**kw:tmp_path/'weights.pt')
    def fail(*args,**kwargs):raise RuntimeError('inference worker failed')
    monkeypatch.setattr(predict,'prepare',fail)
    with pytest.raises(RuntimeError,match='inference worker'):workflow.run(config,until='prepare')
    state=assert_failed(tmp_path/'run','inference worker')
    assert state['active_stage']=='prepare'
    before=(tmp_path/'run/RUN.json').read_bytes()
    with pytest.raises(ValueError,match='configuration changed'):
        workflow.run(dict(state['config'],device='changed'),resume=True)
    assert (tmp_path/'run/RUN.json').read_bytes()==before
