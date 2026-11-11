import json

import pytest
import torch


@pytest.mark.parametrize('cuda,mps,unsupported,want',[
    (True,True,(), 'cuda'),(False,True,(), 'mps'),(False,False,(), 'cpu'),
    (True,True,('cuda',), 'mps'),(True,True,('cuda','mps'), 'cpu'),
    (False,True,('mps',), 'cpu'),
])
def test_auto_device_priority_requires_working_float64_norm(monkeypatch,cuda,mps,unsupported,want):
    from winding_field.cli import resolve_device
    monkeypatch.setattr(torch.cuda,'is_available',lambda:cuda)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:mps)
    ones=torch.ones;norm=torch.linalg.vector_norm
    def allocate(*args,device,**kwargs):
        tensor=ones(*args,device='cpu',**kwargs)
        tensor.requested_device=device
        return tensor
    def reduce(tensor,*,dtype):
        assert dtype==torch.float64
        if tensor.requested_device in unsupported:
            raise (RuntimeError if tensor.requested_device=='cuda' else TypeError)('backend cannot execute FP64 norm')
        return norm(tensor,dtype=dtype)
    monkeypatch.setattr(torch,'ones',allocate)
    monkeypatch.setattr(torch.linalg,'vector_norm',reduce)
    assert resolve_device('auto')==want


def test_real_auto_device_can_execute_the_core_float64_norm():
    from winding_field.cli import resolve_device
    device=resolve_device('auto')
    assert device in ('cuda','mps','cpu')
    assert float(torch.linalg.vector_norm(torch.ones(1,device=device),dtype=torch.float64))==1.


@pytest.mark.parametrize('device',['cpu','cuda:2','mps'])
def test_explicit_device_is_not_probed_or_replaced(monkeypatch,device):
    from winding_field.cli import resolve_device
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('explicit device was probed'))
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:pytest.fail('explicit device was probed'))
    assert resolve_device(device)==device


@pytest.mark.parametrize('command',['run','plan'])
@pytest.mark.parametrize('requested',[None,'auto','cpu','cuda:2','mps'])
def test_new_cli_freezes_auto_device_before_execution(tmp_path,monkeypatch,command,requested):
    from winding_field.cli import main
    from winding_field.util import atomic_json
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    ones=torch.ones
    monkeypatch.setattr(torch,'ones',lambda *args,device,**kwargs:ones(*args,device='cpu',**kwargs))
    def capture(config,**kwargs):
        atomic_json(tmp_path/'captured.json',config)
        return dict(status='PAUSED')
    monkeypatch.setattr('winding_field.workflow.run',capture)
    monkeypatch.setattr('winding_field.workflow.plan',capture)
    args=[command,'--source','unused','--output',str(tmp_path/'out')]
    if requested:args+=['--device',requested]
    assert main(args)==0
    saved=json.loads((tmp_path/'captured.json').read_bytes())
    assert saved['device']==('cuda' if requested in (None,'auto') else requested)
    assert saved['device']!='auto'


@pytest.mark.parametrize('device',[None,'cpu','cuda:2','mps'])
def test_resume_preserves_saved_devices_without_reselection(tmp_path,monkeypatch,device):
    from winding_field.cli import main
    from winding_field.util import atomic_json
    config=dict(output=str(tmp_path),surface_output='native')
    if device:config['device']=device
    atomic_json(tmp_path/'RUN.json',dict(config=config,status='PAUSED'))
    before=(tmp_path/'RUN.json').read_bytes()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('resume reselected a device'))
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:pytest.fail('resume reselected a device'))
    def resume(saved,**kwargs):
        assert saved==config and kwargs['resume'] is True
        return dict(config=saved,status='PAUSED')
    monkeypatch.setattr('winding_field.workflow.run',resume)
    assert main(['resume',str(tmp_path)])==0
    assert (tmp_path/'RUN.json').read_bytes()==before
