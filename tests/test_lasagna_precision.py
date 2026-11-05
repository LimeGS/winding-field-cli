"""The worker must not request unsupported BF16 on the selected GPU."""

from contextlib import contextmanager

import pytest
import torch

from winding_field.lasagna_worker import run_tile_inference


@pytest.mark.parametrize('capability', [(7,5),(8,0)])
def test_selected_cuda_device_precision(monkeypatch,capability):
    selected=torch.device('cuda:1')
    def query(device):
        assert device==selected
        return capability
    monkeypatch.setattr(torch.cuda,'get_device_capability',query)
    @contextmanager
    def autocast(*,device_type,dtype,enabled):
        if enabled and capability[0]<8:
            raise RuntimeError('BF16 unsupported on GTX1660')
        yield
    monkeypatch.setattr(torch,'autocast',autocast)
    # CUDA facilities are mocked on this CPU-only test host; actual worker/model flow remains real.
    data=torch.tensor([2.])
    result=run_tile_inference(None,lambda x:{'output':x*3},data,device=selected)
    assert torch.equal(result,torch.tensor([6.]))
