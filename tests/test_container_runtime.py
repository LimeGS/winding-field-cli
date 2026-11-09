import os
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(not Path('/opt/runtime-provenance/runtime.json').is_file(),
                                reason='installed Docker runtime qualification')


def test_development_profiler_is_not_shipped():
    assert not Path('/opt/nvidia/nsight-compute').exists()
    assert shutil.which('ncu') is None


def test_default_runtime_paths_allow_arbitrary_container_uids():
    for name in ['/home/worker', '/home/worker/.cache', '/home/worker/.cache/winding-field']:
        assert Path(name).stat().st_mode & 0o001, f'arbitrary UID cannot traverse {name}'


def test_cuda_toolkit_keeps_original_nvidia_license_notice():
    import torch
    if torch.version.cuda is None:
        pytest.skip('CPU image has no CUDA toolkit')
    notice = Path('/NGC-DL-CONTAINER-LICENSE').read_bytes()
    assert hashlib.sha256(notice).hexdigest() == 'e4196076c5496c4bb5509be61e3d1cddf36b92a449a10ece1779afce3c65e684'


def test_native_supervisor_can_measure_real_process_rss():
    from types import SimpleNamespace
    from winding_field.native_pipeline import _rss
    assert _rss(SimpleNamespace(pid=os.getpid())) > 0


def test_cuda_environments_share_identical_nvidia_libraries():
    import torch
    if torch.version.cuda is None:
        pytest.skip('CPU image has no NVIDIA wheel libraries')
    relative = 'site-packages/nvidia/cublas/lib/libcublas.so.12'
    cli = Path('/opt/winding-field/lib/python3.12') / relative
    lasagna = Path('/opt/lasagna/lib/python3.14') / relative
    assert cli.samefile(lasagna), 'identical CUDA libraries should be stored once'


@pytest.mark.parametrize('environment', ['winding-field', 'lasagna'])
def test_both_dependency_environments_remain_consistent(environment):
    subprocess.run(['uv', 'pip', 'check', '--python', f'/opt/{environment}/bin/python'], check=True)


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
def test_explicit_native_accelerator_rebuilding_is_preserved(tmp_path, backend):
    import torch
    if backend == 'cuda' and torch.version.cuda is None:
        pytest.skip('CPU image has no CUDA compiler')
    output = tmp_path / backend
    command = [sys.executable, '-I', '-m', 'winding_field.tifxyz_build',
               '--backend', backend, '--output', str(output)]
    if backend == 'cuda':
        command += ['--host-compiler', '/usr/bin/g++']
    subprocess.run(command, check=True, timeout=180)
    assert (output / 'accelerator.json').is_file()
    assert (output / 'predicates.so').is_file()
    if backend == 'cpu' or torch.cuda.is_available():
        env = dict(os.environ, WINDING_TIFXYZ_ACCELERATOR=str(output / 'accelerator.json'))
        if backend == 'cuda':
            uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                            '--format=csv,noheader'], text=True).splitlines()[0]
            env.update(CUDA_VISIBLE_DEVICES=uuid, WINDING_TIFXYZ_CUDA_DEVICE=uuid)
        subprocess.run([sys.executable, '-I', '-c', '''
import sys,numpy as np
from winding_field.tifxyz_predicates import Backend
rows=np.zeros((256,24),np.float64)
rows[:,6]=1;rows[:,13:15]=.25;rows[:,15]=1
with Backend(sys.argv[1]) as backend:
    assert (backend.classify(0,rows)==1).all()
''', backend], env=env, check=True, timeout=60)


def test_lasagna_environment_cuda_jit_executes_from_fresh_writable_cache(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('real GPU required for Lasagna JIT qualification')
    env = dict(os.environ, TORCH_EXTENSIONS_DIR=str(tmp_path / 'jit'),
               MAX_JOBS='1', TORCH_CUDA_ARCH_LIST='.'.join(map(str, torch.cuda.get_device_capability())))
    subprocess.run(['/opt/lasagna/bin/python', '-I', '-c', '''
import sys
from pathlib import Path
from torch.utils.cpp_extension import load
root=Path(sys.argv[1]);root.mkdir()
(root/'probe.cpp').write_text("""
#include <pybind11/pybind11.h>
int probe();
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("probe",&probe);}
""")
(root/'probe.cu').write_text("""
#include <cuda_runtime.h>
__global__ void fill(int* p){p[0]=17;}
int probe(){
    int* p=nullptr;int value=0;
    if(cudaMalloc(&p,sizeof(int))!=cudaSuccess)return -1;
    fill<<<1,1>>>(p);
    auto status=cudaMemcpy(&value,p,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(p);
    return status==cudaSuccess?value:-2;
}
""")
module=load(name='wf_runtime_probe',sources=[str(root/'probe.cpp'),str(root/'probe.cu')])
assert module.probe()==17
''', str(tmp_path / 'source')], env=env, check=True, timeout=600)


@pytest.mark.skipif(not Path('/opt/runtime-provenance/runtime.json').is_file(),
                    reason='installed Docker runtime qualification')
def test_profiler_works_without_a_container_passwd_entry():
    env=dict(os.environ)
    for name in ('LOGNAME','USER','LNAME','USERNAME'):env.pop(name,None)
    result=subprocess.run([sys.executable,'-I','-c','''
import pwd
def missing_user(uid):
    raise KeyError('no container passwd entry')
pwd.getpwuid=missing_user
import torch
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
    assert torch.ones(4).sum().item()==4
assert profile.events()
'''],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
