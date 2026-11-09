import os
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
