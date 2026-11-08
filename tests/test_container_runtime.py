import os
from pathlib import Path
import subprocess
import sys

import pytest


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
