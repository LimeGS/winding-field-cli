import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('case',['solve','blocks'])
def test_solver_benchmark_records_cold_warm_phases_and_fields(tmp_path,case):
    script=Path(__file__).resolve().parents[1]/'tools/bench_solver.py'
    output=tmp_path/'evidence'
    command=[sys.executable,str(script),'--output',str(output),'--size','4',
             '--repeats','2','--profile','--case',case]
    run=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stderr
    report=json.loads((output/'report.json').read_text())
    assert len(report['measurements'])==2
    assert report['measurements'][0]['temperature']=='cold'
    assert report['measurements'][1]['temperature']=='warm'
    assert all(r['solver']['converged'] for r in report['measurements'])
    assert report['phases']['gradient_operator']['calls']>0
    assert report['phases']['reductions']['seconds']>0
    with np.load(output/'fields.npz') as fields:
        np.testing.assert_array_equal(fields['field_0'],fields['field_1'])
    again=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert again.returncode!=0  # Never overwrite evidence.
