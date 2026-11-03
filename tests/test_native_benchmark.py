import json
from pathlib import Path
import subprocess
import sys
import shutil


def test_benchmark_executes_real_mesh_render_and_format_paths(tmp_path):
    script=Path(__file__).resolve().parents[1]/'tools/bench_native_surfaces.py'
    out=tmp_path/'bench'
    process=subprocess.run([sys.executable,'-B',str(script),'--output',str(out),'--seconds','30',
                            '--chart-faces','128','--export-tifxyz'],capture_output=True,text=True,timeout=45)
    assert process.returncode==0,process.stdout+process.stderr
    result=json.loads((out/'RESULT.json').read_bytes())
    assert result['status'] in ('COMPLETE','PARTIAL'),result
    assert result['faces']==1426
    assert result['whole_level'] is False
    assert result['usable_render']
    assert list((out/'level-0000').glob('component-*/sheet.png'))
    assert result['mesh_seconds']>0
    again=subprocess.run([sys.executable,'-B',str(script),'--output',str(out),'--seconds','30',
                          '--chart-faces','128','--export-tifxyz','--resume'],capture_output=True,text=True,timeout=45)
    assert again.returncode==0,again.stdout+again.stderr
    resumed=json.loads((out/'RESULT.json').read_bytes())
    assert resumed['timing_scope']=='existing-products-verification'
    scripts=tmp_path/'changed-tools'; scripts.mkdir()
    shutil.copyfile(script,scripts/script.name)
    helper=script.with_name('native_historical_source.py')
    (scripts/helper.name).write_bytes(helper.read_bytes()+b'\n# changed adapter runtime\n')
    before=(out/'RESULT.json').read_bytes()
    changed=subprocess.run([sys.executable,'-B',str(scripts/script.name),'--output',str(out),'--seconds','30',
                            '--chart-faces','128','--export-tifxyz','--resume'],capture_output=True,text=True,timeout=45)
    assert changed.returncode!=0 and 'request changed' in changed.stderr
    assert (out/'RESULT.json').read_bytes()==before
