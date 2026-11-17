import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from test_lasagna_input import source_case
from test_splines import upstream_curves
from winding_field.util import file_sha


def benchmark():
    path=Path(__file__).resolve().parents[1]/'tools/bench_local_imports.py'
    assert path.is_file(), 'local import benchmark missing'
    spec=importlib.util.spec_from_file_location('bench_local_imports',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def local_case(root, *, physical=True, count=8):
    root.mkdir()
    if physical:
        volume,_,config=source_case(root,shape=(8,8,16),whole=True)
        volume.close();shape=(8,8,16)
    else:
        from test_lasagna_stream import fixture
        ct,_,config=fixture(root);shape=(9,10,11);config['source']=str(ct)
        (root/'geometry.json').write_text(json.dumps(dict(source=str(ct),
            origin_xyz_um=[100.,200.,300.],spacing_xyz_um=[5.,10.,20.])))
    fibres=root/'fibres.afv'
    upstream_curves(fibres,shape,[(1,np.array([[1.,2.,3.],[float(shape[2]-2),2.,3.]]))]*count)
    config.update(fibres=str(fibres),fibres_geometry=str(root/'geometry.json'),chunk=4,
                  threads=1,prediction='qual-v2',cache_bytes=1<<20)
    path=root/'config.json';path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize('physical',[False,True])
def test_benchmark_records_stage_costs_and_preserves_inputs(tmp_path,physical):
    config=local_case(tmp_path/'input',physical=physical)
    def snapshot():
        return {str(p):(file_sha(p),p.stat().st_mtime_ns) for p in (tmp_path/'input').rglob('*')
                if p.is_file() and 'cache' not in p.parts}
    before=snapshot()
    result=benchmark().run(tmp_path/'benchmark',config=config,repeats=1,workers=(1,),profile=True)
    assert before==snapshot()
    assert result['logical_results_equal']
    assert len(result['measurements'])==1
    record=result['measurements'][0]
    assert record['curves']>0 and record['output_bytes']>0 and record['peak_rss_bytes']>0
    assert record['decodes']>0 and record['support_loads']>0 and record['commits']>0
    assert record['hashed_bytes']>0
    assert all(record['stage_seconds'][name]>0 for name in ('prepare','fibres','export_lasagna','export_afv'))
    assert result['profile']['profiled'] and not record['profiled']
    assert result['profile']['pid']!=record['pid']
    assert result['runtime']['core_version']=='1.5.2'


@pytest.mark.parametrize('overlap',['existing','inside','contains'])
def test_benchmark_rejects_existing_or_overlapping_output(tmp_path,overlap):
    config=local_case(tmp_path/'input')
    output={'existing':tmp_path/'input','inside':tmp_path/'input/ct/bench','contains':tmp_path}[overlap]
    with pytest.raises((ValueError,FileExistsError)):
        benchmark().run(output,config=config,repeats=1)
    assert not (tmp_path/'input/ct/bench').exists()
